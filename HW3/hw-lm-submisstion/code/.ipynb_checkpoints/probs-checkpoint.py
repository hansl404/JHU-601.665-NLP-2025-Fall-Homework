#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu> 
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations

import logging
import math
import pickle
import sys

from pathlib import Path

import torch
from torch import nn
from torch import optim
from jaxtyping import Float
from typeguard import typechecked
from typing import Counter, Collection
from collections import Counter
import torch.nn.functional as F
import torch.optim as optim
# from tdqm import tdqm

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Collection[Wordtype]   # and change this to Integerizer[str]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]
TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process(token)
    # Whenever the `for` loop needs another token, read_tokens magically picks up 
    # where it left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    # Convert from an unordered Set to an ordered List.  This ensures that iterating
    # over the vocab will always hit the words in the same order, so that you can 
    # safely store a list or tensor of embeddings in that order, for example.
    return sorted(vocab)   
    # Alternatively, you could choose to represent a Vocab as an Integerizer (see above).
    # Then you won't need to sort, since Integerizers already have a stable iteration order.

##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the 
    # denominator and c(yz) for the backed-off numerator.  Both of these 
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    # 
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    # 
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.  
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly, 
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    def save(self, model_path: Path) -> None:
        log.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            # torch.save is similar to pickle.dump but handles tensors too
        log.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> "LanguageModel":
        log.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
            # torch.load is similar to pickle.load but handles tensors too
            # map_location allows loading tensors on different device than saved
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from file {model_path}")
        log.info(f"Loaded model from {model_path}")
        return model

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")

    def sample(self, max_length: int = 20) -> list[Wordtype]:
        x, y = BOS, BOS
        sentence: list[Wordtype] = []

        for _ in range(max_length):
            words = list(self.vocab)
            probs = torch.tensor([self.prob(x, y, z) for z in words], dtype=torch.float)

            probs = probs / probs.sum()
            z_idx = torch.multinomial(probs, 1).item()
            z = words[z_idx]

            if z == EOS:
                break
            sentence.append(z)
            x, y = y, z

        return sentence


##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )

class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size

class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.

class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Add-lambda with backoff smoothing: p(z | x,y)"""
        return self._prob_trigram(x, y, z)

    def _prob_trigram(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # c(x,y,z) / c(x,y)
        num = self.event_count[(x, y, z)] + self.lambda_ * self.vocab_size * self._prob_bigram(y, z)
        denom = self.context_count[(x, y)] + self.lambda_ * self.vocab_size
        return num / denom if denom > 0 else 1.0 / self.vocab_size

    def _prob_bigram(self, y: Wordtype, z: Wordtype) -> float:
        # c(y,z) / c(y)
        num = self.event_count[(y, z)] + self.lambda_ * self.vocab_size * self._prob_unigram(z)
        denom = self.context_count[(y,)] + self.lambda_ * self.vocab_size
        return num / denom if denom > 0 else 1.0 / self.vocab_size

    def _prob_unigram(self, z: Wordtype) -> float:
        # c(z) / N
        num = self.event_count[(z,)] + self.lambda_
        denom = self.event_count[()] + self.lambda_ * self.vocab_size
        return num / denom

class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab)
        if l2 < 0:
            raise ValueError("Negative regularization strength {l2}")
        self.l2: float = l2

        # TODO: ADD CODE TO READ THE LEXICON OF WORD VECTORS AND STORE IT IN A USEFUL FORMAT.
        # self.dim: int = 99999999999  
        # # TODO: SET THIS TO THE DIMENSIONALITY OF THE VECTORS

        self.epochs = epochs
        self.N_train_tokens: int = -1
        # lexicon is mapping of each word in vocab to a vector embedding
        lexicon: dict[str, torch.Tensor] = {}
        with open(lexicon_file, "r", encoding="utf-8") as f:
            header = f.readline().strip().split()   # header is like 75 10 (# vocab, embed len)
            vocab_size = int(header[0])
            self.dim = int(header[1])

            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector_values = [float(x) for x in parts[1:]]
                lexicon[word] = torch.tensor(vector_values, dtype=torch.float32)
        
        self.vocab_list_custom = list(self.vocab)
        self.vocab_size_custom = len(self.vocab_list_custom)
        self.word2idx_custom = {word: i for i, word in enumerate(self.vocab_list_custom)}
        self.idx2word_custom = {i: word for i, word in enumerate(self.vocab_list_custom)}
       
        self.ool_embedding = lexicon.get(OOL, torch.randn(self.dim)*0.01)   # lexicon contains an OOL embedding

        embedding_matrix = torch.zeros((self.vocab_size_custom, self.dim))  #initialize with 0
        for word,idx in self.word2idx_custom.items():
            vec = lexicon.get(word, self.ool_embedding)  # embedding vector = torch.randn(self.dim)*0.01 for now
            embedding_matrix[idx] = vec

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True) # turns into PyTorch module that model can use?
        # We wrap the following matrices in nn.Parameter objects.
        # This lets PyTorch know that these are parameters of the model
        # that should be listed in self.parameters() and will be
        # updated during training.
        #
        # We can also store other tensors in the model class,
        # like constant coefficients that shouldn't be altered by
        # training, but those wouldn't use nn.Parameter.
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)    # dimensions (dim x dim)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)    # dimensions (dim x dim)
        
        # self.W_out = nn.Parameter(torch.empty(self.dim, self.vocab_size_custom), requires_grad=True)   # dimensions (dim x vocab)
        # # 7: params will be stored in X and Y matrices, start off with 0
        # nn.init.xavier_uniform_(self.X)
        # nn.init.xavier_uniform_(self.Y)
        nn.init.zeros_(self.X)
        nn.init.zeros_(self.Y)
        
        # nn.init.xavier_uniform_(self.W_out)

    # -------------------------------------------------------------------------
    # === Core logits function ===
    # -------------------------------------------------------------------------
    @typechecked
    def logits(self, x: Wordtype, y: Wordtype, z: Wordtype) -> torch.Tensor:
        device = self.X.device
        x_idx = torch.tensor(self.word2idx_custom.get(x, 0), dtype=torch.long, device=device)
        y_idx = torch.tensor(self.word2idx_custom.get(y, 0), dtype=torch.long, device=device)
        z_idx = torch.tensor(self.word2idx_custom.get(z, 0), dtype=torch.long, device=device)

        x_vec = self.embedding(x_idx)
        y_vec = self.embedding(y_idx)
        z_vec = self.embedding(z_idx)

        x_part = torch.dot(x_vec @ self.X, z_vec)
        y_part = torch.dot(y_vec @ self.Y, z_vec)
        return x_part + y_part

    # -------------------------------------------------------------------------
    # === Probability computation ===
    # -------------------------------------------------------------------------
    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return self.log_prob_tensor(x, y, z).item()

    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> torch.Tensor:
        """
        Efficient vectorized version:
            log p(z | x, y) = logits(x, y)[z] - logsumexp(logits(x, y))
        Only one forward pass over the vocabulary (no redundant computation).
        """

        device = self.X.device

        x_idx = torch.tensor(self.word2idx_custom.get(x, 0), dtype=torch.long, device=device)
        y_idx = torch.tensor(self.word2idx_custom.get(y, 0), dtype=torch.long, device=device)
        x_vec = self.embedding(x_idx)  # shape: (dim,)
        y_vec = self.embedding(y_idx)  # shape: (dim,)

        vocab_indices = torch.arange(len(self.vocab), device=device)
        embed_mat = self.embedding(vocab_indices)  # shape: (vocab, dim)

        logits_all = (x_vec @ self.X @ embed_mat.T) + (y_vec @ self.Y @ embed_mat.T)
        logZ = torch.logsumexp(logits_all, dim=-1)  # scalar

        z_idx = torch.tensor(self.word2idx_custom.get(z, 0), dtype=torch.long, device=device)
        logit_target = logits_all[z_idx]  # scalar

        return logit_target - logZ


    # -------------------------------------------------------------------------
    # === Training ===
    # -------------------------------------------------------------------------
    # def train(self, file: Path):  # type: ignore
    #     """
    #     Train model using SGD on L2-regularized negative log-likelihood.
    #     """
    #     eta0 = 0.01
    #     optimizer = optim.SGD(self.parameters(), lr=eta0)
    #     # nn.init.zeros_(self.X)
    #     # nn.init.zeros_(self.Y)

    #     corpus = list(read_trigrams(file))
    #     self.N_train_tokens = len(corpus)

    #     for epoch in range(self.epochs):
    #         total_loss = 0.0
    #         for (x, y, z) in corpus:
    #             optimizer.zero_grad()
    #             # Negative log-likelihood = -log p(z | x, y)
    #             logp = self.log_prob(x, y, z)
    #             loss = -logp + self.l2 * (torch.sum(self.X**2) + torch.sum(self.Y**2))
    #             print(loss)
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
    #         print(f"epoch {epoch+1}: F = {total_loss/self.N_train_tokens:.6f}")
    #     print(f"Finished training on {self.N_train_tokens} tokens.")

    def train(self, file: Path):    # type: ignore
        
        ### Technically this method shouldn't be called `train`,
        ### because this means it overrides not only `LanguageModel.train` (as desired)
        ### but also `nn.Module.train` (which has a different type). 
        ### However, we won't be trying to use the latter method.
        ### The `type: ignore` comment above tells the type checker to ignore this inconsistency.
        
        # Optimization hyperparameters.
        eta0 = 0.00001 # initial learning rate

        # This is why we needed the nn.Parameter above.
        # The optimizer needs to know the list of parameters
        # it should be trying to update.
        optimizer = optim.SGD(self.parameters(), lr=eta0)

        # Initialize the parameter matrices to be full of zeros.
        # nn.init.zeros_(self.X)   # type: ignore
        # nn.init.zeros_(self.Y)   # type: ignore

        N = num_tokens(file)
        self.N_train_tokens = N
        log.info("Start optimizing on {N} training tokens...")

        #####################
        # TODO: Implement your SGD here by taking gradient steps on a sequence
        # of training examples.  Here's how to use PyTorch to make it easy:
        #
        # To get the training examples, you can use the `read_trigrams` function
        # we provided, which will iterate over all N trigrams in the training
        # corpus.  (Its use is illustrated in fileprob.py.)
        #
        for epoch in range(1, self.epochs+1):
            total_loss = 0.0
            trigram_count = 0
            # For each successive training example i, 
            print("total", num_tokens(file))
            for x,y,z in read_trigrams(file,self.vocab):
                optimizer.zero_grad()
                logp = self.log_prob_tensor(x, y, z)
                # print(logp)
                loss = -logp + self.l2 * (torch.sum(self.X**2) + torch.sum(self.Y**2)) / self.N_train_tokens
                # print(loss)
                # compute the stochastic objective F_i(θ).  This is called the "forward" computation. Don't
                # forget to include the regularization term. Part of F_i(θ) will be the
                # log probability of training example i, which the helper method
                # log_prob_tensor computes.  It is important to use log_prob_tensor
                # (as opposed to log_prob which returns float) because torch.Tensor
                # is an object with additional bookkeeping that tracks e.g. the gradient
                # function for backpropagation as well as accumulated gradient values
                # from backpropagation.
                # loss_val = self.loss(x,y,z)  # log_prob_tensor used in loss fn above

                # To get the gradient of this objective (∇F_i(θ)), call the `backward`
                # method on the number you computed at the previous step.  This invokes
                # back-propagation to get the gradient of this number with respect to
                # the parameters θ.  This should be easier than implementing the
                # gradient method from the handout.
                loss.backward()

                # Finally, update the parameters in the direction of the gradient, as
                # shown in Algorithm 1 in the reading handout.  You can do this `+=`
                # yourself, or you can call the `step` method of the `optimizer` object
                # we created above.  See the reading handout for more details on this.

                optimizer.step()

                total_loss += loss.item()
                trigram_count += 1
                # print(trigram_count)
                if trigram_count%10000 ==0 :
                    print(f"checkpoint at {trigram_count}")
            
                # progress_bar.set_postfix(loss=loss_val.item()) # update progress bar
            F = -(total_loss / trigram_count)
            print(f"epoch {epoch}: F = {F:.10f}")   # format like in example
        
        print(f"Finished training on {N} tokens")
        # For the EmbeddingLogLinearLanguageModel, you should run SGD
        # optimization for the given number of epochs and then stop.  You might 
        # want to print progress dots using the `show_progress` method defined above.  
        # Even better, you could show a graphical progress bar using the tqdm module --
        # simply iterate over
        #     tqdm.tqdm(read_trigrams(file), total=N*epochs)
        # instead of iterating over
        #     read_trigrams(file)
        #####################

        log.info("done optimizing.")

        # So how does the `backward` method work?
        #
        # As Python sees it, your parameters and the values that you compute
        # from them are not actually numbers.  They are `torch.Tensor` objects.
        # A Tensor may represent a numeric scalar, vector, matrix, etc.
        #
        # Every Tensor knows how it was computed.  For example, if you write `a
        # = b + exp(c)`, PyTorch not only computes `a` but also stores
        # backpointers in `a` that remember how the numeric value of `a` depends
        # on the numeric values of `b` and `c`.  In turn, `b` and `c` have their
        # own backpointers that remember what they depend on, and so on, all the
        # way back to the parameters.  This is just like the backpointers in
        # parsing!
        #
        # Every Tensor has a `backward` method that computes the gradient of its
        # numeric value with respect to the parameters, using "back-propagation"
        # through this computation graph.  In particular, once you've computed
        # the forward quantity F_i(θ) as a tensor, you can trace backwards to
        # get its gradient -- i.e., to find out how rapidly it would change if
        # each parameter were changed slightly.
  
class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab, lexicon_file, l2, epochs)

        self.rep_weight = nn.Parameter(torch.tensor(-0.5))
        self.skip_weight = nn.Parameter(torch.tensor(0.3))
        self.spell_weight = nn.Parameter(torch.tensor(0.4))

    def spelling_features(self, word: str) -> float:
        """Compute hand-crafted spelling-based features (not trainable)."""
        score = 0.0
        if len(word) == 0:
            return score
        if word[0].isupper():
            score += 1.0
        if word.endswith("ing"):
            score += 1.0
        if word.endswith("ly"):
            score += 1.0
        if any(ch.isdigit() for ch in word):
            score += 1.0
        if word in {".", "!", "?"}:
            score += 1.0
        return score

    @typechecked
    def logits(self, x: Wordtype, y: Wordtype, z: Wordtype):
        """Compute vectorized logits with repetition, skip-bigram, and spelling features."""
        device = self.X.device

        # --- context embeddings ---
        x_idx = torch.tensor(self.word2idx_custom.get(x, 0), dtype=torch.long, device=device)
        y_idx = torch.tensor(self.word2idx_custom.get(y, 0), dtype=torch.long, device=device)
        x_vec = self.embedding(x_idx)
        y_vec = self.embedding(y_idx)

        # --- base logits using X and Y (保持与父类一致结构) ---
        vocab_indices = torch.arange(self.vocab_size_custom, device=device)
        embed_mat = self.embedding(vocab_indices)  # shape: (vocab, dim)

        base_logits = (x_vec @ self.X @ embed_mat.T) + (y_vec @ self.Y @ embed_mat.T)

        # --- repetition feature ---
        rep_mask = torch.zeros(self.vocab_size_custom, device=device)
        for w in (x, y):
            if w in self.word2idx_custom:
                rep_mask[self.word2idx_custom[w]] = 1.0
        rep_term = self.rep_weight * rep_mask

        # --- skip-bigram cosine similarity ---
        skip_sim = F.cosine_similarity(x_vec.unsqueeze(0), embed_mat, dim=1)
        skip_term = self.skip_weight * skip_sim

        # --- spelling features ---
        spell_scores = torch.tensor(
            [self.spelling_features(w) for w in self.vocab_list_custom],
            dtype=torch.float32,
            device=device,
        )
        spell_term = self.spell_weight * spell_scores

        logits = base_logits + rep_term + skip_term + spell_term
        return logits
