# NLP Homework 7: Neuralization

## Setup and Files

As in previous homeworks, you can activate the environment using 

    conda activate nlp-class

In this homework, you'll extend your Conditional Random Field tagger from the previous homework.  First, you will enable it to use contextual features that an HMM cannot use.  Second, you will neuralize it, which means training a neural net to extract the features.  The neural net will have access to a lexicon of word embeddings.

You should begin by copying your HW6 directory as your starting point for HW7.
Add the `.py` files found in the `hw-rnn` folder:

* `lexicon.py` -- lets you load a lexicon of word embeddings from a file
* `crf_backprop.py` - **parameter initialization, gradient zeroing and accumulation, gradient step**
* `crf_test.py` - try out **non-stationary potentials**
* `crf_neural.py` - **parameter initialization, neuralized non-stationary potentials**
* `logsumexp_safe.py` -- patches PyTorch so that backprop can deal with `-inf`
  * _Note:_ only needed if your forward algorithm works in logspace instead of using the scaling trick
* `tag.py` -- extends the version from HW6 to call the new classes in `crf_*.py` above

**Boldface** above indicates methods you will write. 
As with HW6, it will be helpful to experiment with small examples of the classes in Jupyter notebooks.  We have not given you a new notebook, but you can build on the previous ones.

## Implementation goal

Use the `--help` option to see the options you'll support.  You'll work up to being able to run training commands like this:

    python3 tag.py endev --train ensup --crf --reg 0.1 --lr 0.01 --rnn_dim 8 --lexicon words-10.txt --model birnn_8_10.pkl

You can then test your trained model on any input file (here we are testing on the training data, which will obviously have better performance):

    python3 tag.py ensup --model basic_test.pkl

These commands also create output files.

## Steps

Below, we'll give some concrete hints on how to proceed.

### Step 0: Update your Python files

Download the new files mentioned above.  Then make these small adjustments to
files you already edited in HW6:

* If you added to `tag.py` in HW6 (by supporting `--awesome` for extra credit),
  then you may wish to copy your additions into the new version of `tag.py`.

### Step 1: Offload backpropagation and parameter updates to PyTorch

In HW6, you manually implemented the gradient computation for the linear-chain
CRF (using observed minus expected counts).  You also manually updated the
parameters in the direction of the gradient.

Instead, let's have PyTorch compute the gradient with the `.backward()` method
(backprop) and carry out the parameter updates with `torch.optim`.  That's the
standard, easy way to train neural nets and other models.

#### What to know

The new file `crf_backprop.py` will get you started on this nicer
implementation.  It contains a class `ConditionalRandomFieldBackprop`, which
inherits from `ConditionalRandomField` but also from `nn.Module`, which you
already used in
[HW3](https://www.cs.jhu.edu/~jason/465/hw-lm/code/probs.py). The `train()`
logic is inherited from the parent class, but you will have to override the
methods that it calls to reset, accumulate, and follow the gradient.  You will
only have to write a few lines of code.

Make sure you understand how
[`nn.Parameter`](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)
determines which gradients are tracked by
[`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
You'll use a call like `loss.backward()` to accumulate the gradient.

You'll also use the `torch.optim` package (read more about `torch.optim`
[here](https://pytorch.org/docs/stable/optim.html#module-torch.optim) or see
examples in HW3) to perform parameter updates with `optimizer.step()`.  We've
created an `optimizer` for you at the start of `train()`.

The CRF portions of `test_ic` and `test_en` are a useful starting point for testing. Check
that `ConditionalRandomFieldBackprop` behaves just like its parent class
`ConditionalRandomField` when you train on `icsup` and evaluate on `icdev`, or
when you train on `ensup` and evaluate on `endev`.  It does exactly the same
work -- but using PyTorch facilities, which will help you extend the class
further in later steps.

#### Fixing a problem with back-propagation

You may discover that your initial approach only works for minibatches of the
default size 1 (as used in `ic_test`).  When you try larger minibatches (as in
`en_test`), you'll get an error saying that you have run the `backward()` method
twice.

This happens because you are separately computing the gradient of each example
in the minibatch.  In principle, that should work.  When you compute the `loss`
on an example, calling `loss.backward()` will propagate the gradient backwards,
accmulating $\frac{\partial \texttt{loss}}{\partial x}$ at each intermediate
quantity $x$.  Eventually it works back to the parameters $\theta =$ `(WA,
WB)` -- the inputs or "leaves" of the computation graph -- and adds
$\frac{\partial \texttt{loss}}{\partial \theta}$ to their `.grad` attributes .
By doing this for all examples in the minibatch, you can accumulate the gradient
of the total minibatch loss with respect to $\theta$. 

(*Note*: In general, when `backward()` reports a mysterious error, try doing
`torch.autograd.set_detect_anomaly(True)` at the start of training.  This lets
PyTorch give more informative error messages, at the expense of slowing it
down.)]

In this case, the trouble is that in _your_ computation graph, the  examples in a minibatch
_shared some of their computation_.  They didn't just share parameters.  You computed `A` and
`B` just once from the parameters (using `updateAB()`) and then *reused* them
for all examples in the minibatch:

![Three backward passes](minibatch-bad.png)

That was certainly the efficient way to compute all of the losses. However, it
means that when `loss2.backward()` reaches `A` and `B`, it will complain that it
*already* back-propagated from them during `loss1.backward()`, after which it
aggressively freed the memory needed to track their gradients (their `.grad_fn`
attributes).  Thus, it can't back-propgate from those nodes again.

To fix this, when you call `loss1.backward()`, you could include the argument
`retain_graph=True`, which says not to free the memory.

However, that's considered hacky **and will slow your code down a lot**.  A more
standard and more efficient approach is to compute a single `minibatch_loss`
that sums the log-probabilities of all elements in the minibatch.  Then call
`minibatch_loss.backward()`.

![One backward pass](minibatch-good.png)

PyTorch will figure out that it should back-propagate from all examples to
accumulate *total* gradients at `A` and `B` *before* back-propagating from those
total gradients back to `WA` and `WB`.  This is more time-efficient because
`backward()` only has to work backward _once_ from `A,B` to `WA,WB` -- exactly
as `updateAB()` only worked forward _once_ from `WA,WB` to `A,B`.  (It's just
like how the backward algorithm in forward-backward collects the total $\beta$
value at a node before passing it back to earlier nodes.)

Can you figure out how to implement this approach without changing
`ConditionalRandomField.train()` but only overriding some of the
methods that it calls?  It's still only a few lines of code in
`ConditionalRandomFieldBackprop`. _Hint:_ The way that the work is
divided among `_zero_grad()`, `accumulate_logprob_gradient`, and
`logprob_gradient_step` will no longer exactly match the names of
those functions.

### Step 2: Add support for non-stationary features

In HW6, your HMM and CRF models were *stationary*.  That means that the same
probability matrices $A$ and $B$ -- or potential matrices $\phi_A$ and $\phi_B$
in the case of a CRF -- were used at every position $j$ of every sentence.  

A non-stationary HMM would be useful for weather patterns that change over time.
The stationary HMM assumed that $p(\texttt{H}\mid \texttt{C})$ was the same on
day 8 as it was on day 7, but we could relax that assumption.  

Non-stationary CRFs are even more useful.  A major advantage of the CRF design
is that the features for scoring a transition or emission can depend on the
observed input words around that transition or emission.  But then we need
different potential matrices $\phi_A$ and $\phi_B$ at different time steps. This
is discussed in the "Parameterization" section of the reading handout.

Thus, modify your `hmm.py` that you copied from HW6.  Add the functions `A_at()`
and `B_at()`, which will return the `A` and `B` matrices to use at a _particular
position in a sentence_:

    @typechecked
    def A_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        return self.A
    
    @typechecked
    def B_at(self, position: int, sentence: IntegerizedSentence) -> Tensor:
        return self.B

The default definitions above are stationary: they simply return fixed matrices
`self.A` and `self.B` (computed by `updateAB()`).  But you will override them in
non-stationary subclasses.

Now modify your Viterbi/forward/backward in `hmm.py` so that they call `A_at()`
and `B_at()` at every index rather than using `self.A` and `self.B`.  For
example, when you loop over positions `j` of a sentence, you can add the local
definitions

    A = self.A_at(j, isent)  
    B = self.B_at(j, isent)

at the start of the loop body, and then replace `self.A` and `self.B` in the
rest of the loop body with the local variables `A` and `B`.

As in the previous step, this **should not** cause any changes to your results
when tagging with `hmm.py` and `crf.py`.  The following should still get the
same results as in the previous homework:

    python3 tag.py endev --train ensup         # HMM
    python3 tag.py endev --train ensup --crf   # CRF, now with backprop

Before you check this, make sure to change `tag.py` to use your new backprop
code.  The easiest way is to comment out

     from crf import ConditionalRandomField  

and replace it with

     from crf_backprop import ConditionalRandomFieldBackprop as ConditionalRandomField

### Step 3: Add some actual non-stationary features

Now test your implementation by creating a CRF with a few non-stationary
features.  (There is no need to implement a non-stationary _HMM_ in this
homework.)

You can do this by implementing `A_at()` and `B_at()` in the
`ConditionalRandomFieldTest` class (`crf_test.py`).  You will not hand this
class in, so you can experiment however you like.  For example, perhaps $\phi_A$
and/or $\phi_B$ at a position $j$ should depend on $j$ itself, or on the
embeddings of the words $w_{j-1}, w_j, w_{j+1}$.

Whatever design you pick, the `train()` method inherited from the parent class
`ConditionalRandomFieldBackprop` will figure out how to train it for you.
That's the beauty of back-propagation.

#### Running your test class

For convenience, `ConditionalRandomFieldTest` behaves like the
`ConditionalRandomFieldNeural` class that you will implement below.  In
particular, they have the same constructor arguments.  Thus, to try out your
test class, you can use the command-line interface in `tag.py`.  Just
temporarily change

    from crf_neural import ConditionalRandomFieldNeural

to 

    from crf_test import ConditionalRandomFieldTest as ConditionalRandomFieldNeural

You are then (temporarily!) pretending that your test class implements the
neural CRF, and you can pass it a lexicon using the `--lexicon` command-line
argument (or just use the default one-hot lexicon).  Once you're done testing,
change `tag.py` back.

Alternatively, you can call `ConditionalRandomFieldTest` from a notebook like
`test_ic` or `test_en`.

To see whether your features really work, consider making a tiny artificial
dataset where the tag $t_j$ really is influenced (or even determined) by the
position $j$, or by the next word $w_{j+1}$, in a way that can't be modeled by a
first-order HMM or by its discriminative version (a simple CRF).  We have given
you two such datasets in the [`data`](../data) directory, `pos` and `next`.  You
should be able to fit these _much better_ with `ConditionalRandomFieldTest` than
with its stationary parent, `ConditionalRandomFieldBackprop`.

Notice that `posdev` has some long sentences, so it contains a few positions $j$
that never appeared in `possup`.  Does your trained tagger correctly predict the
tags at those positions?  A position-specific emission feature like $(j=38
\wedge w_j=\text{``\texttt{x}''} \wedge t_j=\text{``\texttt{\_}''})$ will fire
on some candidate taggings of the dev data, but this large-$j$ feature won't be
very useful since you never saw it in training data.  If you do want to
generalize to larger $j$ in held-out (dev) data from smaller $j$ in training
data, how about features that look at $j \bmod m$ (especially $j \bmod 4$)?

### Step 4: Implement a biRNN-CRF

Now implement the biRNN-based features suggested in the "Parameterization"
section of the reading handout.  You can use `crf_neural.py` as your starter
code.  The class `ConditionalRandomFieldNeural` inherits from
`ConditionalRandomFieldBackprop`.

#### Some things you'll have to do

* You'll need to add parameters to the model to help you compute all these
  things (the various $\theta, M, U$ parameters described in the reading
  handout). Remember to wrap them as `nn.Parameter` and assign them to class
  attributes so that Pytorch will track their gradients!

* You'll need to override the `A_at()` and `B_at()` functions. For sigmoid and
  concatenation (F.sigmoid and torch.cat) operations, be careful what dimension
  you are computing these along. Test out your functions to make sure they are
  computing what you expect.

* You can use one-hot embeddings for the _tag_ embeddings (so the embedding
  matrix is just the identity matrix `torch.eye()`). 

* The _word_ embeddings will be supplied to the model constructor as a Tensor
  called `lexicon`.  For speed, you can take this to be fixed rather than
  treating it as a fine-tuneable parameter. The lexicon includes embeddings for
  many of the held-out words (`endev`), not just for training words (`ensup`),
  so that you can learn to guess their tags from their pre-trained embeddings.
  (Study `lexicon.py` and `tag.py` for how this is arranged!)  This is not
  cheating because you're not peeking at the gold tags on the held-out words;
  those words just happened to be in your pre-trained lexicon.

* You can start out by using null vectors (`tensor([])`) for the biRNN vectors
  $\vec{h}_j$ and $\vec{h}'_j$.  Once you are ready to implement them, make sure
  that you spend only $O(1)$ time computing each vector (independent of sentence
  length).  There are two reasonable implementations:

  + **Lazy** (compute on demand): Use an `h_at()` function.  Note that its
    return values will depend on $\vec{h}_{j-1}$ and $\vec{h}'_{j+1}$ due to the
    recurrent definition.  But you don't want to do the full $O(n)$ recurrence
    each time you call `h_at()`: that would wastefully recompute vectors that
    you'd computed before.  Thus, implement some caching mechanism where you can
    store vectors for resuse.  One option is Python's `@lru_cache` decorator.

  + **Eager** (precompute): Before training or testing on a sentence, run the
    left-to-right and right-to-left RNNs, and store all resulting token
    encodings in `self` where `A_at()` and `B_at()` can look at them.  You'll
    need to add the following method to `hmm.py`:

        def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Precompute any quantities needed for forward/backward/Viterbi algorithms.
        This method may be overridden in subclasses."""
        pass

    and make sure to call it from `forward_pass`, `backward_pass`, and
    `Viterbi_tagging`.  Then you can override it in
    `ConditionalRandomFieldNeural` (and any other subclass that needs this kind
    of pre-computation).

#### Testing and speeding up your biRNN-CRF implementation

* The artificial `next` and `pos` datasets in the [`data`](../data) directory
  are a great way to test. The tagging patterns here are deterministic, but as
  as you found earlier, they can't be picked up by the simple bigram HMM or CRF
  designs from the previous homework.  Even a tiny BiRNN-CRF should be able to
  discover the patterns very quickly, with cross-entropy $\rightarrow$ 0 and
  accuracy $\rightarrow$ 100%.  For example:

      # uses one-hot embeddings since there's no lexicon
      python3 tag.py nextdev --train nextsup --model next-rnn2.pkl --crf --rnn_dim 2 --eval_interval 200 --max_steps 6000
      python3 tag.py posdev --train possup --model pos-rnn2.pkl --crf --rnn_dim 2 --eval_interval 200 --max_steps 6000

* For the `en` (English part-of-speech) dataset, a sample `tag.py` invocation
  with some reasonable initial guesses of the hyperparameters is given near the
  start of these instructions.  You may be able to get better or faster training
  by modifying some of the hyperparameters, potentially including the lexicon.
  You may want to start out testing with smaller corpora such as `ensup-tiny`,
  which you can use for both training and evaluation.

* During training, you may notice that the progress bar pauses at the end of
  each minibatch as the system runs backprop (that is, if you're doing backprop
  on minibatch loss) and updates the parameters.  It also pauses when the system
  saves model checkpoints.

* To help you tune hyperparameters like learning rate and minibatch size, you
  might find it useful to monitor a quantity that we call "learning speed" as
  you train.  See the `learning_speed` lines that we've added to the HW6 version
  of [`crf.py`](https://cs.jhu.edu/~jason/465/hw-tag/code/crf.py); you could add
  them to your own copy.

* Training will be very slow if you do not use vectorized tensor operations!
  Make sure to avoid using for loops in `A_at()` and `B_at()`. *Hint:* Think
  about your output dimensions.  `A_at()` should return a k × k matrix, hence
  the final output dimension after multiplying out your weights (see sec. H.4,
  eq. 45) should be k^2. `B_at()` should first compute a k × 1 matrix since the
  word emitted at a given position is known (but should still ultimately return
  a k × V matrix).

* Even with batched tensor operations, your bi-RNN may still take several hours
  to train (depending on your machine and the details of your code), so start
  early and plan ahead!   

  We recommend accelerating training with a GPU.  This won't speed up `for`
  loops, which are serial, but it will accelerate tensor operations using
  parallelism.  There are several ways to get GPU access, but we recommend
  Kaggle (for instructions, see the reading handout). 

  You won't get the *full* advantage of the GPU without more work,
  unfortunately. To fully occupy its processors, you would have to modify your
  forward algorithm to run on all sentences in a minibatch in parallel. This
  means adding a dimension to all the tensors: for example, an alpha vector
  becomes a matrix, with the new dimension indicating which sentence you're
  working on.  This is how neural net training (and testing) code is really
  implemented; it allows *large* minibatches to be processed rapidly using the
  hardware. But we won't make you do it here.

### Step 5: Experiment

Experiment with your tagger, as described in the homework handout.  Some sample
command lines using `tag.py` were given near the start of these instructions.
You will find it useful to look in the `.eval` logfiles that they create.

For quick tests to make sure your code doesn't crash, feel free to use the
icecream data (or truncated versions of the English files).  You may also want
to use small `--eval_interval` or `--max_steps` so that your code finishes quickly.

Even if you use `tag.py` at the command line instead of creating a notebook, you
still have to keep track of your experiments somehow.  We recommend that you
maintain a simple file (one command per line) that lists all of your training
and testing commands (one per line).  That way, you'll know exactly which
command-line options you used to create each file in your directory.  You can
easily re-run a command by pasting it into your terminal.

#### Pro tip

You can also execute this file using `bash`, which will run *all* of the
commands.  That rebuilds all your models and output files from scratch using
your current code.  And once you're treating it as a bash file, you can make use
of other bash features like comments and loops.

If you're running a lot of experiments in that way, you may want quieter output.
The `-q` option suppresses the logger's stderr output, except for any
warning/error messages.  To suppress progress bars, set the environment variable
`TQDM_DISABLE`.  In Linux you can set the environment of a single command like
this:

    TQDM_DISABLE=1 python3 tag.py -q ...

### Step 6: Informed embeddings

For this part of the assignment (see handout), you should complete the
`problex_lexicon()` method in `lexicon.py`.   You can then try it out with the
`--problex` option to `tag.py`.  

You can try `--problex` both with and without `--lexicon`.

You can also try omitting both.  In that case, `tag.py` will fall back to
using simple one-hot embeddings of the training words (as long as it uses a
neural model at all -- you can force it to still select
`ConditionalRandomFieldNeural` by specifying `--rnn-dim`).
