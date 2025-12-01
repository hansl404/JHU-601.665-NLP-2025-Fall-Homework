#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""

    neural = True    # class attribute that indicates that constructor needs extra args
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        self.E = lexicon

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)



    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """

        # See the "Parameterization" section of the reading handout to determine
        # what dimensions all your parameters will need.

        d = self.rnn_dim         
        k = self.k               
        e = self.e               

        # h_j     = sigma( M [1; h_{j-1}; w_j] )
        # h'_j    = sigman( M'[1; w_j; h'_{j+1}] )
        self.M = nn.Parameter(torch.empty(d, 1 + d + e))
        self.M_prime = nn.Parameter(torch.empty(d, 1 + e + d))

        dim_in_A = 1 + d + k + k + d      # 1 + 2d + 2k
        self.U_a = nn.Parameter(torch.empty(d, dim_in_A))
        self.theta_a = nn.Parameter(torch.empty(d))

        dim_in_B = 1 + d + k + e + d      # 1 + 2d + k + e
        self.U_b = nn.Parameter(torch.empty(d, dim_in_B))
        self.theta_b = nn.Parameter(torch.empty(d))

        for P in (self.M, self.M_prime, self.U_a, self.U_b):
            nn.init.xavier_uniform_(P)

        nn.init.normal_(self.theta_a, mean=0.0, std=0.1)
        nn.init.normal_(self.theta_b, mean=0.0, std=0.1)

        self.count_params()

    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""

        device = self.E.device
        dtype = self.E.dtype

        n = len(isent)
        d = self.rnn_dim

        # forward RNN: h_j 
        h_prefix = []
        h_prev = torch.zeros(d, device=device, dtype=dtype)  # h_{-1} = 0
        for j in range(n):
            w_idx = isent[j][0]                 
            w_vec = self.E[w_idx].to(device=device, dtype=dtype)  

            inp = torch.cat([
                torch.ones(1, device=device, dtype=dtype),  # constant 1
                h_prev,
                w_vec
            ])                                              # [1 + d + e]

            h_j = torch.sigmoid(self.M @ inp)               # [d]
            h_prefix.append(h_j)
            h_prev = h_j

        # backward RNN: h'_j 
        h_suffix = [None] * n
        h_next = torch.zeros(d, device=device, dtype=dtype)   # h'_n = 0
        for j in reversed(range(n)):
            w_idx = isent[j][0]
            w_vec = self.E[w_idx].to(device=device, dtype=dtype)

            inp = torch.cat([
                torch.ones(1, device=device, dtype=dtype),
                w_vec,
                h_next
            ])                                               # [1 + e + d]

            h_j = torch.sigmoid(self.M_prime @ inp)         # [d]
            h_suffix[j] = h_j
            h_next = h_j

        self._h_prefix = h_prefix
        self._h_suffix = h_suffix
        self._current_isent = isent  


    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        n = len(sentence)
        k = self.k
        d = self.rnn_dim
        device = self.E.device
        dtype = self.E.dtype

        if position <= 0:
            h_left = torch.zeros(d, device=device, dtype=dtype)
        else:
            h_left = self._h_prefix[position - 1]

        if position >= n:
            h_right = torch.zeros(d, device=device, dtype=dtype)
        else:
            h_right = self._h_suffix[position]

        eye = self.eye.to(device=device, dtype=dtype)  # [k, k]

        s_ids = torch.arange(k, device=device)
        t_ids = torch.arange(k, device=device)
        s_flat = s_ids.repeat_interleave(k)   # [k*k]
        t_flat = t_ids.repeat(k)             # [k*k]

        S = eye[s_flat]   # [k*k, k]
        T = eye[t_flat]   # [k*k, k]

        ctx = torch.cat([
            torch.ones(1, device=device, dtype=dtype),
            h_left,
            h_right
        ])                                    # [1 + 2d]
        ctx = ctx.unsqueeze(0).expand(k * k, -1)   # [k*k, 1+2d]

        X = torch.cat([ctx, S, T], dim=1)     # [k*k, 1 + 2d + 2k]

        H = torch.sigmoid(F.linear(X, self.U_a))   # [k*k, d]

        scores = H @ self.theta_a                  # [k*k]
        scores = scores.view(k, k)                 # [k, k]

        A = torch.exp(scores)                      # [k, k]

        maskA = torch.ones_like(A)
        maskA[:, self.bos_t] = 0.0   
        maskA[self.eos_t, :] = 0.0   

        A = A * maskA                

        return A

        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""

        n = len(sentence)
        k = self.k
        V = self.V
        d = self.rnn_dim
        device = self.E.device
        dtype = self.E.dtype

        B = torch.ones(k, V, device=device, dtype=dtype)

        if position <= 0 or position >= n - 1:
            B[self.bos_t, :] = 0.0
            B[self.eos_t, :] = 0.0
            return B

        w_idx = sentence[position][0]

        h_left = self._h_prefix[position - 1]
        h_right = self._h_suffix[position]

        w_vec = self.E[w_idx].to(device=device, dtype=dtype)

        ctx = torch.cat([
            torch.ones(1, device=device, dtype=dtype),
            h_left,
            w_vec,
            h_right
        ])                                    # [1 + d + e + d] = [1 + 2d + e]
        ctx = ctx.unsqueeze(0).expand(k, -1)  # [k, 1 + 2d + e]

        eye = self.eye.to(device=device, dtype=dtype)   # [k, k]
        T = eye                                         

        X = torch.cat([ctx, T], dim=1)                 # [k, 1 + 2d + e + k]

        H = torch.sigmoid(F.linear(X, self.U_b))       # [k, d]
        scores = H @ self.theta_b                      # [k]
        col = torch.exp(scores)                        # [k]

        mask = torch.ones_like(col)
        mask[self.bos_t] = 0.0
        mask[self.eos_t] = 0.0
        col = col * mask

        B[:, w_idx] = col   

        return B

