#!/usr/bin/env python3

# Subclass ConditionalRandomFieldBackprop to get a model that uses some
# contextual features of your choice.  This lets you test the revision to hmm.py
# that uses those features.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import tensor, Tensor, cuda
from jaxtyping import Float

from corpus import Tag, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldTest(ConditionalRandomFieldBackprop):
    """A CRF with some arbitrary non-stationary features, for testing."""
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                #  lexicon: Tensor,
                #  rnn_dim: int,
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""

        # an __init__() call to the nn.Module class must be made before assignment on the child.
        nn.Module.__init__(self)  

        # self.E = lexicon          # rows are word embeddings
        # self.e = lexicon.size(1)  # dimensionality of word embeddings
        # self.rnn_dim = rnn_dim

        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        # for possup/posdev
        self.period = 4
        self.W_pos = nn.Parameter(0.01 * torch.randn(self.period, self.k))

        # for nextsup/nextdev
        self.W_next = nn.Parameter(0.01 * torch.randn(self.V, self.k))

    @override
    def updateAB(self) -> None:
        # Your non-stationary A_at() and B_at() might not make any use of the
        # stationary A and B matrices computed by the parent.  So we override
        # the parent so that we won't waste time computing self.A, self.B.
        #
        # But if you decide that you want A_at() and B() at to refer to self.A
        # and self.B (for example, multiplying stationary and non-stationary
        # potentials), then you'll still need to compute them; in that case,
        # don't override the parent in this way.
        pass   # do nothing

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        # [docstring will be inherited from parent method]

        # You need to override this function to compute your non-stationary features.

        non_stationary_A = self.W_pos.new_ones(self.k, self.k)   # [k, k], using the same dtype and devices

        non_stationary_A[:, self.bos_t] = 0.0
        non_stationary_A[self.eos_t, :] = 0.0
        return non_stationary_A   # example

        # [docstring will be inherited from parent method]
        n = len(sentence)

        B = self.W_pos.new_ones(self.k, self.V)   # [k, V]

        if position == 0 or position == n - 1:
            return B
        curr_w = sentence[position][0]
        next_w = sentence[position + 1][0]
        idx = position % self.period
        print(position, n, idx, self.W_pos.shape, self.W_next.shape, curr_w, next_w)
        log_phi = self.W_pos[idx] + self.W_next[next_w]    # shape [k]

        log_phi = log_phi.clone()
        log_phi[self.bos_t] = -inf
        log_phi[self.eos_t] = -inf

        B[:, curr_w] = torch.exp(log_phi)  
        return B
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        n = len(sentence)

        B = self.W_pos.new_ones(self.k, self.V)   # [k, V]

        if position == 0 or position == n - 1:
            return B

        curr_w = sentence[position][0]

        idx = position % self.period
        log_phi = self.W_pos[idx].clone()    # [k]

        if position + 1 < n:
            next_w = sentence[position + 1][0]
            if 0 <= next_w < self.V:
                log_phi = log_phi + self.W_next[next_w]

        log_phi[self.bos_t] = -inf
        log_phi[self.eos_t] = -inf

        B[:, curr_w] = torch.exp(log_phi)   

        return B
