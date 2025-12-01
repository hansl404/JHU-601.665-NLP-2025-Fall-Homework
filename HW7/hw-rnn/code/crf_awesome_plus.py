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
from crf_awesome import ConditionalRandomFieldNeural as ConditionalRandomFieldNeuralAwesome

logger = logging.getLogger(Path(__file__).stem)  

torch.manual_seed(1337)
cuda.manual_seed(69_420) 

class ConditionalRandomFieldNeuralPlus(ConditionalRandomFieldNeuralAwesome):
    @override
    def init_params(self) -> None:
        super().init_params()

        d = self.rnn_dim
        k = self.k
        e = self.e

        self.W_A = nn.Parameter(torch.empty(d, 1 + 2 * d))
        self.b_A = nn.Parameter(torch.empty(d))

        self.W_A_out = nn.Parameter(torch.empty(k * k, d))

        self.W_B = nn.Parameter(torch.empty(d, 1 + 2 * d + e))
        self.b_B = nn.Parameter(torch.empty(d))

        self.W_B_out = nn.Parameter(torch.empty(k, d))

        for P in (self.W_A, self.W_A_out, self.W_B, self.W_B_out):
            nn.init.xavier_uniform_(P)
        for b in (self.b_A, self.b_B):
            nn.init.normal_(b, mean=0.0, std=0.1)

        self.count_params()

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
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

        ctx = torch.cat(
            [torch.ones(1, device=device, dtype=dtype), h_left, h_right]
        )

        z = torch.tanh(F.linear(ctx, self.W_A, self.b_A))         

        scores_flat = F.linear(z, self.W_A_out)                 
        scores = scores_flat.view(k, k)                        

        A = torch.exp(scores)

        maskA = torch.ones_like(A)
        maskA[:, self.bos_t] = 0.0
        maskA[self.eos_t, :] = 0.0

        return A * maskA

    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
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

        ctx = torch.cat(
            [torch.ones(1, device=device, dtype=dtype), h_left, w_vec, h_right]
        )

        z = torch.tanh(F.linear(ctx, self.W_B, self.b_B))         
        scores = F.linear(z, self.W_B_out)                         
        col = torch.exp(scores)                                    

        mask = torch.ones_like(col)
        mask[self.bos_t] = 0.0
        mask[self.eos_t] = 0.0
        col = col * mask

        B[:, w_idx] = col
        return B
