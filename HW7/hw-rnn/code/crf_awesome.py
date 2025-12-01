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

logger = logging.getLogger(Path(__file__).stem)  

torch.manual_seed(1337)
cuda.manual_seed(69_420) 

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):

    neural = True   
    
    @override
    def __init__(self,
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False,
                 tune_lexicon: bool = False):

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim

        self.e = lexicon.size(1)
        self._lexicon_init = lexicon
        self._tune_lexicon = tune_lexicon

        super().__init__(tagset, vocab, unigram)


    @override
    def init_params(self) -> None:

        if self._tune_lexicon:
            self.E = nn.Parameter(self._lexicon_init)
        else:
            self.register_buffer("E", self._lexicon_init)

        d = self.rnn_dim
        k = self.k
        e = self.e

        self.M = nn.Parameter(torch.empty(d, 1 + d + e))
        self.M_prime = nn.Parameter(torch.empty(d, 1 + e + d))

        dim_in_A = 1 + d + k + k + d    
        self.U_a = nn.Parameter(torch.empty(d, dim_in_A))
        self.theta_a = nn.Parameter(torch.empty(d))

        dim_in_B = 1 + d + k + e + d   
        self.U_b = nn.Parameter(torch.empty(d, dim_in_B))
        self.theta_b = nn.Parameter(torch.empty(d))

        for P in (self.M, self.M_prime, self.U_a, self.U_b):
            nn.init.xavier_uniform_(P)

        nn.init.normal_(self.theta_a, mean=0.0, std=0.1)
        nn.init.normal_(self.theta_b, mean=0.0, std=0.1)

        self.count_params()

    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:

        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:

        device = self.E.device
        dtype = self.E.dtype

        n = len(isent)
        d = self.rnn_dim

        h_prefix = []
        h_prev = torch.zeros(d, device=device, dtype=dtype)  
        for j in range(n):
            w_idx = isent[j][0]                 
            w_vec = self.E[w_idx].to(device=device, dtype=dtype)  

            inp = torch.cat([
                torch.ones(1, device=device, dtype=dtype),  
                h_prev,
                w_vec
            ])                                             

            h_j = torch.sigmoid(self.M @ inp)             
            h_prefix.append(h_j)
            h_prev = h_j

        h_suffix = [None] * n
        h_next = torch.zeros(d, device=device, dtype=dtype)  
        for j in reversed(range(n)):
            w_idx = isent[j][0]
            w_vec = self.E[w_idx].to(device=device, dtype=dtype)

            inp = torch.cat([
                torch.ones(1, device=device, dtype=dtype),
                w_vec,
                h_next
            ])                                              

            h_j = torch.sigmoid(self.M_prime @ inp)        
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

        eye = self.eye.to(device=device, dtype=dtype)  

        s_ids = torch.arange(k, device=device)
        t_ids = torch.arange(k, device=device)
        s_flat = s_ids.repeat_interleave(k)  
        t_flat = t_ids.repeat(k)            

        S = eye[s_flat] 
        T = eye[t_flat] 

        ctx = torch.cat([
            torch.ones(1, device=device, dtype=dtype),
            h_left,
            h_right
        ])                                  
        ctx = ctx.unsqueeze(0).expand(k * k, -1)   

        X = torch.cat([ctx, S, T], dim=1)   

        H = torch.sigmoid(F.linear(X, self.U_a))   

        scores = H @ self.theta_a                
        scores = scores.view(k, k)                

        A = torch.exp(scores)                   

        maskA = torch.ones_like(A)
        maskA[:, self.bos_t] = 0.0   
        maskA[self.eos_t, :] = 0.0   

        A = A * maskA             

        return A

        
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

        ctx = torch.cat([
            torch.ones(1, device=device, dtype=dtype),
            h_left,
            w_vec,
            h_right
        ])                                    
        ctx = ctx.unsqueeze(0).expand(k, -1)  

        eye = self.eye.to(device=device, dtype=dtype)   
        T = eye                                         

        X = torch.cat([ctx, T], dim=1)                 

        H = torch.sigmoid(F.linear(X, self.U_b))       
        scores = H @ self.theta_b                     
        col = torch.exp(scores)                       

        mask = torch.ones_like(col)
        mask[self.bos_t] = 0.0
        mask[self.eos_t] = 0.0
        col = col * mask

        B[:, w_idx] = col 

        return B

