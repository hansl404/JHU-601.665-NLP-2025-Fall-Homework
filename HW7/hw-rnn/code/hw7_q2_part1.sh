#!/bin/bash

# --- Q2(a): Baselines on English POS tagging (endev / ensup) ---

# HMM baseline
python -u tag.py ../data/endev --train ../data/ensup \
  --model model/en-hmm.pkl --device cuda \
  > logs/q2_en_hmm.log 2>&1

# Stationary CRF baseline (no neural features)
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.0 --lr 0.01 --batch_size 30 \
  --model model/en-crf-basic.pkl --device cuda \
  > logs/q2_en_crf_basic.log 2>&1

# --- Q2(a,b,c): biRNN-CRF with different RNN dims, fixed lexicon ---

# Base neural setting: d = 8, lexicon words-10
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-10.txt \
  --model model/en-birnn-d8-w10.pkl --device cuda \
  > logs/q2_en_birnn_d8_w10.log 2>&1

nohup python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.0 --lr 0.01 --batch_size 30 \
  --model model/en-crf-basic_adamw.pkl \
  > logs/q2_en_crf_basic_adamw.log 2>&1 &
# rnn_dim = 4
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 30 \
  --rnn_dim 4 --lexicon ../lexicons/words-10.txt \
  --model model/en-birnn-d4-w10.pkl --device cuda \
  > logs/q2_en_birnn_d4_w10.log 2>&1

# rnn_dim = 16 (lexicon = words-50)
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 30 \
  --rnn_dim 16 --lexicon ../lexicons/words-50.txt \
  --model model/en-birnn-d16-w50.pkl --device cuda \
  > logs/q2_en_birnn_d16_w50.log 2>&1

# rnn_dim = 16 (lexicon = words-50)
TQDM_DISABLE=1 nohup python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 30 \
  --rnn_dim 32 --lexicon ../lexicons/words-10.txt \
  --model model/en-birnn-d32-w10.pkl\
  > logs/q2_en_birnn_d32_w10.log 2>&1 &
