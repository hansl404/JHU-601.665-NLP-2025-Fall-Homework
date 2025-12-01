#!/bin/bash
# Homework 7: Neuralization – experiment command list
# Run this from the homework code directory (where tag.py lives).
# Data are in ../data, lexicons are in ../lexicon.

############################
# Q1. Implement a Neural CRF
############################

# Q1(a,b): Show that HMM/CRF cannot fit patterns in the synthetic next/pos datasets.

# HMM baselines
python3 tag.py ../data/nextdev --train ../data/nextsup --model model/next-hmm.pkl
python3 tag.py ../data/posdev  --train ../data/possup  --model model/pos-hmm.pkl

# Stationary CRF baselines (no neural features)
python3 tag.py ../data/nextdev --train ../data/nextsup --crf --model model/next-crf.pkl
python3 tag.py ../data/posdev  --train ../data/possup  --crf --model model/pos-crf.pkl

# Q1(a,c): biRNN-CRF on next/pos with different RNN dimensions d.

# Small biRNN-CRF (d = 2), as suggested in INSTRUCTIONS
python3 tag.py ../data/nextdev --train ../data/nextsup --crf --rnn_dim 2 --model model/next-rnn2.pkl --eval_interval 200 --max_steps 6000
python3 tag.py ../data/posdev  --train ../data/possup  --crf --rnn_dim 2 --model model/pos-rnn2.pkl  --eval_interval 200 --max_steps 6000

# Try d = 0 (no RNN context → only embeddings)
python3 tag.py ../data/nextdev --train ../data/nextsup --crf --rnn_dim 0 --model model/next-rnn0.pkl --eval_interval 200 --max_steps 6000
python3 tag.py ../data/posdev  --train ../data/possup  --crf --rnn_dim 0 --model model/pos-rnn0.pkl  --eval_interval 200 --max_steps 6000

# Try a larger d to see effect on speed / overfitting
python3 tag.py ../data/nextdev --train ../data/nextsup --crf --rnn_dim 8 --model model/next-rnn8.pkl --eval_interval 200 --max_steps 6000
python3 tag.py ../data/posdev  --train ../data/possup  --crf --rnn_dim 8 --model model/pos-rnn8.pkl  --eval_interval 200 --max_steps 6000

####################
# Q2. Experiment (en)
####################

# Q2(a): Baselines on English POS tagging (endev / ensup).

# HMM baseline
python3 tag.py ../data/endev --train ../data/ensup --model model/en-hmm.pkl

# Stationary CRF baseline (no neural features)
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.0 --lr 0.05 --batch_size 30   --model model/en-crf-basic.pkl

# Q2(a,b,c): biRNN-CRF under different hyperparameters.

# Base neural setting suggested in INSTRUCTIONS (d = 8, lexicon words-10)
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-10.txt   --model model/en-birnn-d8-w10.pkl

# Vary RNN dimensionality (keeping lexicon fixed)
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 4 --lexicon ../lexicon/words-10.txt   --model model/en-birnn-d4-w10.pkl

python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 16 --lexicon ../lexicon/words-10.txt   --model model/en-birnn-d16-w10.pkl

# Vary lexicon size (keeping RNN dim fixed at 8)
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-20.txt   --model model/en-birnn-d8-w20.pkl

python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50.pkl

python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-100.txt   --model model/en-birnn-d8-w100.pkl

# Try Google-Syntactic lexicons (words-gs-*)
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-gs-10.txt   --model model/en-birnn-d8-wgs10.pkl

python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-gs-50.txt   --model model/en-birnn-d8-wgs50.pkl

# Q2(b,c): Effect of training hyperparameters (lr, reg, batch_size).

# Lower regularization
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.01 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50-reg0.01.pkl

# No regularization
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.0 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50-reg0.pkl

# Higher learning rate
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.05 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50-lr0.05.pkl

# Larger minibatch
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 60   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50-b60.pkl

# Smaller minibatch
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 10   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50-b10.pkl

# Q2(d): Evaluate trained models on training data (overfitting check).

# Baseline CRF on training set
python3 tag.py ../data/ensup --model model/en-crf-basic.pkl

# Best biRNN-CRF models on training set
python3 tag.py ../data/ensup --model model/en-birnn-d8-w50.pkl
python3 tag.py ../data/ensup --model model/en-birnn-d8-w50-reg0.pkl

# Optional: use smaller training sets to see data-size effect.
python3 tag.py ../data/endev --train ../data/ensup-tiny --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50-tiny.pkl

python3 tag.py ../data/endev --train ../data/ensup4k --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50-4k.pkl

python3 tag.py ../data/endev --train ../data/ensup10k --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50-10k.pkl

python3 tag.py ../data/endev --train ../data/ensup25k --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-w50-25k.pkl

###########################
# Q3. Informed Embeddings
###########################

# Use the same base hyperparameters as a good biRNN-CRF (e.g., d = 8, lexicon words-50)
# and vary embedding sources: one-hot, CBOW only, problex only, CBOW+problex.
# (All on ensup/endev.)

# 3.1: One-hot embeddings only (no CBOW lexicon, no problex)
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8   --model model/en-birnn-d8-onehot.pkl

# 3.2: CBOW embeddings only (lexicon, no problex)
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt   --model model/en-birnn-d8-cbow-w50.pkl

# 3.3: Frequency-based features only (problex, no CBOW lexicon)
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --problex   --model model/en-birnn-d8-problex-only.pkl

# 3.4: CBOW + frequency features together (lexicon + problex)
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicon/words-50.txt --problex   --model model/en-birnn-d8-cbow+problex-w50.pkl

# 3.5: Isolate effect of embeddings by turning off RNN context (rnn_dim = 0).

# One-hot only, no context
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 0   --model model/en-emb-only-onehot.pkl

# CBOW only, no context
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 0 --lexicon ../lexicon/words-50.txt   --model model/en-emb-only-cbow-w50.pkl

# Problex only, no context
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 0 --problex   --model model/en-emb-only-problex.pkl

# CBOW+problex, no context
python3 tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 0 --lexicon ../lexicon/words-50.txt --problex   --model model/en-emb-only-cbow+problex-w50.pkl

###############################
# Q4. Extensions (extra credit)
###############################
# These commands are only illustrative; they assume you have implemented the
# corresponding ideas (e.g., affixes_lexicon, trainable embeddings, etc.).

# Example: add affix or shape features via a custom lexicon (after you implement affixes_lexicon)
# python3 tag.py ../data/endev --train ../data/ensup --crf \
#   --reg 0.1 --lr 0.01 --batch_size 30 \
#   --rnn_dim 8 --lexicon ../lexicon/words-50.txt --problex \
#   --model model/en-birnn-d8-cbow+problex+affix.pkl

# Example: run with TQDM_DISABLE and -q for quieter logs:
# TQDM_DISABLE=1 python3 tag.py -q ../data/endev --train ../data/ensup --crf \
#   --reg 0.1 --lr 0.01 --batch_size 30 \
#   --rnn_dim 8 --lexicon ../lexicon/words-50.txt \
#   --model model/en-birnn-d8-w50-quiet.pkl
