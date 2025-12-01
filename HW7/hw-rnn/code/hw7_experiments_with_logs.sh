#!/bin/bash
# Homework 7: Neuralization – experiment command list
# Run this from the homework code directory (where tag.py lives).
# Data are in ../data, lexicons are in ../lexicon.

mkdir -p logs
export TQDM_DISABLE=1
############################
# Q1. Implement a Neural CRF
############################

# # Q1(a,b): Show that HMM/CRF cannot fit patterns in the synthetic next/pos datasets.

# # HMM baselines
# python tag.py ../data/nextdev --train ../data/nextsup --max_step 100 --model model/next-hmm.pkl > logs/q1_next_hmm.log 2>&1
# python tag.py ../data/posdev  --train ../data/possup  --max_step 100 --model model/pos-hmm.pkl > logs/q1_pos_hmm.log 2>&1

# # Stationary CRF baselines (no neural features)
# python tag.py ../data/nextdev --train ../data/nextsup --crf --model model/next-crf.pkl   > logs/q1_next_crf.log 2>&1
# python tag.py ../data/posdev  --train ../data/possup  --crf --model model/pos-crf.pkl   > logs/q1_pos_crf.log 2>&1
 
# Q1(a,c): biRNN-CRF on next/pos with different RNN dimensions d.

# Small biRNN-CRF (d = 2), as suggested in INSTRUCTIONS
# python tag.py ../data/nextdev --train ../data/nextsup --crf --rnn_dim 2 --model model/next-rnn2.pkl --eval_interval 200 --max_steps 6000 --loss viterbi_error  > logs/q1_next_rnn2.log 2>&1
# python tag.py ../data/posdev  --train ../data/possup  --crf --rnn_dim 2 --model model/pos-rnn2.pkl  --eval_interval 200 --max_steps 6000 --loss viterbi_error > logs/q1_pos_rnn2.log 2>&1

# python tag.py ../data/nextdev --train ../data/nextsup --crf --rnn_dim 15 --max_steps 1000 --eval_interval 200 
# # Try d = 0 (no RNN context → only embeddings)
# python tag.py ../data/nextdev --train ../data/nextsup --crf --rnn_dim 0 --model model/next-rnn0.pkl --eval_interval 200 --max_steps 6000 --loss viterbi_error  > logs/q1_next_rnn0.log 2>&1
# python tag.py ../data/posdev  --train ../data/possup  --crf --rnn_dim 0 --model model/pos-rnn0.pkl  --eval_interval 200 --max_steps 6000 --loss viterbi_error  > logs/q1_pos_rnn0.log 2>&1

# # Try a larger d to see effect on speed / overfitting
# python tag.py ../data/nextdev --train ../data/nextsup --crf --rnn_dim 8 --model model/next-rnn8.pkl --eval_interval 200 --max_steps 6000 --loss viterbi_error  > logs/q1_next_rnn8.log 2>&1
# python tag.py ../data/posdev  --train ../data/possup  --crf --rnn_dim 8 --model model/pos-rnn8.pkl  --eval_interval 200 --max_steps 6000 --loss viterbi_error  > logs/q1_pos_rnn8.log 2>&1

# # Try a larger d to see effect on speed / overfitting
# python tag.py ../data/nextdev --train ../data/nextsup --crf --rnn_dim 15 --model model/next-rnn8.pkl --eval_interval 200 --max_steps 6000 --loss viterbi_error  > logs/q1_next_rnn15.log 2>&1
# python tag.py ../data/posdev  --train ../data/possup  --crf --rnn_dim 15 --model model/pos-rnn8.pkl  --eval_interval 200 --max_steps 6000 --loss viterbi_error  > logs/q1_pos_rnn15.log 2>&1

####################
# Q2. Experiment (en)
####################

# Q2(a): Baselines on English POS tagging (endev / ensup).

# # HMM baseline
# python tag.py ../data/endev --train ../data/ensup --model model/en-hmm.pkl   > logs/q2_en_hmm.log 2>&1

# # Stationary CRF baseline (no neural features)
# python tag.py ../data/endev --train ../data/ensup --crf --reg 0.0 --lr 0.05 --batch_size 30 --model model/en-crf-basic.pkl > logs/q2_en_crf_basic.log 2>&1

# # Q2(a,b,c): biRNN-CRF under different hyperparameters.

# # Base neural setting suggested in INSTRUCTIONS (d = 8, lexicon words-10)
# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-10.txt  --model model/en-birnn-d8-w10.pkl   > logs/q2_en_birnn_d8_w10.log 2>&1

# # Vary RNN dimensionality (keeping lexicon fixed)
# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 4 --lexicon ../lexicons/words-10.txt   --model model/en-birnn-d4-w10.pkl   > logs/q2_en_birnn_d4_w10.log 2>&1

# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 16 --lexicon ../lexicons/words-10.txt   --model model/en-birnn-d16-w10.pkl   > logs/q2_en_birnn_d16_w10.log 2>&1

# # Vary lexicon size (keeping RNN dim fixed at 8)
# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-20.txt   --model model/en-birnn-d8-w20.pkl   > logs/q2_en_birnn_d8_w20.log 2>&1

# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50.pkl   > logs/q2_en_birnn_d8_w50.log 2>&1

# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-100.txt   --model model/en-birnn-d8-w100.pkl   > logs/q2_en_birnn_d8_w100.log 2>&1

# # Try Google-Syntactic lexicons (words-gs-*)
# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-gs-10.txt   --model model/en-birnn-d8-wgs10.pkl   > logs/q2_en_birnn_d8_wgs10.log 2>&1

# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-gs-50.txt   --model model/en-birnn-d8-wgs50.pkl   > logs/q2_en_birnn_d8_wgs50.log 2>&1

# # Q2(b,c): Effect of training hyperparameters (lr, reg, batch_size).

# # Lower regularization
# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.01 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50-reg0.01.pkl   > logs/q2_en_birnn_d8_w50_reg0.01.log 2>&1

# # No regularization
# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.0 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50-reg0.pkl   > logs/q2_en_birnn_d8_w50_reg0.log 2>&1

# # Higher learning rate
# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.05 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50-lr0.05.pkl   > logs/q2_en_birnn_d8_w50_lr0.05.log 2>&1

# # Larger minibatch
# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 60   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50-b60.pkl   > logs/q2_en_birnn_d8_w50_b60.log 2>&1

# # Smaller minibatch
# python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 10   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50-b10.pkl   > logs/q2_en_birnn_d8_w50_b10.log 2>&1

# Q2(d): Evaluate trained models on training data (overfitting check).

# # Baseline CRF on training set
# python tag.py ../data/ensup --model model/en-crf-basic.pkl   > logs/q2_en_crf_basic_on_train.log 2>&1

# # Best biRNN-CRF models on training set
# python tag.py ../data/ensup --model model/en-birnn-d8-w50.pkl   > logs/q2_en_birnn_d8_w50_on_train.log 2>&1
# python tag.py ../data/ensup --model model/en-birnn-d8-w50-reg0.pkl   > logs/q2_en_birnn_d8_w50_reg0_on_train.log 2>&1

# # Optional: use smaller training sets to see data-size effect.
# python tag.py ../data/endev --train ../data/ensup-tiny --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50-tiny.pkl   > logs/q2_en_birnn_d8_w50_tiny.log 2>&1

# python tag.py ../data/endev --train ../data/ensup4k --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50-4k.pkl   > logs/q2_en_birnn_d8_w50_4k.log 2>&1

# python tag.py ../data/endev --train ../data/ensup10k --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50-10k.pkl   > logs/q2_en_birnn_d8_w50_10k.log 2>&1

# python tag.py ../data/endev --train ../data/ensup25k --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-w50-25k.pkl   > logs/q2_en_birnn_d8_w50_25k.log 2>&1

###########################
# Q3. Informed Embeddings
###########################

# Use the same base hyperparameters as a good biRNN-CRF (e.g., d = 8, lexicon words-50)
# and vary embedding sources: one-hot, CBOW only, problex only, CBOW+problex.
# (All on ensup/endev.)

# 3.1: One-hot embeddings only (no CBOW lexicon, no problex)
python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8   --model model/en-birnn-d8-onehot.pkl   > logs/q3_en_birnn_d8_onehot.log 2>&1

# 3.2: CBOW embeddings only (lexicon, no problex)
python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt   --model model/en-birnn-d8-cbow-w50.pkl   > logs/q3_en_birnn_d8_cbow_w50.log 2>&1

# 3.3: Frequency-based features only (problex, no CBOW lexicon)
python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --problex   --model model/en-birnn-d8-problex-only.pkl   > logs/q3_en_birnn_d8_problex_only.log 2>&1

# 3.4: CBOW + frequency features together (lexicon + problex)
python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 8 --lexicon ../lexicons/words-50.txt --problex   --model model/en-birnn-d8-cbow+problex-w50.pkl   > logs/q3_en_birnn_d8_cbow+problex_w50.log 2>&1

# 3.5: Isolate effect of embeddings by turning off RNN context (rnn_dim = 0).

# One-hot only, no context
python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 0   --model model/en-emb-only-onehot.pkl   > logs/q3_en_emb_only_onehot.log 2>&1

# CBOW only, no context
python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 0 --lexicon ../lexicons/words-50.txt   --model model/en-emb-only-cbow-w50.pkl   > logs/q3_en_emb_only_cbow_w50.log 2>&1

# Problex only, no context
python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 0 --problex   --model model/en-emb-only-problex.pkl   > logs/q3_en_emb_only_problex.log 2>&1

# CBOW+problex, no context
python tag.py ../data/endev --train ../data/ensup --crf   --reg 0.1 --lr 0.01 --batch_size 30   --rnn_dim 0 --lexicon ../lexicons/words-50.txt --problex   --model model/en-emb-only-cbow+problex-w50.pkl   > logs/q3_en_emb_only_cbow+problex_w50.log 2>&1

# ###############################
# # Q4. Extensions (extra credit)
# ###############################
# # These commands are only illustrative; they assume you have implemented the
# # corresponding ideas (e.g., affixes_lexicon, trainable embeddings, etc.).

# # Example: add affix or shape features via a custom lexicon (after you implement affixes_lexicon)
# # python tag.py ../data/endev --train ../data/ensup --crf \
# #   --reg 0.1 --lr 0.01 --batch_size 30 \
# #   --rnn_dim 8 --lexicon ../lexicons/words-50.txt --problex \
# #   --model model/en-birnn-d8-cbow+problex+affix.pkl \
# #   > logs/q4_en_birnn_d8_cbow+problex+affix.log 2>&1

# # Example: run with TQDM_DISABLE and -q for quieter logs:
# # TQDM_DISABLE=1 python tag.py -q ../data/endev --train ../data/ensup --crf \
# #   --reg 0.1 --lr 0.01 --batch_size 30 \
# #   --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
# #   --model model/en-birnn-d8-w50-quiet.pkl \
# #   > logs/q4_en_birnn_d8_w50_quiet.log 2>&1


# 3.1: One-hot embeddings only (no CBOW lexicon, no problex)
python -u tag.py ../data/endev --model model/en-birnn-d8-onehot.pkl

# 3.2: CBOW embeddings only (lexicon, no problex)
python -u tag.py ../data/endev --model model/en-birnn-d8-cbow-w50.pkl

# 3.3: Frequency-based features only (problex, no CBOW lexicon)
python -u tag.py ../data/endev --model model/en-birnn-d8-problex-only.pkl

# 3.4: CBOW + frequency features together (lexicon + problex)
python -u tag.py ../data/endev --model model/en-birnn-d8-cbow+problex-w50.pkl
