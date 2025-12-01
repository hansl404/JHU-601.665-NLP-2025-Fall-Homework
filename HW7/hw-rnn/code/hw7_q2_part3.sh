#!/bin/bash

# --- Q2(b,c): Effect of training hyperparameters (reg, lr, batch_size) ---

# reg = 0.01
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.01 --lr 0.01 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
  --model model/en-birnn-d8-w50-reg0.01.pkl --device cuda \
  > logs/q2_en_birnn_d8_w50_reg0.01.log 2>&1

# reg = 0.0  (修掉 device typo)
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.0 --lr 0.01 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
  --model model/en-birnn-d8-w50-reg0.pkl --device cuda \
  > logs/q2_en_birnn_d8_w50_reg0.log 2>&1

# higher lr = 0.05
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.05 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
  --model model/en-birnn-d8-w50-lr0.05.pkl --device cuda \
  > logs/q2_en_birnn_d8_w50_lr0.05.log 2>&1

# higher lr = 0.1
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.1 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
  --model model/en-birnn-d8-w50-lr0.05.pkl --device cuda \
  > logs/q2_en_birnn_d8_w50_lr0.1.log 2>&1 

TQDM_DISABLE=1 nohup python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.1 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
  --model model/en-birnn-d8-w50-lr0.05.pkl \
  > logs/q2_en_birnn_d8_w50_lr0.1.log 2>&1 &

# larger minibatch = 60
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 60 \
  --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
  --model model/en-birnn-d8-w50-b60.pkl --device cuda \
  > logs/q2_en_birnn_d8_w50_b60.log 2>&1

# 如果你想要 "smaller minibatch = 10"，可以再开这一行：
# python -u tag.py ../data/endev --train ../data/ensup --crf \
#   --reg 0.1 --lr 0.01 --batch_size 10 \
#   --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
#   --model model/en-birnn-d8-w50-b10.pkl --device cuda \
#   > logs/q2_en_birnn_d8_w50_b10.log 2>&1

# --- Q2(d): Evaluate trained models on training data (overfitting check) ---
# （等前面模型训练完再单独跑也行）

# Baseline CRF on training set
python -u tag.py ../data/ensup --model model/en-crf-basic.pkl --device cuda \
  > logs/q2_en_crf_basic_on_train.log 2>&1

# Best biRNN-CRF models on training set
# python -u tag.py ../data/ensup --model model/en-birnn-d8-w50.pkl --device cuda \
#   > logs/q2_en_birnn_d8_w50_on_train.log 2>&1

# python -u tag.py ../data/ensup --model model/en-birnn-d8-w50-reg0.pkl --device cuda \
#   > logs/q2_en_birnn_d8_w50_reg0_on_train.log 2>&1

# --- Optional: smaller training sets (data-size effect) ---
# 根据时间决定要不要打开

# python -u tag.py ../data/endev --train ../data/ensup-tiny --crf \
#   --reg 0.1 --lr 0.01 --batch_size 30 \
#   --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
#   --model model/en-birnn-d8-w50-tiny.pkl --device cuda \
#   > logs/q2_en_birnn_d8_w50_tiny.log 2>&1

# python -u tag.py ../data/endev --train ../data/ensup4k --crf \
#   --reg 0.1 --lr 0.01 --batch_size 30 \
#   --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
#   --model model/en-birnn-d8-w50-4k.pkl --device cuda \
#   > logs/q2_en_birnn_d8_w50_4k.log 2>&1

# python -u tag.py ../data/endev --train ../data/ensup10k --crf \
#   --reg 0.1 --lr 0.01 --batch_size 30 \
#   --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
#   --model model/en-birnn-d8-w50-10k.pkl --device cuda \
#   > logs/q2_en_birnn_d8_w50_10k.log 2>&1

# python -u tag.py ../data/endev --train ../data/ensup25k --crf \
#   --reg 0.1 --lr 0.01 --batch_size 30 \
#   --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
#   --model model/en-birnn-d8-w50-25k.pkl --device cuda \
#   > logs/q2_en_birnn_d8_w50_25k.log 2>&1
