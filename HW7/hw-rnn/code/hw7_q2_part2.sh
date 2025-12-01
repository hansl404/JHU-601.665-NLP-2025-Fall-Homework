#!/bin/bash

# --- Q2(a,b): biRNN-CRF with different lexicons (fix rnn_dim=8) ---

# lexicon = words-20
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-20.txt \
  --model model/en-birnn-d8-w20.pkl --device cuda \
  > logs/q2_en_birnn_d8_w20.log 2>&1

# lexicon = words-50
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-50.txt \
  --model model/en-birnn-d8-w50.pkl --device cuda \
  > logs/q2_en_birnn_d8_w50.log 2>&1

# lexicon = words-100
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-100.txt \
  --model model/en-birnn-d8-w100.pkl --device cuda \
  > logs/q2_en_birnn_d8_w100.log 2>&1

# Google-Syntactic lexicon = words-gs-10
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-gs-10.txt \
  --model model/en-birnn-d8-wgs10.pkl --device cuda \
  > logs/q2_en_birnn_d8_wgs10.log 2>&1

# Google-Syntactic lexicon = words-gs-50
python -u tag.py ../data/endev --train ../data/ensup --crf \
  --reg 0.1 --lr 0.01 --batch_size 30 \
  --rnn_dim 8 --lexicon ../lexicons/words-gs-50.txt \
  --model model/en-birnn-d8-wgs50.pkl --device cuda \
  > logs/q2_en_birnn_d8_wgs50.log 2>&1
