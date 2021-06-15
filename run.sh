#!/usr/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=0

########## Train BiLSTM oneshot VC model ##########
python main.py --config ./conf/bilstm_ppg2mel_vctk_libri_oneshotvc.yaml \
               --oneshotvc \
               --bilstm
###################################################

########## Train Seq2seq oneshot VC model ###########
#python main.py --config ./conf/seq2seq_mol_ppg2mel_vctk_libri_oneshotvc.yaml \
               #--oneshotvc \
###################################################
