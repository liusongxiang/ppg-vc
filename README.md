# ppg-vc
Phonetic PosteriorGram (PPG)-Based Voice Conversion (VC)

This repo implements different kinds of PPG-based VC models.

Notes:

- The PPG model provided in `conformer_ppg_model` is based on Hybrid CTC-Attention phoneme recognizer, trained with LibriSpeech (960hrs). PPGs have frame-shift of 10 ms, with dimensionality of 144. This modelis very much similar to the one used in [this paper](https://arxiv.org/pdf/2011.05731v2.pdf).

- This repo uses [HifiGAN V1](https://github.com/jik876/hifi-gan) as the vocoder model, sampling rate of synthesized audio is 24kHz.

## Highlights
- Any-to-many VC
- Any-to-Any VC (a.k.a. few/one-shot VC)

## How to use
### Data preprocessing
- Please run `1_compute_ctc_att_bnf.py` to compute PPG features.
- Please run `2_compute_f0.py` to compute fundamental frequency.
- Please run `3_compute_spk_dvecs.py` to compute speaker d-vectors.

### Training
- Please refer to `run.sh`

### Conversion
- Plesae refer to `test.sh`

## TODO
- [ ] Upload pretraind models.

## Citations
```
@ARTICLE{liu2021any,
  author={Liu, Songxiang and Cao, Yuewen and Wang, Disong and Wu, Xixin and Liu, Xunying and Meng, Helen},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Any-to-Many Voice Conversion With Location-Relative Sequence-to-Sequence Modeling}, 
  year={2021},
  volume={29},
  number={},
  pages={1717-1728},
  doi={10.1109/TASLP.2021.3076867}
}

@inproceedings{Liu2018,
  author={Songxiang Liu and Jinghua Zhong and Lifa Sun and Xixin Wu and Xunying Liu and Helen Meng},
  title={Voice Conversion Across Arbitrary Speakers Based on a Single Target-Speaker Utterance},
  year=2018,
  booktitle={Proc. Interspeech 2018},
  pages={496--500},
  doi={10.21437/Interspeech.2018-1504},
  url={http://dx.doi.org/10.21437/Interspeech.2018-1504}
}
