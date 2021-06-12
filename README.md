# ppg-vc
PPG-Based Voice Conversion

# How to use
## Data preprocessing
- Please run `1_compute_ctc_att_bnf.py` to compute PPG features.
- Please run `2_compute_f0.py` to compute fundamental frequency.
- Please run `3_compute_spk_dvecs.py` to compute speaker d-vectors.

## Training
- Please refer to `run.sh`

## Conversion
- Plesae refer to `test.sh`

# TODO
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
