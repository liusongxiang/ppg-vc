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
[] Upload pretraind models.
