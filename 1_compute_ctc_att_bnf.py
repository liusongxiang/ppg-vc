"""
Compute CTC-Attention Seq2seq ASR encoder bottle-neck features (BNF).
"""
import sys
import os
import argparse
import torch
import glob2
import soundfile
import librosa

import numpy as np
from tqdm import tqdm
from conformer_ppg_model.build_ppg_model import load_ppg_model


SAMPLE_RATE=16000


def compute_bnf(
    output_dir: str,
    wav_dir: str,
    train_config: str,
    model_file: str,
):
    device = "cuda"
    
    # 1. Build PPG model
    ppg_model_local = load_ppg_model(train_config, model_file, device)

    # 2. Glob wav files
    wav_file_list = glob2.glob(f"{wav_dir}/**/*.wav")
    print(f"Globbing {len(wav_file_list)} wav files.")
    
    # 3. start to compute ppgs
    os.makedirs(output_dir, exist_ok=True)
    for wav_file in tqdm(wav_file_list):
        audio, sr = soundfile.read(wav_file, always_2d=False)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        wav_tensor = torch.from_numpy(audio).float().to(device).unsqueeze(0)
        wav_length = torch.LongTensor([audio.shape[0]]).to(device)
        with torch.no_grad():
            bnf = ppg_model_local(wav_tensor, wav_length) 
            # bnf = torch.nn.functional.softmax(asr_model.ctc.ctc_lo(bnf), dim=2)
        bnf_npy = bnf.squeeze(0).cpu().numpy()
        fid = os.path.basename(wav_file).split(".")[0]
        bnf_fname = f"{output_dir}/{fid}.ling_feat.npy"
        np.save(bnf_fname, bnf_npy, allow_pickle=False)


def get_parser():
    parser = argparse.ArgumentParser(description="compute ppg or ctc-bnf or ctc-att-bnf")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="./conformer_ppg_model/en_conformer_ctc_att/config.yaml",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="./conformer_ppg_model/en_conformer_ctc_att/24epoch.pth",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    kwargs = vars(args)
    compute_bnf(**kwargs)
