import os
import glob2
import numpy as np
import io
from tqdm import tqdm
import soundfile
import resampy
import pyworld

import torch
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def compute_f0(
    wav, 
    sr,
    f0_floor=20.0,
    f0_ceil=600.0,
    frame_period=10.0
):
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(
        wav, sr, frame_period=frame_period, f0_floor=20.0, f0_ceil=600.0)
    return f0.astype(np.float32)


def compute_f0_from_wav(
    wavfile_path,
    sampling_rate,
    f0_floor, 
    f0_ceil,
    frame_period_ms,
):
    # try:
    wav, sr = soundfile.read(wavfile_path)
    if len(wav) < sr:
        return None, sr, len(wav)
    if sr != sampling_rate:
        wav = resampy.resample(wav, sr, sampling_rate)
        sr = sampling_rate
    f0 = compute_f0(wav, sr, f0_floor, f0_ceil, frame_period_ms)
    return f0, sr, len(wav)


def process_one(
    wav_file_path,
    args,
    output_dir,
):
    fid = os.path.basename(wav_file_path)[:-4]
    save_fname = f"{output_dir}/{fid}.f0.npy"
    if os.path.isfile(save_fname):
        return

    f0, sr, wav_len = compute_f0_from_wav(
        wav_file_path, args.sampling_rate,
        args.f0_floor, args.f0_ceil, args.frame_period_ms)
    if f0 is None:
        return
    np.save(save_fname, f0, allow_pickle=False)


def run(args):
    """Compute merged f0 values."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    wav_dir = args.wav_dir
    # Get file id list
    wav_file_list = glob2.glob(f"{wav_dir}/**/*.wav")
    print(f"Globbed {len(wav_file_list)} wave files.")
    
    # Multi-process worker
    if args.num_workers < 2 :
        for wav_file_path in tqdm(wav_file_list):
            process_one(wav_file_path, args, output_dir)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for wav_file_path in wav_file_list:
                futures.append(executor.submit(
                    partial(
                        process_one, wav_file_path, args, output_dir,
                    )
                ))
            results = [future.result() for future in tqdm(futures)]
    

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Compute merged f0 values")
    parser.add_argument(
        "--wav_dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--frame_period_ms",
        default=10,
        type=float,
    )
    parser.add_argument(
        "--sampling_rate",
        default=24000,
        type=int,
    )
    parser.add_argument(
        "--f0_floor",
        default=80,
        type=int,
    )
    parser.add_argument(
        "--f0_ceil",
        default=600,
        type=int
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    run(args)


if __name__ == "__main__":
    main()   
