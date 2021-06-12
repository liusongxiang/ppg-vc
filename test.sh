export CUDA_VISIBLE_DEVICES=7

stage=1
stop_stage=1
config=$1
echo ${config}

# =============== One-shot VC ================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  #config=./conf/seq2seq_mol_ppg2mel_vctk_libri_oneshotvc_r4_normMel.yaml
  #config=./conf/bilstm_ppg2mel_vctk_libri_oneshotvc.yaml
  exp_name="$(basename "${config}" .yaml)"
  echo Experiment name: "${exp_name}"
  model_file="$(ls -dt "ckpt/${exp_name}_seed0"/*.pth | head -1 || true)"
  echo  Checkpoint file: ${model_file}
  src_wav_dir="/home/shaunxliu/data/cmu_arctic/cmu_us_rms_arctic/wav"
  ref_wav_path="/home/shaunxliu/data/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0001.wav"
  output_dir="vc_gen_wavs/$(basename "${config}" .yaml)"

  python convert_from_wav.py \
    --ppg2mel_model_train_config ${config} \
    --ppg2mel_model_file ${model_file} \
    --src_wav_dir "${src_wav_dir}" \
    --ref_wav_path "${ref_wav_path}" \
    -o "${output_dir}"
fi
