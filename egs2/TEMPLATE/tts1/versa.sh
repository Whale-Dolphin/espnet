srctexts="data/${train_set}/text"
test_set="eval1"
data_feats="${dumpdir}/raw"
_data=${data_feats}/${test_set}
tts_exp=exp/tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g2p_en_no_space

inference_nj=16
nj=32
gpu_inference=true

. ./path.sh
. ./cmd.sh

# VERSA eval related
skip_scoring=true # Skip scoring stages.
skip_wer=true # Skip WER evaluation.
whisper_tag=medium # Whisper model tag.
whisper_dir=local/whisper # Whisper model directory.
cleaner=whisper_en # Text cleaner for whisper model.
hyp_cleaner=whisper_en # Text cleaner for hypothesis.

mos_config=mos.yaml # MOS evaluation configuration.
spk_config=spk.yaml # Speaker evaluation configuration.

versa_eval_params=(
    # mos
    spk
) # Parameters for VERSA evaluation.
inference_tag=decode_vits_latest

_gen_dir=${tts_exp}/${inference_tag}/${test_set}

if ! ${skip_wer}; then
    ./scripts/utils/evaluate_asr.sh \
        --whisper_tag ${whisper_tag} \
        --whisper_dir ${whisper_dir} \
        --cleaner ${cleaner} \
        --hyp_cleaner ${hyp_cleaner} \
        --inference_nj ${inference_nj} \
        --nj ${nj} \
        --gt_text ${_data}/text \
        --gpu_inference ${gpu_inference} \
        ${_gen_dir}/wav/wav_test.scp ${_gen_dir}/scoring/eval_wer
fi

for _eval_item in "${versa_eval_params[@]}"; do
    _eval_flag=eval_${_eval_item}
    if ${!_eval_flag}; then
        _opts=
        _eval_dir=${_gen_dir}/scoring/eval_${_eval_item}
        mkdir -p ${_eval_dir}

        if [ ${_eval_item} == "mos" ]; then
            _pred_file=${_gen_dir}/wav/wav_test.scp
            _score_config=${mos_config}
            _gt_file=
        elif [ ${_eval_item} == "spk" ]; then
            _pred_file=${_gen_dir}/wav/wav_test.scp
            _score_config=${spk_config}
            _gt_file=${ref_dir}/utt2spk
        fi

        _nj=$(( ${inference_nj} < $(wc -l < ${_pred_file}) ? ${inference_nj} : $(wc -l < ${_pred_file}) ))

        _split_files=""
        for n in $(seq ${_nj}); do
            _split_files+="${_eval_dir}/pred.${n} "
        done
        utils/split_scp.pl ${_pred_file} ${_split_files}

        if [ -n "${_gt_file}" ]; then
            _split_files=""
            for n in $(seq ${_nj}); do
                _split_files+="${_eval_dir}/gt.${n} "
            done
            utils/split_scp.pl ${_gt_file} ${_split_files}
            _opts+="--gt ${_eval_dir}/gt.JOB"
        fi

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_eval_dir}"/eval_${_eval_item}.JOB.log \
        python -m versa.bin.scorer \
            --pred ${_eval_dir}/pred.JOB \
            --score_config ${_score_config} \
            --cache_folder ${_eval_dir}/cache \
            --use_gpu ${gpu_inference} \
            --output_file ${_eval_dir}/result.JOB.txt \
            --io soundfile \
            ${_opts} || exit 1;

        pyscripts/utils/aggregate_eval.py \
            --logdir ${_eval_dir} \
            --scoredir ${_eval_dir} \
            --nj ${_nj}
    fi
done