#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

remove_self_attn=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --remove_self_attn)
            remove_self_attn="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

checkpoint_path=weights/mdlm.ckpt

T=0
sampling_steps=1024
p=0.9
eta=0.02
t_on=0.55
t_off=0.05
alpha_on=0.9


if [[ "$remove_self_attn" == "true" ]]; then
    generated_seqs_path=outputs/rm-self-att-remdm-loop_T-${sampling_steps}_eta-${eta}_ton-${t_on}_toff-${t_off}_alphaon-${alpha_on}_topp-${p}.json
else
    generated_seqs_path=outputs/remdm-loop_T-${sampling_steps}_eta-${eta}_ton-${t_on}_toff-${t_off}_alphaon-${alpha_on}_topp-${p}.json
fi

export HYDRA_FULL_ERROR=1

python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path="${checkpoint_path}" \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-loop" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    seed=1 \
    sampling.num_sample_batches=200 \
    sampling.generated_seqs_path="${generated_seqs_path}" \
    sampling.nucleus_p=${p} \
    sampling.sampler=remdm-loop \
    sampling.eta=${eta} \
    sampling.t_on=${t_on} \
    sampling.t_off=${t_off} \
    sampling.alpha_on=${alpha_on} \
    +model.remove_self_attn=${remove_self_attn} \
    hydra.job.chdir=false