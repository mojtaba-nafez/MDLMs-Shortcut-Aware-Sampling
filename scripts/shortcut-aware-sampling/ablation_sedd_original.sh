#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export HYDRA_FULL_ERROR=1


python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=sedd \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path="${PWD}/weights/sedd.ckpt" \
    time_conditioning=true \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/sedd" \
    T=0 \
    sampling.steps=32 \
    seed=1 \
    sampling.num_sample_batches=20 \
    sampling.generated_seqs_path="${PWD}/outputs/sedd-original-32.json" \
    sampling.predictor="analytic" \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false



python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=sedd \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path="${PWD}/weights/sedd.ckpt" \
    time_conditioning=true \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/sedd" \
    T=0 \
    sampling.steps=64 \
    seed=1 \
    sampling.num_sample_batches=20 \
    sampling.generated_seqs_path="${PWD}/outputs/sedd-original-64.json" \
    sampling.predictor="analytic" \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false


python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=sedd \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path="${PWD}/weights/sedd.ckpt" \
    time_conditioning=true \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/sedd" \
    T=0 \
    sampling.steps=128 \
    seed=1 \
    sampling.num_sample_batches=20 \
    sampling.generated_seqs_path="${PWD}/outputs/sedd-original-128.json" \
    sampling.predictor="analytic" \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false



python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=sedd \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path="${PWD}/weights/sedd.ckpt" \
    time_conditioning=true \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/sedd" \
    T=0 \
    sampling.steps=256 \
    seed=1 \
    sampling.num_sample_batches=20 \
    sampling.generated_seqs_path="${PWD}/outputs/sedd-original-256.json" \
    sampling.predictor="analytic" \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false



python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=sedd \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path="${PWD}/weights/sedd.ckpt" \
    time_conditioning=true \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/sedd" \
    T=0 \
    sampling.steps=512 \
    seed=1 \
    sampling.num_sample_batches=20 \
    sampling.generated_seqs_path="${PWD}/outputs/sedd-original-512.json" \
    sampling.predictor="analytic" \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false



python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=sedd \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path="${PWD}/weights/sedd.ckpt" \
    time_conditioning=true \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/sedd" \
    T=0 \
    sampling.steps=1024 \
    seed=1 \
    sampling.num_sample_batches=20 \
    sampling.generated_seqs_path="${PWD}/outputs/sedd-original-1024.json" \
    sampling.predictor="analytic" \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false



python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=sedd \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path="${PWD}/weights/sedd.ckpt" \
    time_conditioning=true \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/sedd" \
    T=0 \
    sampling.steps=2048 \
    seed=1 \
    sampling.num_sample_batches=20 \
    sampling.generated_seqs_path="${PWD}/outputs/sedd-original-2048.json" \
    sampling.predictor="analytic" \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false



python -u -m main \
    mode=sample_eval \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    eval.perplexity_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=sedd \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path="${PWD}/weights/sedd.ckpt" \
    time_conditioning=true \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/sedd" \
    T=0 \
    sampling.steps=4096 \
    seed=1 \
    sampling.num_sample_batches=20 \
    sampling.generated_seqs_path="${PWD}/outputs/sedd-original-4096.json" \
    sampling.predictor="analytic" \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false





