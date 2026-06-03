
# RCP Setup & Commands

### Intractive session  (8-hours)

```bash
runai submit \
  --name mdlm-sampler-intractive \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash /scratch/mnafez/remdm-shortcut-removal/epfl-rcp-bootstrap.sh
```

```bash
runai bash mdlm-sampler-intractive -- bash --login
```

- Check [epfl-rcp-bootstrap.sh](epfl-rcp-bootstrap.sh) for: 
  - 8h (Intractive session)
  - Simlink for /scratch 
  - Activate conda env by default


# Transfer Checkpoint to RCP

```bash
scp -r /home/mnafez/Downloads/MDM-Checkpoints/mdlm.ckpt jumphost.rcp.epfl.ch:/home/nafez/scratch/remdm-shortcut-removal/weights
```

# Submit Job (None-Intractive)

```bash
runai submit \
  --name gidd-remdm-loop-original \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/remdm-loop.sh --remove_self_attn false
    "
```

```bash
runai submit \
  --name gidd-remdm-loop-remove-self-att \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/remdm-loop.sh --remove_self_attn ture
    "
```



```bash
runai submit \
  --name gidd-remdm-conf-original \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/remdm-conf.sh --remove_self_attn false
    "
```

```bash
runai submit \
  --name gidd-remdm-conf-remove-self-att \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/remdm-conf.sh --remove_self_attn ture
    "
```



```bash
runai submit \
  --name gidd-mdlm-original \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/mdlm.sh --remove_self_attn false
    "
```

```bash
runai submit \
  --name gidd-mdlm-remove-self-att \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/remdm-shortcut-removal && \
    bash scripts/mdlm.sh --remove_self_attn ture
    "
```



# Evaluation inside intractive session:

```bash
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
    eval.checkpoint_path=/home/nafez/scratch/remdm-shortcut-removal/weights/mdlm.ckpt \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/mdlm" \
    T=0 \
    sampling.steps=1024 \
    seed=1 \
    sampling.num_sample_batches=10 \
    sampling.generated_seqs_path=/home/nafez/scratch/remdm-shortcut-removal/outputs/mdlm_T-1024_topp-0.9.json \
    sampling.nucleus_p=0.9 \
    sampling.sampler="mdlm"
```

```bash
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
    eval.checkpoint_path=/home/nafez/scratch/remdm-shortcut-removal/weights/mdlm.ckpt \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-conf" \
    T=0 \
    sampling.steps=1024 \
    seed=1 \
    sampling.num_sample_batches=1 \
    sampling.generated_seqs_path=/home/nafez/scratch/remdm-shortcut-removal/outputs/remdm-conf_T-1024_topp-0.9.json \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remdm-conf"
```





```bash
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
    eval.checkpoint_path=/home/nafez/scratch/remdm-shortcut-removal/weights/mdlm.ckpt \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/remdm-conf" \
    T=0 \
    sampling.steps=1024 \
    seed=1 \
    sampling.num_sample_batches=1 \
    sampling.generated_seqs_path=/home/nafez/scratch/remdm-shortcut-removal/outputs/shortcut-removal-forward-backward_T-1024_topp-0.9.json \
    sampling.nucleus_p=0.9 \
    sampling.sampler="forward-backward"
```



<!-- forward-backward -->

# Temp command:


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
    eval.checkpoint_path=/home/nafez/scratch/remdm-shortcut-removal/weights/mdlm.ckpt \
    time_conditioning=false \
    +wandb.offline=true \
    hydra.run.dir="${PWD}/outputs/mdlm" \
    T=0 \
    sampling.steps=1024 \
    seed=1 \
    sampling.num_sample_batches=1 \
    sampling.generated_seqs_path=/home/nafez/scratch/remdm-shortcut-removal/outputs/temp_mdlm_T-1024_topp-0.9.json \
    sampling.nucleus_p=0.9 \
    sampling.sampler="mdlm"