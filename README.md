
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


# Evaluation

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
    sampling.num_sample_batches=10 \
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
    sampling.generated_seqs_path=/home/nafez/scratch/remdm-shortcut-removal/outputs/remdm_shortcut-removal-conf_T-1024_topp-0.9.json \
    sampling.nucleus_p=0.9 \
    sampling.sampler="remdm-conf"
```



