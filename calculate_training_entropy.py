'''
python calculate_training_entropy.py \
    eval.checkpoint_path=/home/nafez/scratch/remdm-shortcut-removal/weights/mdlm.ckpt \
    loader.num_workers=4 \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false \
    data=lm1b
'''

import os
import random
import hydra
import lightning as L
import omegaconf
import torch
import dataloader
import diffusion

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


def compute_entropy(batch_input_ids):
    entropies = []
    for row in batch_input_ids:
        counts = torch.unique(row, return_counts=True, sorted=True)[1]
        entropies.append(
            torch.special.entr(counts.float() / counts.sum()).sum().item()
        )
    return entropies


def compute_gen_ppl(model, tokenizer, samples):
    model.gen_ppl_metric.reset()
    model.compute_generative_perplexity(samples)
    return model.gen_ppl_metric.compute().cpu().item()


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    L.seed_everything(config.seed)
    # print(omegaconf.OmegaConf.to_yaml(config))
    tokenizer = dataloader.get_tokenizer(config)
    train_ds, _ = dataloader.get_dataloaders(config, tokenizer, skip_valid=True)

    model = diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
    ).to('cuda')
    model.eval()

    for split_name, loader in [('train', train_ds),]:
        all_entropies = []
        all_samples = []

        for batch in loader:
            input_ids = batch['input_ids']
            all_entropies.extend(compute_entropy(input_ids))
            all_samples.extend(tokenizer.batch_decode(input_ids))
            print(f"Batch shape: {input_ids.shape}", end=' ')
            print(f"[{split_name}] processed {len(all_entropies)} samples", end='\r')


        gen_ppl = compute_gen_ppl(model, tokenizer, random.sample(all_samples, 2000))
        avg_entropy = sum(all_entropies) / len(all_entropies)
        print(f"\n\n[{split_name}] entropy={avg_entropy:.4f}  gen_ppl={gen_ppl:.4f}")


if __name__ == '__main__':
    main()