'''
python calculate_real_text_entropy.py \
    eval.checkpoint_path=/home/nafez/scratch/remdm-shortcut-removal/weights/mdlm.ckpt \
    +sampling.revise_step=false \
    +model.remove_self_attn=false \
    +sampling.mask_embedding_blending=false \
    +text_path=/home/nafez/scratch/remdm-shortcut-removal/text_clean.txt
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


def load_and_chunk_text(text_path, tokenizer, chunk_size=1024):
    print("++++=", text_path)
    """Read text file, tokenize, and split into chunks of chunk_size tokens."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"Total tokens: {len(token_ids)}")

    # Drop the last incomplete chunk
    num_chunks = len(token_ids) // chunk_size
    token_ids = token_ids[:num_chunks * chunk_size]

    chunks = torch.tensor(token_ids).view(num_chunks, chunk_size)
    print(f"Total chunks of {chunk_size} tokens: {num_chunks}")
    return chunks


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    L.seed_everything(config.seed)
    tokenizer = dataloader.get_tokenizer(config)

    text_path = config.get('text_path', 'text.txt')
    chunk_size = config.model.length  # 1024

    # Load and chunk
    chunks = load_and_chunk_text(text_path, tokenizer, chunk_size)  # [N, 1024]

    # Load model
    model = diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
    ).to('cuda')
    model.eval()

    # Process in batches
    batch_size = config.loader.batch_size
    all_entropies = []
    all_samples = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]  # [B, 1024]
        all_entropies.extend(compute_entropy(batch))
        all_samples.extend(tokenizer.batch_decode(batch))
        print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks", end='\r')

    gen_ppl = compute_gen_ppl(model, tokenizer, random.sample(all_samples, min(2000, len(all_samples))))
    avg_entropy = sum(all_entropies) / len(all_entropies)
    print(f"\n\nentropy={avg_entropy:.4f}  gen_ppl={gen_ppl:.4f}")
    print(f"Total chunks processed: {len(all_entropies)}")


if __name__ == '__main__':
    main()