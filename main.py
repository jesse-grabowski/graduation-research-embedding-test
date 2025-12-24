from typing import Iterable, List, Optional, Tuple
import time

import torch
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer

def fetch_dataset(year):
    ds = load_dataset(
        "dell-research-harvard/AmericanStories",
        "subset_years",
        year_list=[year],
        trust_remote_code=True,
        revision="77e27fa69c4788dfaad1c9efd8a226d5a32d3e9a",
    )
    return ds[year]

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def embed_dataset(
    ds: Dataset,
    text_field: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_length: int = 256,
    device: Optional[torch.device] = None,
) -> Tuple[List[List[float]], dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")

    model = SentenceTransformer(model_name, device=str(device))

    tokenizer = model.tokenizer

    total_texts = len(ds)
    total_tokens = 0
    start = time.perf_counter()

    embeddings: List[List[float]] = []

    pbar = tqdm(range(0, total_texts, batch_size), desc="Embedding", unit="batch")
    for i in pbar:
        batch_texts = ds[text_field][i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        batch_tokens = int(enc["attention_mask"].sum().item())
        total_tokens += batch_tokens

        # Encode (SentenceTransformers handles batching; we pass max_length for truncation)
        batch_emb = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
            device=str(device),
        )
        # save memory for now and don't keep embeddings
        # embeddings.extend(batch_emb.tolist())

        elapsed = time.perf_counter() - start
        toks_per_s = total_tokens / elapsed if elapsed > 0 else 0.0
        pbar.set_postfix(
            toks_s=f"{toks_per_s:,.0f}",
            toks=batch_tokens,
            total_toks=f"{total_tokens:,}",
        )

    elapsed = time.perf_counter() - start
    stats = {
        "total_texts": total_texts,
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "toks_per_s": (total_tokens / elapsed) if elapsed > 0 else 0.0,
        "model_name": model_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "device": str(device),
        "text_field": text_field,
    }
    return embeddings, stats

if __name__ == "__main__":
    device = get_best_device()
    print("Using device:", device)

    ds = fetch_dataset("1800")
    print("Rows:", len(ds))
    print("Columns:", ds.column_names)

    text_field = "article"

    embeddings, stats = embed_dataset(
        ds,
        text_field=text_field,
        device=device,
        batch_size=1,
        max_length=256,
        model_name="Qwen/Qwen3-Embedding-4B",
    )

    print("Done.")
    print(stats)