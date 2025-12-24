from typing import Iterable, Optional
import time

import torch
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer


# -------------------------
# Dataset
# -------------------------
def fetch_dataset(year: str) -> Dataset:
    ds = load_dataset(
        "dell-research-harvard/AmericanStories",
        "subset_years",
        year_list=[year],
        trust_remote_code=True,
        revision="77e27fa69c4788dfaad1c9efd8a226d5a32d3e9a",
    )
    return ds[year]


# -------------------------
# Device selection
# -------------------------
def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# -------------------------
# Chunk generator (STANDARD)
# -------------------------
def iter_token_chunks(
    texts: Iterable[str],
    tokenizer,
    chunk_size: int = 128,
    overlap: int = 16,
):
    """
    Yield token windows (lists of token ids).
    Standard streaming chunker used in production pipelines.
    """
    step = chunk_size - overlap
    for text in texts:
        if not isinstance(text, str) or not text.strip() or len(text) < chunk_size:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        for i in range(0, len(ids), step):
            window = ids[i : i + chunk_size]
            if not window:
                break
            yield window
            if i + chunk_size >= len(ids):
                break


# -------------------------
# Embedding pipeline
# -------------------------
def embed_dataset(
    ds: Dataset,
    text_field: str,
    model_name: str,
    device: Optional[torch.device] = None,
    chunk_size: int = 128,
    overlap: int = 16,
    embed_batch_size: int = 64,   # tuned for Qwen 4B
):
    if device is None:
        device = get_best_device()

    print(f"Loading model {model_name} on {device}")
    model = SentenceTransformer(model_name, device=str(device))

    # IMPORTANT for Qwen: keep internal padding aligned
    model.max_seq_length = chunk_size

    tokenizer = model.tokenizer

    total_tokens = 0
    total_chunks = 0
    start = time.perf_counter()

    chunk_text_batch = []
    chunk_token_batch = 0

    pbar = tqdm(total=len(ds), desc="Embedding", unit="doc")

    for row in ds:
        text = row[text_field]

        for window in iter_token_chunks(
            [text],
            tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
        ):
            chunk_text_batch.append(
                tokenizer.decode(
                    window,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )
            chunk_token_batch += len(window)

            if len(chunk_text_batch) == embed_batch_size:
                # ---- embed fixed-size batch ----
                model.encode(
                    chunk_text_batch,
                    batch_size=embed_batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )

                total_tokens += chunk_token_batch
                total_chunks += len(chunk_text_batch)

                chunk_text_batch.clear()
                chunk_token_batch = 0

                elapsed = time.perf_counter() - start
                toks_per_s = total_tokens / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix(
                    toks_s=f"{toks_per_s:,.0f}",
                    total_toks=f"{total_tokens:,}",
                    chunks=f"{total_chunks:,}",
                )

        pbar.update(1)

    # ---- final flush ----
    if chunk_text_batch:
        model.encode(
            chunk_text_batch,
            batch_size=len(chunk_text_batch),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        total_tokens += chunk_token_batch
        total_chunks += len(chunk_text_batch)

    elapsed = time.perf_counter() - start
    pbar.close()

    stats = {
        "documents": len(ds),
        "chunks": total_chunks,
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0.0,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embed_batch_size": embed_batch_size,
        "model": model_name,
        "device": str(device),
    }
    return stats


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    device = get_best_device()
    print("Using device:", device)

    ds = fetch_dataset("1800")
    print("Rows:", len(ds))
    print("Columns:", ds.column_names)

    stats = embed_dataset(
        ds,
        text_field="article",
        model_name="Qwen/Qwen3-Embedding-4B",
        device=device,
        chunk_size=128,
        overlap=16,
        embed_batch_size=1,
    )

    print("\nDone.")
    for k, v in stats.items():
        print(f"{k}: {v}")
