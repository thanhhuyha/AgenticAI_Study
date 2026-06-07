from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class RAGConfig:
    # Edit hardcoded paths for your environment
    embedding_model_path: str = "/path/to/local/bge-base-en-v1.5"
    manual_md_dir: Path = Path("/path/to/aaa_manual_md")
    aaa_code_dir: Path = Path("/path/to/production_aaa_codes")
    rag_store_dir: Path = Path("outputs/aaa_rag_store")
    max_chunk_chars: int = 1800
    overlap_chars: int = 200
    top_k_knowledge: int = 6
    top_k_code_examples: int = 4


def iter_files(root: Path, exts: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for ext in exts:
        out.extend(root.rglob(f"*{ext}"))
    return sorted(out)


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class LocalEmbedder:
    def __init__(self, model_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model(**toks)
                pooled = mean_pool(out.last_hidden_state, toks["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.cpu().numpy())
        return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 768), dtype=np.float32)


def build_rag_store(cfg: RAGConfig) -> None:
    embedder = LocalEmbedder(cfg.embedding_model_path)
    docs: list[dict] = []

    manual_files = iter_files(cfg.manual_md_dir, (".md",))
    for fp in manual_files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        for idx, chunk in enumerate(chunk_text(text, cfg.max_chunk_chars, cfg.overlap_chars)):
            docs.append({"kind": "manual", "source": str(fp), "chunk_id": idx, "text": chunk})

    code_files = iter_files(cfg.aaa_code_dir, (".aaa", ".txt", ".rule"))
    for fp in code_files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        for idx, chunk in enumerate(chunk_text(text, cfg.max_chunk_chars, cfg.overlap_chars)):
            docs.append({"kind": "code", "source": str(fp), "chunk_id": idx, "text": chunk})

    if not docs:
        raise ValueError("No chunks found. Check manual_md_dir and aaa_code_dir.")

    texts = [d["text"] for d in docs]
    embs = embedder.encode(texts).astype(np.float32)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    cfg.rag_store_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(cfg.rag_store_dir / "aaa.index"))
    (cfg.rag_store_dir / "docs.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    print(f"Built RAG store with {len(docs)} chunks at {cfg.rag_store_dir}")


def load_rag_store(store_dir: Path) -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(store_dir / "aaa.index"))
    docs = json.loads((store_dir / "docs.json").read_text(encoding="utf-8"))
    return index, docs


def retrieve(cfg: RAGConfig, query: str) -> dict:
    embedder = LocalEmbedder(cfg.embedding_model_path)
    index, docs = load_rag_store(cfg.rag_store_dir)

    q = embedder.encode([query]).astype(np.float32)
    scores, ids = index.search(q, k=min(40, len(docs)))
    ranked = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        d = docs[idx]
        d["score"] = float(score)
        ranked.append(d)

    knowledge = [d for d in ranked if d["kind"] == "manual"][: cfg.top_k_knowledge]
    examples = [d for d in ranked if d["kind"] == "code"][: cfg.top_k_code_examples]
    return {"knowledge": knowledge, "examples": examples}


def build_final_prompt(user_request: str, retrieved: dict) -> str:
    knowledge_text = "\n\n".join(
        f"[Knowledge {i+1}] Source: {d['source']}\n{d['text']}" for i, d in enumerate(retrieved["knowledge"])
    )
    example_text = "\n\n".join(
        f"[Example {i+1}] Source: {d['source']}\n{d['text']}" for i, d in enumerate(retrieved["examples"])
    )

    return f"""You are an AAA coding assistant.
Use the reference material below to answer the user request.
Prioritize syntax and usage patterns from provided knowledge and examples.

### Retrieved AAA knowledge/syntax
{knowledge_text}

### Similar AAA code examples
{example_text}

### User request
{user_request}

### Output requirements
1) Provide the AAA solution/code.
2) Explain key syntax/logic choices briefly.
3) If debugging, point out root causes and fixes.
"""


if __name__ == "__main__":
    # Example usage:
    # 1) Build index once: build_rag_store(RAGConfig())
    # 2) Retrieve + prompt:
    #    r = retrieve(RAGConfig(), "Write an AAA rule to check spacing between X and Y")
    #    print(build_final_prompt("Write an AAA rule ...", r))
    cfg = RAGConfig()
    build_rag_store(cfg)
