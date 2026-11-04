"""Reusable FAISS vector store, used twice:
  RAG 1: catalog/         -> Query Agent tool      (lookup_catalog)
  RAG 2: knowledge_base/  -> Knowledge Agent tool  (search_kb)

Cost safeguards:
  * Documents are embedded ONCE (one batched call) and the index is cached on disk.
  * A hash of the documents + embedding model is stored with the index; it is rebuilt
    only when a document actually changes.
  * The store loads lazily, once per process (not on every Streamlit rerun).
  * Each search embeds only the short customer question (~30 tokens).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from backend import config

EmbedFn = Callable[[list[str]], np.ndarray]

MAX_CHARS = 900     # keeps every chunk small (~200 tokens)
BATCH_SIZE = 100    # texts per embedding API call

# Running total of tokens sent for embeddings in this process (for cost visibility).
usage = {"tokens": 0, "calls": 0}


# ----------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------
def openai_embed(texts: list[str]) -> np.ndarray:
    """Embed texts with OpenAI, in batches. Returns a float32 matrix."""
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        resp = client.embeddings.create(
            model=config.EMBEDDING_MODEL, input=texts[i : i + BATCH_SIZE]
        )
        vectors.extend(d.embedding for d in resp.data)
        usage["tokens"] += resp.usage.total_tokens
        usage["calls"] += 1
    return np.array(vectors, dtype="float32")


# ----------------------------------------------------------------------------
# Chunking: one chunk per heading section (one per Q&A pair in FAQ-style sections)
# ----------------------------------------------------------------------------
def _pack(blocks: list[str]) -> list[str]:
    """Greedily join paragraphs until a chunk would exceed MAX_CHARS."""
    groups, current = [], ""
    for b in blocks:
        if current and len(current) + len(b) + 2 > MAX_CHARS:
            groups.append(current)
            current = b
        else:
            current = f"{current}\n\n{b}" if current else b
    if current:
        groups.append(current)
    return groups


def chunk_markdown(text: str, source: str) -> list[dict]:
    title, heading, body = "", None, []
    sections: list[tuple[str | None, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((heading, body))
            heading, body = line[3:].strip(), []
        else:
            body.append(line)
    sections.append((heading, body))

    chunks = []
    for heading, body in sections:
        body_text = "\n".join(body).strip()
        if not body_text:
            continue
        label = heading or "Overview"
        blocks = [b.strip() for b in re.split(r"\n\s*\n", body_text) if b.strip()]
        groups = blocks if any(b.startswith("Q:") for b in blocks) else _pack(blocks)
        for g in groups:
            chunks.append(
                {"source": source, "heading": label, "text": f"{title} - {label}\n{g}"}
            )
    return chunks


# ----------------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------------
class FaissStore:
    def __init__(
        self,
        name: str,
        docs_dir: Path,
        index_dir: Path = config.INDEX_DIR,
        embed_fn: EmbedFn = openai_embed,
    ):
        self.name = name
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.embed_fn = embed_fn
        self.chunks: list[dict] = []
        self.index: faiss.Index | None = None
        self._ready = False

    # -- building / loading --------------------------------------------------
    def _doc_files(self) -> list[Path]:
        files = sorted(self.docs_dir.glob("*.md")) + sorted(self.docs_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .md/.txt documents found in {self.docs_dir}")
        return files

    def _chunks_from_docs(self) -> list[dict]:
        chunks: list[dict] = []
        for f in self._doc_files():
            chunks.extend(chunk_markdown(f.read_text(encoding="utf-8"), f.name))
        return chunks

    def _digest(self) -> str:
        h = hashlib.sha256(config.EMBEDDING_MODEL.encode())
        for f in self._doc_files():
            h.update(f.name.encode())
            h.update(f.read_bytes())
        return h.hexdigest()

    def ensure_ready(self) -> None:
        """Load the cached index, or build it once. Safe to call repeatedly."""
        if self._ready:
            return
        digest = self._digest()
        idx_path = self.index_dir / f"{self.name}.index"
        meta_path = self.index_dir / f"{self.name}_chunks.json"

        if idx_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("hash") == digest:
                self.index = faiss.read_index(str(idx_path))
                self.chunks = meta["chunks"]
                self._ready = True
                print(f"[rag:{self.name}] loaded {len(self.chunks)} chunks from cache (no API call)")
                return

        self.chunks = self._chunks_from_docs()
        try:
            before = usage["tokens"]
            matrix = self.embed_fn([c["text"] for c in self.chunks])
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(matrix.shape[1])  # inner product == cosine
            index.add(matrix)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(idx_path))
            meta_path.write_text(
                json.dumps({"hash": digest, "chunks": self.chunks}, ensure_ascii=False),
                encoding="utf-8",
            )
            self.index = index
            print(
                f"[rag:{self.name}] built index: {len(self.chunks)} chunks, "
                f"{usage['tokens'] - before} tokens embedded"
            )
        except Exception as e:  # no key / no network: keep the app alive with keyword search
            self.index = None
            print(f"[rag:{self.name}] WARNING embedding failed ({e}); using keyword fallback")
        self._ready = True

    # -- searching -----------------------------------------------------------
    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return up to k hits: {source, heading, text, score, method}, best first."""
        self.ensure_ready()
        if self.index is not None:
            try:
                q = self.embed_fn([query])
                faiss.normalize_L2(q)
                scores, ids = self.index.search(q, min(k, len(self.chunks)))
                return [
                    {**self.chunks[i], "score": round(float(s), 4), "method": "vector"}
                    for s, i in zip(scores[0], ids[0])
                    if i != -1
                ]
            except Exception as e:
                print(f"[rag:{self.name}] WARNING query embedding failed ({e}); keyword fallback")
        return self._keyword_search(query, k)

    def _keyword_search(self, query: str, k: int) -> list[dict]:
        stop = {"the", "and", "for", "you", "your", "are", "was", "with", "how", "can", "what",
                "that", "this", "have", "has", "not", "but", "from", "when", "why", "does"}
        words = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in stop}
        q = words(query)
        if not q:
            return []
        scored = []
        for c in self.chunks:
            overlap = len(q & words(c["text"])) / len(q)
            if overlap > 0:
                scored.append({**c, "score": round(overlap, 4), "method": "keyword"})
        return sorted(scored, key=lambda h: h["score"], reverse=True)[:k]

    @staticmethod
    def is_relevant(hits: list[dict]) -> bool:
        """True if the best hit is strong enough to answer from. False -> 'no match'."""
        if not hits:
            return False
        limit = config.RELEVANCE_THRESHOLD if hits[0]["method"] == "vector" else config.KEYWORD_THRESHOLD
        return hits[0]["score"] >= limit


# ----------------------------------------------------------------------------
# One shared instance per RAG, per process
# ----------------------------------------------------------------------------
_stores: dict[str, FaissStore] = {}


def get_store(name: str) -> FaissStore:
    """name is 'catalog' (RAG 1) or 'kb' (RAG 2)."""
    if name not in _stores:
        dirs = {"catalog": config.CATALOG_DIR, "kb": config.KB_DIR}
        _stores[name] = FaissStore(name, dirs[name])
    return _stores[name]
