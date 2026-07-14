from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import settings


@dataclass
class KBChunk:
    doc_path: str
    title: str
    content: str
    section: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _split_markdown(path: Path, root: Path) -> list[KBChunk]:
    text = path.read_text(encoding="utf-8")
    rel_path = str(path.relative_to(root)).replace("\\", "/")
    title = _extract_title(text, path.stem)
    sections = re.split(r"\n---+\n", text)
    chunks: list[KBChunk] = []
    for i, section in enumerate(sections):
        section = section.strip()
        if len(section) < 40:
            continue
        section_title = title
        for line in section.splitlines():
            if line.startswith("## "):
                section_title = line[3:].strip()
                break
        chunks.append(
            KBChunk(
                doc_path=rel_path,
                title=title,
                content=section,
                section=section_title,
            )
        )
    return chunks


class KnowledgeBaseRetriever:
    """BM25 retrieval over knowledge-base markdown chunks."""

    def __init__(self, kb_dir: Path | None = None, top_k: int = 4):
        self.kb_dir = kb_dir or settings.knowledge_base_dir
        self.top_k = top_k
        self.chunks: list[KBChunk] = []
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []
        self._build_index()

    def _build_index(self) -> None:
        for md_path in sorted(self.kb_dir.rglob("*.md")):
            self.chunks.extend(_split_markdown(md_path, self.kb_dir))
        self._corpus_tokens = [_tokenize(c.content) for c in self.chunks]
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)

    def retrieve(self, query: str, top_k: int | None = None) -> list[KBChunk]:
        k = top_k or self.top_k
        if not self._bm25 or not self.chunks:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return self.chunks[:k]
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0] or [
            self.chunks[i] for i in ranked[:k]
        ]

    def format_context(self, query: str, top_k: int | None = None) -> str:
        chunks = self.retrieve(query, top_k)
        if not chunks:
            return "No relevant knowledge-base excerpts found."
        parts = []
        for i, chunk in enumerate(chunks, 1):
            excerpt = chunk.content[:1200]
            parts.append(
                f"[{i}] {chunk.doc_path} — {chunk.section}\n{excerpt}"
            )
        return "\n\n".join(parts)


_retriever: KnowledgeBaseRetriever | None = None


def get_retriever() -> KnowledgeBaseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeBaseRetriever()
    return _retriever
