from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    try:
        from langchain_community.vectorstores import Chroma
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Chroma integration not found. Install 'langchain-chroma' or 'langchain-community'."
        ) from exc

from langchain_openai import OpenAIEmbeddings


@dataclass
class RetrievalChunk:
    content: str
    metadata: Dict[str, Any]
    score: float


class SlideRetriever:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_model: str,
        top_k: int,
        min_relevance_score: float,
    ) -> None:
        embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score

    def search(
        self,
        query: str,
        course_week_filter: str | None = None,
    ) -> List[RetrievalChunk]:
        search_kwargs: Dict[str, Any] = {"k": self.top_k}
        if course_week_filter:
            search_kwargs["filter"] = {"course_week": course_week_filter}

        chunks = self._search_with_relevance_scores(query, search_kwargs)
        if chunks:
            return chunks

        return self._search_with_distance_scores(query, search_kwargs)

    def _search_with_relevance_scores(
        self,
        query: str,
        search_kwargs: Dict[str, Any],
    ) -> List[RetrievalChunk]:
        chunks: List[RetrievalChunk] = []
        try:
            raw: List[Tuple[Any, float]] = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                **search_kwargs,
            )
        except Exception:
            return chunks

        for doc, relevance in raw:
            rel = max(0.0, min(1.0, float(relevance)))
            if rel < self.min_relevance_score:
                continue
            chunks.append(
                RetrievalChunk(
                    content=doc.page_content,
                    metadata=doc.metadata or {},
                    score=rel,
                )
            )
        return chunks

    def _search_with_distance_scores(
        self,
        query: str,
        search_kwargs: Dict[str, Any],
    ) -> List[RetrievalChunk]:
        raw: List[Tuple[Any, float]] = self.vectorstore.similarity_search_with_score(
            query,
            **search_kwargs,
        )

        chunks: List[RetrievalChunk] = []
        for doc, distance in raw:
            # For distance-like scores, smaller is better. This maps distance to
            # a bounded relevance score in [0, 1] where higher is better.
            rel = 1.0 / (1.0 + math.exp(float(distance) - 1.0))
            if rel < self.min_relevance_score:
                continue
            chunks.append(
                RetrievalChunk(
                    content=doc.page_content,
                    metadata=doc.metadata or {},
                    score=rel,
                )
            )
        return chunks


def format_context(chunks: List[RetrievalChunk]) -> str:
    if not chunks:
        return "No relevant slide context found."

    lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        lecture = chunk.metadata.get("lecture_title", "Unknown lecture")
        slide = chunk.metadata.get("slide_number", "?")
        source = f"[Doc {idx} | {lecture} | Slide {slide} | score={chunk.score:.2f}]"
        lines.append(source)
        lines.append(chunk.content.strip())
        lines.append("")

    return "\n".join(lines).strip()


def resolve_env() -> Dict[str, Any]:
    return {
        "persist_directory": os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
        "collection_name": os.getenv("CHROMA_COLLECTION_NAME", "analytics_and_society_slides"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "top_k": int(os.getenv("TOP_K", "6")),
        "min_relevance_score": float(os.getenv("MIN_RELEVANCE_SCORE", "0.0")),
        "course_week_filter": os.getenv("COURSE_WEEK_FILTER", "").strip() or None,
    }
