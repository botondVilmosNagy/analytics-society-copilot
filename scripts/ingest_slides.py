from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:
    try:
        from langchain_community.vectorstores import Chroma
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Chroma integration not found. Install 'langchain-chroma' or 'langchain-community'."
        ) from exc

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Analytics and Society slide PDF into Chroma vector database."
    )
    parser.add_argument(
        "--pdf-path",
        default=os.getenv("SLIDES_PDF_PATH", "data/slides/analytics_and_society_slides.pdf"),
        help="Path to the combined course slides PDF file.",
    )
    parser.add_argument(
        "--course-name",
        default=os.getenv("COURSE_NAME", "Analytics and Society"),
        help="Course name metadata stored with each chunk.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "1200")),
        help="Chunk size used for text splitting.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "180")),
        help="Chunk overlap used for text splitting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("INGEST_BATCH_SIZE", "64")),
        help="Insert chunks in batches to avoid memory spikes.",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete existing vectors in the target collection before ingesting.",
    )
    return parser.parse_args()


def _normalized_lines(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def _extract_week(text: str, previous_week: str) -> str:
    patterns = [
        r"\bweek\s*(\d{1,2})\b",
        r"\bblock\s*(\d{1,2})\b",
        r"\blecture\s*(\d{1,2})\b",
    ]
    lowered = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return previous_week


def _extract_lecture_title(text: str, previous_title: str) -> str:
    lines = _normalized_lines(text)
    if not lines:
        return previous_title

    for line in lines[:8]:
        cleaned = re.sub(r"\s+", " ", line).strip(" -:|")
        if not cleaned:
            continue
        if len(cleaned) < 8 or len(cleaned) > 140:
            continue
        if cleaned.lower().startswith(("source", "http", "www")):
            continue
        return cleaned

    return previous_title


def _infer_slide_metadata(page_text: str, prev_week: str, prev_title: str) -> Tuple[str, str]:
    week = _extract_week(page_text, prev_week)
    title = _extract_lecture_title(page_text, prev_title)
    return week, title


def extract_documents(pdf_path: Path, course_name: str) -> List[Document]:
    reader = PdfReader(str(pdf_path))
    docs: List[Document] = []
    current_week = "?"
    current_lecture_title = "Unknown lecture"

    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        current_week, current_lecture_title = _infer_slide_metadata(
            text,
            current_week,
            current_lecture_title,
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "course_name": course_name,
                    "source": str(pdf_path),
                    "slide_number": page_index + 1,
                    "course_week": current_week,
                    "lecture_title": current_lecture_title,
                },
            )
        )

    return docs


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def batched(items: List[Document], size: int) -> List[List[Document]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    load_dotenv()
    args = parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Slides PDF not found at {pdf_path}. Put your file there or pass --pdf-path."
        )

    chroma_persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "analytics_and_society_slides")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    print(f"Reading PDF: {pdf_path}")
    page_docs = extract_documents(pdf_path=pdf_path, course_name=args.course_name)
    print(f"Extracted text from {len(page_docs)} pages.")

    print("Splitting text into chunks...")
    chunks = split_documents(page_docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"Generated {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_persist_directory,
    )

    if args.reset_collection:
        print("Resetting existing collection...")
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name=chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_persist_directory,
        )

    print(
        f"Writing to Chroma collection '{chroma_collection_name}' in '{chroma_persist_directory}'..."
    )
    for batch in batched(chunks, args.batch_size):
        vectorstore.add_documents(batch)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
