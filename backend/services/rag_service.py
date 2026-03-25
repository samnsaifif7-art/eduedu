"""
RAG Service — FAISS + Sentence Transformers + Groq / Llama-3.
Falls back to extractive answer if Groq key is not set.
"""
from __future__ import annotations

import io
import time
import logging
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("edumind.rag")

# Add project root to sys.path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class RAGService:
    def __init__(self):
        self._model = None          # SentenceTransformer
        self._index = None          # faiss index
        self._chunks: list[str] = []
        self._sources: list[str] = []
        self._groq_client = None
        self._ready = False

    # ── Lazy init ────────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return
        logger.info("Loading sentence-transformers model …")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("✅ Sentence-transformers ready.")

    def _load_groq(self):
        if self._groq_client is not None:
            return
        if not config.GROQ_API_KEY:
            logger.warning("No GROQ_API_KEY — will use extractive fallback.")
            return
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=config.GROQ_API_KEY)
            logger.info("✅ Groq/Llama-3 ready.")
        except Exception as exc:
            logger.warning("Groq init failed: %s — using extractive fallback.", exc)

    def warmup(self):
        """Pre-load models at server startup."""
        self._load_model()
        self._load_groq()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_pdf(self, file_bytes: bytes, filename: str) -> int:
        """Parse PDF → chunk → embed → FAISS index. Returns chunk count."""
        self._load_model()
        import faiss
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        full_text = " ".join(
            page.extract_text() or "" for page in reader.pages
        )

        chunks = self._make_chunks(full_text)
        if not chunks:
            raise ValueError("Could not extract text from PDF.")

        embeddings = self._model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Merge with existing index if any
        if self._index is not None:
            combined = faiss.IndexFlatIP(dim)
            # re-add old embeddings not feasible without storing — reset instead
            logger.info("Replacing existing index with new document.")

        self._index = index
        self._chunks = chunks
        if filename not in self._sources:
            self._sources.append(filename)
        self._ready = True
        logger.info("Indexed %d chunks from '%s'.", len(chunks), filename)
        return len(chunks)

    def _make_chunks(self, text: str) -> list[str]:
        words = text.split()
        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
        return [
            " ".join(words[i : i + config.CHUNK_SIZE])
            for i in range(0, len(words), step)
            if words[i : i + config.CHUNK_SIZE]
        ]

    # ── Query ─────────────────────────────────────────────────────────────────

    def ask(self, question: str, top_k: int = 3) -> dict:
        if not self._ready:
            raise RuntimeError("No document indexed yet. Please ingest a PDF first.")

        self._load_model()
        t0 = time.perf_counter()

        q_emb = self._model.encode(
            [question], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self._index.search(q_emb, min(top_k, len(self._chunks)))
        top_chunks = [self._chunks[i] for i in indices[0] if i < len(self._chunks)]
        confidence = float(np.mean(scores[0])) if scores[0].size else 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        context = "\n\n".join(top_chunks)
        answer, model_used = self._generate_answer(question, context)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer,
            "sources": list(self._sources),
            "confidence": round(confidence, 3),
            "model_used": model_used,
            "response_time_ms": round(elapsed_ms, 1),
        }

    def _generate_answer(self, question: str, context: str) -> tuple[str, str]:
        self._load_groq()
        if self._groq_client:
            try:
                prompt = (
                    "You are EduBot, an AI tutor for school students. "
                    "Answer the question using ONLY the provided context. "
                    "Be clear, concise, and student-friendly.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\nAnswer:"
                )
                chat = self._groq_client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.3,
                )
                return chat.choices[0].message.content.strip(), "Llama-3 (Groq)"
            except Exception as exc:
                logger.warning("Groq call failed: %s — falling back.", exc)

        # Extractive fallback: return most relevant sentence
        sentences = context.replace("\n", " ").split(". ")
        q_words = set(question.lower().split())
        scored = sorted(
            sentences,
            key=lambda s: len(q_words & set(s.lower().split())),
            reverse=True,
        )
        answer = ". ".join(scored[:3]).strip()
        if not answer:
            answer = "I could not find a relevant answer in the provided document."
        return answer, "Extractive (offline)"

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "indexed": self._ready,
            "chunks_count": len(self._chunks),
            "sources": list(self._sources),
        }


rag_service = RAGService()
