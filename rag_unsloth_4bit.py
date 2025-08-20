
"""
RAG with Unsloth Dynamic 4-bit Quantization (Single File)

- Colab-ready, minimal external wiring required.
- Implements components defined in design.md and requirements.md:
  * UnslothModelManager
  * DocumentProcessor
  * VectorRetriever
  * RAGPipeline
  * Data models & memory monitor
  * Colab utilities (install, demo)

Notes
-----
- This module assumes availability of: unsloth, transformers, sentence-transformers, faiss-cpu, bitsandbytes, torch, numpy.
- Safe imports + graceful fallbacks are included.
- Use in Colab:
    from rag_unsloth_4bit import *
    install_dependencies()        # optional helper
    pipeline = RAGPipeline.default_demo()  # builds a small end-to-end demo
    print(pipeline.process_query("What is this pipeline about?"))
"""

from __future__ import annotations

import os
import gc
import json
import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

# --- Safe imports ---
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# sentence-transformers
_ST_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    pass

# transformers + bitsandbytes (optional for CPU fallback if absent)
_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    pass

# unsloth (optional - preferred path)
_UNSLOTH_AVAILABLE = False
try:
    # Unsloth fast loader API
    from unsloth import FastLanguageModel  # type: ignore
    _UNSLOTH_AVAILABLE = True
except Exception:
    pass

# bitsandbytes quantization config (optional - used by Unsloth or vanilla transformers path)
_BNB_AVAILABLE = False
try:
    from transformers import BitsAndBytesConfig  # type: ignore
    _BNB_AVAILABLE = True
except Exception:
    pass

import numpy as np

# --------- Logging ----------
logger = logging.getLogger("rag_unsloth_4bit")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ==============================
# Data Models
# ==============================

@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray]
    source_document: str
    chunk_index: int


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    relevance_score: float
    rank: int


@dataclass
class RAGResponse:
    query: str
    response: str
    retrieved_chunks: List[RetrievalResult]
    generation_metadata: Dict[str, Any]
    memory_usage: Dict[str, Any]


# ==============================
# Configs
# ==============================

@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"  # "float16" or "bfloat16"
    keep_attn_ln_fp16: bool = True  # selective precision preservation


@dataclass
class RetrievalConfig:
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    similarity_metric: str = "dot"  # "dot" or "l2"
    relevance_threshold: Optional[float] = None
    chunk_size: int = 800
    overlap: int = 200
    max_context_tokens: int = 2000


# ==============================
# Memory Monitor
# ==============================

class MemoryMonitor:
    @staticmethod
    def vram() -> Dict[str, Any]:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"device": "cpu", "total_gb": None, "allocated_gb": None, "reserved_gb": None}
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        return {
            "device": torch.cuda.get_device_name(device),
            "total_gb": round(total, 3),
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
        }

    @staticmethod
    def cleanup():
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================
# Unsloth Model Manager
# ==============================

class UnslothModelManager:
    def __init__(self, model_name: str, qconf: Optional[QuantizationConfig] = None, use_chat_template: bool = True):
        self.model_name = model_name
        self.qconf = qconf or QuantizationConfig()
        self.use_chat_template = use_chat_template

        self.tokenizer = None
        self.model = None
        self.load_notes = {}

    def _bnb_config(self) -> Optional["BitsAndBytesConfig"]:
        if not _BNB_AVAILABLE:
            return None
        compute_dtype = torch.bfloat16 if self.qconf.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=self.qconf.load_in_4bit,
            bnb_4bit_use_double_quant=self.qconf.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.qconf.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def load_quantized_model(self) -> Tuple[Any, Any]:
        """
        Loads a dynamic 4-bit quantized model with Unsloth if available,
        else falls back to vanilla transformers with bnb 4-bit, else CPU fp16.
        """
        logger.info("Loading quantized model: %s", self.model_name)
        mem_before = MemoryMonitor.vram()

        # Preferred: Unsloth fast loader
        if _UNSLOTH_AVAILABLE:
            try:
                # Unsloth FastLanguageModel handles 4-bit loading internally via bitsandbytes.
                max_seq_length = 4096
                dtype = "bfloat16" if self.qconf.bnb_4bit_compute_dtype == "bfloat16" else "float16"
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=self.qconf.load_in_4bit,
                )
                # Optional: selective precision hints (no-op for some models; kept for clarity)
                self.load_notes["path"] = "unsloth.FastLanguageModel"
                self.model, self.tokenizer = model, tokenizer
                logger.info("Loaded via Unsloth FastLanguageModel.")
                return self.model, self.tokenizer
            except Exception as e:
                logger.warning("Unsloth path failed: %s", e)

        # Fallback: transformers + BitsAndBytes
        if _TRANSFORMERS_AVAILABLE and _BNB_AVAILABLE:
            try:
                bnb_cfg = self._bnb_config()
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if TORCH_AVAILABLE else None,
                    quantization_config=bnb_cfg if self.qconf.load_in_4bit else None,
                    torch_dtype=torch.bfloat16 if self.qconf.bnb_4bit_compute_dtype == "bfloat16" else torch.float16,
                )
                self.load_notes["path"] = "transformers+bnb"
                self.model, self.tokenizer = model, tokenizer
                logger.info("Loaded via transformers + bitsandbytes.")
                return self.model, self.tokenizer
            except Exception as e:
                logger.warning("Transformers+bnb path failed: %s", e)

        # Last fallback: CPU full/fp16-ish
        if _TRANSFORMERS_AVAILABLE:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.load_notes["path"] = "transformers_cpu_fp"
            self.model, self.tokenizer = model, tokenizer
            logger.info("Loaded CPU path (no quantization).")
            return self.model, self.tokenizer

        raise RuntimeError("No compatible model loading path found. Please install unsloth/transformers/bitsandbytes.")

    def get_memory_usage(self) -> Dict[str, Any]:
        return MemoryMonitor.vram()

    def optimize_memory_allocation(self):
        """
        Placeholder for advanced strategies: gradient checkpointing, KV cache control, etc.
        For inference-only pipelines, we ensure CUDA cache cleanliness.
        """
        MemoryMonitor.cleanup()


# ==============================
# Document Processing
# ==============================

class DocumentProcessor:
    def __init__(self, rconf: RetrievalConfig):
        self.rconf = rconf
        self.embedder = None
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self._ensure_embedder()

    def _ensure_embedder(self):
        if not _ST_AVAILABLE:
            raise RuntimeError("sentence-transformers not available. Please install it.")
        self.embedder = SentenceTransformer(self.rconf.embed_model_name)

    @staticmethod
    def _sliding_window_chunk(text: str, size: int, overlap: int) -> List[str]:
        tokens = text.split()
        if not tokens:
            return []
        chunks = []
        step = max(1, size - overlap)
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + size]
            if not chunk_tokens:
                break
            chunks.append(" ".join(chunk_tokens))
            if i + size >= len(tokens):
                break
        return chunks

    def chunk_documents(self, documents: List[Tuple[str, str]]) -> List[DocumentChunk]:
        """
        documents: List of (doc_id, text)
        """
        out: List[DocumentChunk] = []
        for doc_id, text in documents:
            parts = self._sliding_window_chunk(text, self.rconf.chunk_size, self.rconf.overlap)
            for idx, part in enumerate(parts):
                out.append(DocumentChunk(
                    id=str(uuid.uuid4()),
                    content=part,
                    metadata={"source": doc_id},
                    embedding=None,
                    source_document=doc_id,
                    chunk_index=idx,
                ))
        self.chunks = out
        logger.info("Chunked %d documents into %d chunks.", len(documents), len(out))
        return out

    def generate_embeddings(self, chunks: Optional[List[DocumentChunk]] = None) -> np.ndarray:
        chunks = chunks or self.chunks
        if not chunks:
            return np.zeros((0, 384), dtype="float32")
        texts = [c.content for c in chunks]
        # Batch encode
        embeddings = self.embedder.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
        # Normalize for dot-product similarity, if desired
        if self.rconf.similarity_metric == "dot":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
        for c, e in zip(chunks, embeddings):
            c.embedding = e.astype("float32")
        logger.info("Generated embeddings for %d chunks.", len(chunks))
        return embeddings.astype("float32")

    def create_index(self, embeddings: Optional[np.ndarray] = None):
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss not available. Please install faiss-cpu.")
        embeddings = embeddings if embeddings is not None else np.vstack([c.embedding for c in self.chunks if c.embedding is not None])
        if embeddings.size == 0:
            raise ValueError("No embeddings to index.")
        d = embeddings.shape[1]
        if self.rconf.similarity_metric == "dot":
            index = faiss.IndexFlatIP(d)  # inner product
        else:
            index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        self.index = index
        logger.info("Created FAISS index with %d vectors (dim=%d).", embeddings.shape[0], d)

    def _encode_query(self, query: str) -> np.ndarray:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        if self.rconf.similarity_metric == "dot":
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        return q_emb.astype("float32")


# ==============================
# Retriever
# ==============================

class VectorRetriever:
    def __init__(self, processor: DocumentProcessor):
        if processor.index is None:
            raise ValueError("Processor must have an index before creating a retriever.")
        self.processor = processor

    def retrieve_chunks(self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[RetrievalResult]:
        top_k = top_k or self.processor.rconf.top_k
        threshold = threshold if threshold is not None else self.processor.rconf.relevance_threshold

        q_emb = self.processor._encode_query(query).reshape(1, -1)
        index = self.processor.index
        if self.processor.rconf.similarity_metric == "dot":
            scores, idxs = index.search(q_emb, top_k)
        else:
            # For L2, smaller is better, convert to negative distances as scores
            dists, idxs = index.search(q_emb, top_k)
            scores = -dists

        results: List[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
            if idx < 0 or idx >= len(self.processor.chunks):
                continue
            if threshold is not None and score < threshold:
                continue
            results.append(RetrievalResult(
                chunk=self.processor.chunks[idx],
                relevance_score=float(score),
                rank=rank + 1,
            ))
        return results


# ==============================
# RAG Pipeline
# ==============================

class RAGPipeline:
    def __init__(
        self,
        model_manager: UnslothModelManager,
        processor: DocumentProcessor,
        retriever: VectorRetriever,
        rconf: RetrievalConfig,
    ):
        self.mm = model_manager
        self.processor = processor
        self.retriever = retriever
        self.rconf = rconf

    @staticmethod
    def _apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
        # If tokenizer has a chat template, use it; otherwise fallback.
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        return f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[USER]: {user_prompt}\n[ASSISTANT]:"

    def format_context(self, retrieved: List[RetrievalResult]) -> str:
        parts = []
        for r in retrieved:
            meta = json.dumps(r.chunk.metadata, ensure_ascii=False)
            parts.append(f"[Rank {r.rank} | Score {r.relevance_score:.4f} | Meta {meta}]\n{r.chunk.content}")
        text = "\n\n---\n\n".join(parts)
        # Token limit heuristic
        if len(text.split()) > self.rconf.max_context_tokens:
            text = " ".join(text.split()[: self.rconf.max_context_tokens]) + " ..."
        return text

    def _generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        if self.mm.model is None or self.mm.tokenizer is None:
            raise RuntimeError("Model not loaded.")
        tokenizer = self.mm.tokenizer
        model = self.mm.model

        inputs = tokenizer(prompt, return_tensors="pt")
        if TORCH_AVAILABLE:
            device = 0 if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(0.0, float(temperature)),
            top_p=0.95,
            repetition_penalty=1.05,
        )

        try:
            # Optional: stream to stdout in notebooks
            streamer = None
            if _TRANSFORMERS_AVAILABLE:
                try:
                    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    gen_kwargs["streamer"] = streamer
                except Exception:
                    pass

            with torch.no_grad() if TORCH_AVAILABLE else DummyContext():
                output_ids = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # If chat template included the prompt, trim it out:
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text.strip()
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return "Sorry, I couldn't generate a response due to an internal error."

    def generate_response(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a helpful assistant that answers *only* using the provided CONTEXT. "
            "Cite facts from the context. If context is insufficient, say so briefly."
        )
        user_prompt = f"Question: {query}\n\nCONTEXT:\n{context}"
        prompt = self._apply_chat_template(self.mm.tokenizer, system_prompt, user_prompt)
        return self._generate(prompt)

    def process_query(self, query: str, *, top_k: Optional[int] = None, temperature: float = 0.2) -> RAGResponse:
        retrieved = self.retriever.retrieve_chunks(query, top_k=top_k)
        context = self.format_context(retrieved) if retrieved else "(no relevant context retrieved)"
        response = self.generate_response(query, context)

        return RAGResponse(
            query=query,
            response=response,
            retrieved_chunks=retrieved,
            generation_metadata={
                "model_name": self.mm.model_name,
                "load_path": self.mm.load_notes.get("path", "unknown"),
                "temperature": temperature,
                "top_k": top_k or self.rconf.top_k,
            },
            memory_usage=self.mm.get_memory_usage(),
        )

    # --------- Convenience Builders (for Colab demos) ---------

    @classmethod
    def build(
        cls,
        *,
        model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        qconf: Optional[QuantizationConfig] = None,
        rconf: Optional[RetrievalConfig] = None,
        documents: Optional[List[Tuple[str, str]]] = None,
    ) -> "RAGPipeline":
        rconf = rconf or RetrievalConfig()
        dp = DocumentProcessor(rconf)
        if documents:
            dp.chunk_documents(documents)
        else:
            # Tiny default docs
            docs = [
                ("design.md", "The RAG pipeline uses Unsloth dynamic 4-bit quantization to run an LLM efficiently."
                               " It retrieves chunks from a vector store and generates grounded responses."),
                ("requirements.md", "Users can load quantized models, index documents, retrieve top-k chunks with scores,"
                                    " and generate responses grounded in context with memory monitoring."),
            ]
            dp.chunk_documents(docs)

        dp.generate_embeddings()
        dp.create_index()

        mm = UnslothModelManager(model_name=model_name, qconf=qconf)
        mm.load_quantized_model()
        retriever = VectorRetriever(dp)
        return cls(mm, dp, retriever, rconf)

    @classmethod
    def default_demo(cls) -> "RAGPipeline":
        return cls.build()


# ==============================
# Utilities
# ==============================

def install_dependencies():
    """
    Convenience function for Colab.
    """
    import subprocess, sys
    pkgs = [
        "unsloth",
        "transformers",
        "accelerate",
        "sentence-transformers",
        "faiss-cpu",
        "bitsandbytes",
        "torch",  # In Colab, torch is preinstalled with CUDA; this pip line can be skipped if desired.
    ]
    for p in pkgs:
        try:
            __import__(p.split("==")[0].split("[")[0])
            logger.info("Package already available: %s", p)
        except Exception:
            logger.info("Installing: %s", p)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])


class DummyContext:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


# ==============================
# __main__ demo (safe/no heavy calls by default)
# ==============================

if __name__ == "__main__":
    print("Module loaded. For a live demo in Colab:")
    print("  from rag_unsloth_4bit import *")
    print("  install_dependencies()  # optional")
    print("  pipeline = RAGPipeline.default_demo()")
    print("  print(pipeline.process_query('What is this pipeline about?').response)")
