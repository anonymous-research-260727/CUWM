import math
import time
import threading
from typing import List, Optional
import hashlib
from examples.ui_world_model.cloudgpt_aoai import get_openai_client
from collections import Counter, OrderedDict
from typing import Any, Dict, Optional, List, Tuple
# ---- optional local deps ----
# Prefer sentence-transformers if available
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False

# Fallback to transformers if sentence-transformers not available
try:
    import torch
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    _HAS_TF = True
except Exception:
    torch = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    _HAS_TF = False

def _stable_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        if p is None:
            p = ""
        if not isinstance(p, str):
            p = str(p)
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()

class _LocalEmbedder:
    """
    Lazy-loaded local embedding backend.
    Uses sentence-transformers if installed, else transformers mean-pooling.
    Thread-safe lazy init.
    """
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, max_length: int = 512):
        self.model_name_or_path = model_name_or_path
        self.device = device  # e.g. "cuda", "cpu", None->auto
        self.max_length = max_length

        self._lock = threading.Lock()
        self._ready = False

        # ST path
        self._st_model = None

        # transformers path
        self._tok = None
        self._mdl = None

    def _ensure_loaded(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return

            if _HAS_ST:
                # SentenceTransformer handles pooling internally
                dev = self.device
                # device can be None; sentence-transformers auto chooses
                self._st_model = SentenceTransformer(self.model_name_or_path, device=dev)
                self._ready = True
                return

            if not _HAS_TF:
                raise RuntimeError(
                    "Local embedding fallback requested, but neither sentence-transformers nor transformers is available. "
                    "Please install one of them."
                )

            if self.device is None:
                dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            else:
                dev = self.device

            self._tok = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
            self._mdl = AutoModel.from_pretrained(self.model_name_or_path)
            self._mdl.to(dev)
            self._mdl.eval()
            self._device_resolved = dev
            self._ready = True

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        # last_hidden_state: [B, T, H], attention_mask: [B, T]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
        summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
        denom = mask.sum(dim=1).clamp(min=1e-6)                         # [B,1]
        return summed / denom

    def embed_one(self, text: str) -> List[float]:
        self._ensure_loaded()

        if self._st_model is not None:
            vec = self._st_model.encode(
                [text],
                show_progress_bar=False,
                normalize_embeddings=False,  # we normalize ourselves统一逻辑
            )[0]
            vec = [float(x) for x in vec]
            return vec

        # transformers mean-pooling
        assert self._tok is not None and self._mdl is not None
        assert torch is not None

        enc = self._tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device_resolved) for k, v in enc.items()}

        with torch.no_grad():
            out = self._mdl(**enc)
            last = out.last_hidden_state
            pooled = self._mean_pool(last, enc["attention_mask"])  # [1,H]
            vec = pooled[0].detach().float().cpu().tolist()
            return [float(x) for x in vec]

class _LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._d = OrderedDict()
        self._lock = threading.Lock()

    def get(self, k: str):
        with self._lock:
            if k not in self._d:
                return None
            v = self._d.pop(k)
            self._d[k] = v
            return v

    def set(self, k: str, v: Any):
        with self._lock:
            if k in self._d:
                self._d.pop(k)
            self._d[k] = v
            if len(self._d) > self.max_size:
                self._d.popitem(last=False)
                
class EmbeddingClient:
    """
    Thread-safe embeddings wrapper:
    - API first (OpenAI-compatible)
    - retry + cache
    - if API fails -> fallback to local embedding model (lazy-loaded)
    """
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_retries: int = 2,
        backoff_sec: float = 0.8,
        timeout_sec: int = 60,
        max_cache_size: int = 20000,

        # local fallback config
        local_model_name_or_path: str = "/path/to/cuwm/model/Qwen/Qwen3-Embedding-4B",
        local_device: Optional[str] = None,     # "cuda" / "cpu" / None(auto)
        local_max_length: int = 512,

        # circuit breaker: when API keeps failing, skip API for some seconds
        api_circuit_break_sec: float = 60.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.timeout_sec = timeout_sec

        self._tls = threading.local()
        self._cache = _LRUCache(max_cache_size)

        # local fallback backend (lazy loaded)
        self._local = _LocalEmbedder(
            model_name_or_path=local_model_name_or_path,
            device=local_device,
            max_length=local_max_length,
        )

        # circuit breaker state
        self._api_down_until = 0.0
        self._api_down_lock = threading.Lock()
        self.api_circuit_break_sec = api_circuit_break_sec

    def _get_client(self):
        c = getattr(self._tls, "client", None)
        if c is None:
            c = get_openai_client()
            self._tls.client = c
        return c

    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
        return [float(x) / norm for x in vec]

    def _api_available_now(self) -> bool:
        with self._api_down_lock:
            return time.time() >= self._api_down_until

    def _trip_circuit_breaker(self):
        with self._api_down_lock:
            self._api_down_until = max(self._api_down_until, time.time() + self.api_circuit_break_sec)

    def embed(self, text: str) -> List[float]:
        # cache split: api/local (avoid local result blocking api recovery)
        key_api = _stable_hash("api", self.model, text)
        key_local = _stable_hash("local", "fallback", text)

        # if api is currently allowed, check api cache first
        if self._api_available_now():
            hit = self._cache.get(key_api)
            if hit is not None:
                return hit
        else:
            # api circuit-open -> go local fast path
            hit = self._cache.get(key_local)
            if hit is not None:
                return hit

        last_err = None

        # ---- try API first (unless circuit open) ----
        if self._api_available_now():
            for attempt in range(self.max_retries + 1):
                try:
                    client = self._get_client()
                    # NOTE: if you need timeout, you'd configure it inside get_openai_client()
                    resp = client.embeddings.create(model=self.model, input=text)
                    vec = resp.data[0].embedding
                    out = self._l2_normalize([float(x) for x in vec])
                    self._cache.set(key_api, out)
                    return out
                except Exception as e:
                    last_err = e
                    if attempt < self.max_retries:
                        time.sleep(self.backoff_sec * (2 ** attempt))
                        continue
                    # trip breaker after exhausting retries
                    self._trip_circuit_breaker()
                    break

        # ---- fallback to local ----
        try:
            hit = self._cache.get(key_local)
            if hit is not None:
                return hit

            vec_local = self._local.embed_one(text)
            out_local = self._l2_normalize(vec_local)
            self._cache.set(key_local, out_local)
            return out_local
        except Exception as e2:
            # both failed -> raise with combined info
            raise RuntimeError(
                f"Embedding failed. api_err={repr(last_err)} local_err={repr(e2)}"
            )
