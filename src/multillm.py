# multillm.py
from __future__ import annotations

import os
import subprocess
import logging
import json
from multiprocessing import Process, Queue
from typing import Tuple, Dict, List
from src.llm_response_cache import (
    key_text,
    cache_get_text,
    cache_set_text,
)
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _num_gpus() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip()])
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
        return sum(1 for l in out.splitlines() if l.startswith("GPU "))
    except Exception:
        return 0


@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    max_context_tokens: int
    tp: int = 4
    gpu_memory_utilization: float = 0.90


def _worker(worker_id: int, cuda_visible_devices: str, cfg: LLMConfig, inq: Queue, outq: Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    logger.info(f"Worker {worker_id} starting on CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg.model_name,
        tensor_parallel_size=cfg.tp,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_context_tokens,
    )

    logger.info(f"Worker {worker_id} LLM initialized")

    while True:
        item = inq.get()
        if item is None:
            logger.info(f"Worker {worker_id} shutting down")
            return

        req_id, prompts, sp = item
        logger.info(f"Worker {worker_id} processing request {req_id} with {len(prompts)} prompt(s)")

        n = int(sp.get("n", 1) or 1)
        outputs = llm.generate(list(prompts), SamplingParams(**sp))
        if n > 1:
            texts = [[cand.text for cand in o.outputs] for o in outputs]
        else:
            texts = [[(o.outputs[0].text if o.outputs else "")] for o in outputs]
        outq.put((req_id, worker_id, texts))


class MultiLLM:
    llm_cfg: LLMConfig

    def __init__(
        self,
        model_name: str,
        max_context_tokens: int,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.90,
    ):
        self.llm_cfg = LLMConfig(
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            tp=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self.inqs: List[Queue] = []
        self.outq: Queue = Queue()
        self.procs: List[Process] = []
        self._req = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        tp = self.llm_cfg.tp
        g = _num_gpus()
        n = g // tp
        if n <= 0:
            raise RuntimeError(f"Need >= {tp} GPUs, found {g}")

        logger.info(f"Detected {g} GPU(s), starting {n} worker(s)")

        for i in range(n):
            devs = ",".join(str(i * tp + j) for j in range(tp))
            inq = Queue()
            p = Process(target=_worker, args=(i, devs, self.llm_cfg, inq, self.outq))
            p.start()

            logger.info(f"Started worker pid={p.pid} on GPUs [{devs}]")

            self.inqs.append(inq)
            self.procs.append(p)

    def close(self):
        logger.info("Shutting down workers")
        for q in self.inqs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            try:
                p.join(timeout=2)
            except Exception:
                pass

    def get_responses(
        self,
        prompts: Tuple[str, ...],
        sampling_params: Dict,
    ) -> List[List[str]]:
        if not prompts:
            return []

        if not self.procs:
            raise RuntimeError("MultiLLM not started")

        model_name = self.llm_cfg.model_name
        n_samples = int(sampling_params.get("n", 1) or 1)

        keys = [key_text(model=model_name, prompt=p, **sampling_params) for p in prompts]

        cached: List[List[str] | None] = []
        for k in keys:
            v = cache_get_text(k)
            if v is None:
                cached.append(None)
            else:
                if n_samples > 1:
                    cached.append(list(json.loads(v)))
                else:
                    cached.append([v])

        missing_idx = [i for i, v in enumerate(cached) if v is None]

        if not missing_idx:
            logger.info("All prompts served from cache")
        else:
            missing_prompts = tuple(prompts[i] for i in missing_idx)

            logger.info(
                f"Cache hits: {len(prompts) - len(missing_prompts)}/{len(prompts)}, "
                f"dispatching {len(missing_prompts)} prompt(s)"
            )

            req_id = self._req
            self._req += 1

            n = len(self.procs)
            buckets = [[] for _ in range(n)]
            for j, pr in enumerate(missing_prompts):
                buckets[j % n].append((j, pr))

            for w, items in enumerate(buckets):
                if items:
                    _, ps = zip(*items)
                    self.inqs[w].put((req_id, ps, sampling_params))

            generated: List[List[str] | None] = [None] * len(missing_prompts)
            remaining = sum(1 for b in buckets if b)

            while remaining:
                rid, wid, texts = self.outq.get()
                if rid != req_id:
                    continue

                items = buckets[wid]
                if not items:
                    continue

                idxs = [i for i, _ in items]
                for i, t in zip(idxs, texts):
                    generated[i] = list(t)

                buckets[wid] = []
                remaining -= 1

            for local_i, global_i in enumerate(missing_idx):
                val = generated[local_i] or []
                if n_samples > 1:
                    cached[global_i] = val
                    cache_set_text(keys[global_i], json.dumps(list(val)))
                else:
                    text = val[0] if val else ""
                    cached[global_i] = [text]
                    cache_set_text(keys[global_i], text)

        logger.debug(f"EXAMPLE:\n\nPROMPT:\n\n{prompts[0]}\n\nRESPONSE:\n\n{cached[0]}")
        logger.debug(
            "Cache summary: %d/%d hits",
            len(prompts) - len(missing_idx),
            len(prompts),
        )

        return [v if v is not None else [""] for v in cached]
