"""
Benchmarking harness. For a given model + dataloader, measures:
  - peak memory (MB, process RSS delta during inference)
  - per-image latency (ms)
  - throughput (images/sec)
  - accuracy and macro F1

Run standalone for a single config, or import run_benchmark() from
run_sweep.py to sweep quantization level x batch size x resolution.
"""

import gc
import json
import multiprocessing as mp
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil


_PROC = psutil.Process()


def _rss_mb() -> float:
    """Total resident memory, INCLUDING shared-library code pages."""
    return _PROC.memory_info().rss / 2**20


def _data_mb() -> float:
    """Private data memory: weights, activations, buffers — EXCLUDING library
    code pages. (Plain RSS also counts the PyTorch kernel code that gets paged
    in the first time a quantized op runs, which made a 2 MB int8 model look
    like it "used" ~40 MB.) Linux: RssAnon; Windows: private bytes; macOS: RSS."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    mi = _PROC.memory_info()
    return getattr(mi, "private", mi.rss) / 2**20


def _trim_heap():
    """Return freed heap pages to the OS (Linux) so RSS reflects live memory."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class PeakSampler(threading.Thread):
    """Samples private data memory and total RSS every ~1 ms; keeps the maxima."""

    def __init__(self, interval=0.001):
        super().__init__(daemon=True)
        self.interval, self._halt = interval, threading.Event()
        self.peak, self.peak_rss = _data_mb(), _rss_mb()

    def run(self):
        while not self._halt.is_set():
            self.peak = max(self.peak, _data_mb())
            self.peak_rss = max(self.peak_rss, _rss_mb())
            time.sleep(self.interval)

    def stop(self):
        self._halt.set()
        self.join()
        self.peak = max(self.peak, _data_mb())
        self.peak_rss = max(self.peak_rss, _rss_mb())
        return self.peak


def _worker(cfg: dict, q):
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import torch
        torch.set_num_threads(cfg["threads"])
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        from quantize import load_model_for_level, set_backend
        set_backend(cfg.get("backend"))

        x_all = torch.load(cfg["timing_tensor"], weights_only=True)
        if cfg["level"] == "fp16":
            x_all = x_all.half()
        bs = cfg["batch_size"]
        n = (x_all.shape[0] // bs) * bs
        batches = [x_all[i:i + bs].contiguous() for i in range(0, n, bs)]

        gc.collect()
        _trim_heap()
        time.sleep(0.2)
        rss_base, rss_total_base = _data_mb(), _rss_mb()

        if cfg["model_path"].endswith(".ts"):
            model = torch.jit.load(cfg["model_path"]).eval()      # deployment format (all levels)
        else:
            model = load_model_for_level(cfg["arch"], cfg["num_classes"], cfg["level"], cfg["model_path"],
                                         cfg.get("backend"))
        gc.collect()
        _trim_heap()
        rss_loaded = _data_mb()

        sampler = PeakSampler()
        sampler.start()
        with torch.inference_mode():
            for b in batches[: max(1, cfg["warmup_batches"])]:
                model(b)
            trial_thr, trial_lat_img, batch_times = [], [], []
            t_deadline = time.perf_counter() + cfg["max_seconds"]
            while True:
                t_trial = 0.0
                for b in batches:
                    t0 = time.perf_counter()
                    model(b)
                    dt = time.perf_counter() - t0
                    t_trial += dt
                    batch_times.append(dt * 1000)
                trial_thr.append(n / t_trial)
                trial_lat_img.append(t_trial / n * 1000)
                k = len(trial_thr)
                if k >= cfg["min_trials"]:
                    rel_sem = statistics.stdev(trial_thr) / (k ** 0.5) / statistics.mean(trial_thr)
                    if rel_sem < cfg["target_rel_sem"] or k >= cfg["max_trials"] \
                            or time.perf_counter() > t_deadline:
                        break
        peak = sampler.stop()

        bt = sorted(batch_times)
        q.put({
            "ok": True,
            "rss_baseline_mb": rss_base,
            "model_load_mb": rss_loaded - rss_base,
            "peak_inference_mb": peak - rss_base,
            "activation_mb": max(0.0, peak - rss_loaded),
            "peak_rss_total_mb": sampler.peak_rss,
            "peak_rss_incl_code_mb": sampler.peak_rss - rss_total_base,
            "n_trials": len(trial_thr),
            "trial_throughputs": trial_thr,
            "throughput_images_per_sec": statistics.mean(trial_thr),
            "throughput_std": statistics.stdev(trial_thr) if len(trial_thr) > 1 else 0.0,
            "latency_ms_per_image": statistics.mean(trial_lat_img),
            "latency_ms_per_image_std": statistics.stdev(trial_lat_img) if len(trial_lat_img) > 1 else 0.0,
            "batch_latency_ms_mean": statistics.mean(batch_times),
            "batch_latency_ms_p50": bt[len(bt) // 2],
            "batch_latency_ms_p95": bt[min(len(bt) - 1, int(0.95 * len(bt)))],
            "torch_threads": torch.get_num_threads(),
            "quant_engine": torch.backends.quantized.engine,
        })
    except Exception as e:  # report instead of hanging the sweep
        import traceback
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})


def run_in_fresh_process(cfg: dict, timeout: float = 1800) -> dict:
    """Run one benchmark config in a clean spawned interpreter."""
    defaults = dict(threads=os.cpu_count() or 1, warmup_batches=3, min_trials=5, max_trials=30,
                    target_rel_sem=0.02, max_seconds=120, backend=None)
    cfg = {**defaults, **cfg}
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(cfg, q))
    proc.start()
    try:
        res = q.get(timeout=timeout)
    except Exception as e:
        res = {"ok": False, "error": f"timeout/no result: {e}"}
    proc.join(10)
    if proc.is_alive():
        proc.terminate()
    return res


if __name__ == "__main__":
    # Standalone: benchmark one saved model.
    #   python src/benchmark.py --arch mobilenet_v2 --level int8 \
    #       --model_path models/mobilenet_v2_int8_qat_res224.pt --num_classes 22 \
    #       --timing_tensor results/cache/timing_res224.pt --batch_size 1
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--timing_tensor", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    print(json.dumps(run_in_fresh_process(vars(a)), indent=2))
