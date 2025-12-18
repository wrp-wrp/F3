#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Optional


@dataclass
class Run:
    meta: dict
    iters: list[dict] = field(default_factory=list)
    recall: Optional[dict] = None


def pctl(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    idx = int(round((p / 100.0) * (len(xs) - 1)))
    idx = max(0, min(len(xs) - 1, idx))
    return xs[idx]


def read_runs(jsonl_path: Path) -> list[Run]:
    runs: list[Run] = []
    current: Run | None = None
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("event") == "meta":
            current = Run(meta=obj)
            runs.append(current)
            continue
        if current is None:
            continue
        if obj.get("event") == "recall":
            current.recall = obj
            continue
        if "iteration" in obj:
            current.iters.append(obj)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "jsonl",
        nargs="?",
        default="results/p0_min_matrix/runs.jsonl",
        help="JSONL produced by exp_scripts/p0_min_matrix.sh",
    )
    args = ap.parse_args()

    path = Path(args.jsonl)
    runs = read_runs(path)
    if not runs:
        raise SystemExit(f"no runs found in {path}")

    # CSV header (flat, one row per run)
    header = [
        "engine",
        "index_name",
        "posting_codec",
        "nlist",
        "dim",
        "nq",
        "k",
        "nprobe",
        "cache_enabled",
        "host_dist",
        "decoded_cache_budget_bytes",
        "profile_stages",
        "artifact_native_decoded_cache",
        "wall_ms_avg",
        "wall_ms_p50",
        "wall_ms_p95",
        "wall_ms_p99",
        "recall_at_k",
        "recall_queries",
    ]
    print(",".join(header))

    for r in runs:
        meta = r.meta
        engine = meta.get("engine", "")
        idx = meta.get("index_name", "")
        codec = meta.get("posting_codec", meta.get("artifact_posting_codec", ""))
        nlist = meta.get("nlist", "")
        dim = meta.get("dim", "")
        nq = meta.get("nq", "")
        k = meta.get("k", "")
        nprobe = meta.get("nprobe", "")
        cache_enabled = meta.get("cache_enabled", "")
        host_dist = meta.get("host_dist", "")
        decoded_budget = meta.get("decoded_cache_budget_bytes", "")
        profile = meta.get("profile_stages", "")
        native_decoded_cache = meta.get("artifact_native_decoded_cache", "")

        wall = [float(it.get("wall_ms", "nan")) for it in r.iters if "wall_ms" in it]
        wall_avg = mean(wall) if wall else float("nan")
        wall_p50 = pctl(wall, 50)
        wall_p95 = pctl(wall, 95)
        wall_p99 = pctl(wall, 99)

        recall_at_k = ""
        recall_queries = ""
        if r.recall:
            recall_at_k = r.recall.get("recall_at_k", "")
            recall_queries = r.recall.get("recall_queries", "")

        row = [
            engine,
            str(idx),
            str(codec),
            str(nlist),
            str(dim),
            str(nq),
            str(k),
            str(nprobe),
            str(cache_enabled),
            str(host_dist),
            str(decoded_budget),
            str(profile),
            str(native_decoded_cache),
            f"{wall_avg:.6f}" if wall else "",
            f"{wall_p50:.6f}" if wall else "",
            f"{wall_p95:.6f}" if wall else "",
            f"{wall_p99:.6f}" if wall else "",
            str(recall_at_k),
            str(recall_queries),
        ]
        print(",".join(row))


if __name__ == "__main__":
    main()
