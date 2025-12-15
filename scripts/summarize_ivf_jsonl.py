#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def percentile(values: List[float], p: float) -> float:
    if not values:
        raise ValueError("empty values")
    values = sorted(values)
    i = int(round(p * (len(values) - 1)))
    i = max(0, min(len(values) - 1, i))
    return values[i]


def summarize_one(path: str) -> Optional[Dict[str, Any]]:
    meta = None
    ms: List[float] = []
    series: Dict[str, List[float]] = {}
    last: Optional[Dict[str, Any]] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("event") == "meta":
                meta = o
                continue
            last = o
            if "wall_ms" in o:
                ms.append(float(o["wall_ms"]))
            for k in [
                "fetch_ms",
                "transfer_ms",
                "kernel_total_ms",
                "centroid_ms",
                "decode_ms",
                "compute_ms",
                "dist_ms",
                "heap_ms",
            ]:
                if k in o:
                    try:
                        v = float(o[k])
                    except Exception:
                        continue
                    series.setdefault(k, []).append(v)
    if meta is None or not ms:
        return None

    out: Dict[str, Any] = {
        "case": os.path.basename(path).removesuffix(".jsonl"),
        "engine": meta.get("engine", ""),
        "nq": meta.get("nq", ""),
        "k": meta.get("k", ""),
        "nprobe": meta.get("nprobe", ""),
        "posting_codec": meta.get("posting_codec", ""),
        "cache_enabled": meta.get("cache_enabled", ""),
        "index_bytes": meta.get("index_bytes", ""),
        "n": len(ms),
        "p50_ms": percentile(ms, 0.50),
        "p95_ms": percentile(ms, 0.95),
        "p99_ms": percentile(ms, 0.99),
    }
    out["profile_stages"] = bool(meta.get("profile_stages", False))
    if last:
        for k in ["chunks_fetched", "cache_hits", "compressed_bytes_in", "raw_bytes_decoded"]:
            if k in last:
                out[k] = last[k]

    for k, vals in series.items():
        if not vals:
            continue
        out[f"p50_{k}"] = percentile(vals, 0.50)
        out[f"p95_{k}"] = percentile(vals, 0.95)
        out[f"p99_{k}"] = percentile(vals, 0.99)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="Directory containing *.jsonl output files")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl")))
    rows = [r for r in (summarize_one(p) for p in paths) if r is not None]
    rows.sort(key=lambda r: str(r["case"]))

    if args.format == "json":
        print(json.dumps({"results_dir": args.results_dir, "rows": rows}, indent=2))
        return

    for r in rows:
        extra = ""
        if "chunks_fetched" in r:
            extra = (
                f" chunks_fetched={r.get('chunks_fetched')} cache_hits={r.get('cache_hits')} "
                f"fetch_ms={float(r.get('p50_fetch_ms', 0.0)):.3f}"
            )
            if "p50_transfer_ms" in r:
                extra += f" transfer_ms={float(r.get('p50_transfer_ms', 0.0)):.3f}"
        if r.get("profile_stages") and "p50_dist_ms" in r:
            extra += (
                f" stages_p50(centroid/decode/dist/heap)={float(r.get('p50_centroid_ms', 0.0)):.3f}/"
                f"{float(r.get('p50_decode_ms', 0.0)):.3f}/"
                f"{float(r.get('p50_dist_ms', 0.0)):.3f}/"
                f"{float(r.get('p50_heap_ms', 0.0)):.3f}"
            )
        print(
            f"{r['case']}: index={r['index_bytes']} "
            f"p50={r['p50_ms']:.3f} p95={r['p95_ms']:.3f} p99={r['p99_ms']:.3f}{extra}"
        )


if __name__ == "__main__":
    main()
