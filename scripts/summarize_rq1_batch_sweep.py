#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv_path",
        nargs="?",
        default="results/rq1_batch_sweep/all_results.csv",
        help="CSV produced by exp_scripts/rq1_batch_sweep.sh",
    )
    ap.add_argument("--format", choices=["text", "csv"], default="text")
    args = ap.parse_args()

    with open(args.csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by (codec, nq) and keep per-mode row.
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        codec = r.get("codec", "")
        nq = int(float(r.get("nq", "0") or 0))
        mode = r.get("mode", "")
        grouped[(codec, nq)][mode] = r

    modes = ["native", "wasm_warm", "wasm_warm_hostdist", "wasm_cold"]

    out_rows = []
    for (codec, nq), by_mode in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        base = {"codec": codec, "nq": nq}
        native_ms = None
        if "native" in by_mode:
            native_ms = float(by_mode["native"].get("avg_latency_ms", "nan"))

        for mode in modes:
            if mode not in by_mode:
                continue
            r = by_mode[mode]
            avg_ms = float(r.get("avg_latency_ms", "nan"))
            host_copy_ms = float(r.get("host_copy_time_ms", "nan"))
            compute_ms = float(r.get("compute_time_ms", "nan"))
            chunks_fetched = int(float(r.get("chunks_fetched", "0") or 0))
            decoded_cache_hits = int(float(r.get("decoded_cache_hits", "0") or 0))
            ratio = (avg_ms / native_ms) if native_ms and native_ms > 0 else None
            out_rows.append(
                {
                    **base,
                    "mode": mode,
                    "avg_ms": f"{avg_ms:.3f}",
                    "avg_ms_per_q": f"{(avg_ms / nq):.4f}" if nq else "",
                    "ratio_to_native": f"{ratio:.3f}" if ratio is not None else "",
                    "host_copy_ms": f"{host_copy_ms:.3f}",
                    "compute_ms": f"{compute_ms:.3f}",
                    "chunks_fetched": str(chunks_fetched),
                    "decoded_cache_hits": str(decoded_cache_hits),
                }
            )

    if args.format == "csv":
        w = csv.DictWriter(
            f=open("/dev/stdout", "w", newline="", encoding="utf-8"),
            fieldnames=[
                "codec",
                "nq",
                "mode",
                "avg_ms",
                "avg_ms_per_q",
                "ratio_to_native",
                "host_copy_ms",
                "compute_ms",
                "chunks_fetched",
                "decoded_cache_hits",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)
        return

    # Text/markdown-ish output (easy copy/paste).
    print(
        "codec\tnq\tmode\tavg_ms\tavg_ms/q\tratio_to_native\thost_copy_ms\tcompute_ms\tchunks_fetched\tdecoded_cache_hits"
    )
    for r in out_rows:
        print(
            f"{r['codec']}\t{r['nq']}\t{r['mode']}\t{r['avg_ms']}\t{r['avg_ms_per_q']}\t"
            f"{r['ratio_to_native']}\t{r['host_copy_ms']}\t{r['compute_ms']}\t"
            f"{r['chunks_fetched']}\t{r['decoded_cache_hits']}"
        )


if __name__ == "__main__":
    main()

