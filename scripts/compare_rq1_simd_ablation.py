#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(row: dict[str, str]) -> tuple[str, int, str]:
    return (row.get("codec", ""), int(float(row.get("nq", "0") or 0)), row.get("mode", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dir",
        nargs="?",
        default="results/rq1_simd_ablation",
        help="Directory produced by exp_scripts/rq1_simd_ablation.sh",
    )
    args = ap.parse_args()

    base = Path(args.dir)
    simd_csv = base / "simd" / "all_results.csv"
    nosimd_csv = base / "nosimd" / "all_results.csv"
    if not simd_csv.exists() or not nosimd_csv.exists():
        raise SystemExit(f"missing csvs: {simd_csv} {nosimd_csv}")

    simd = {key(r): r for r in load(simd_csv)}
    nosimd = {key(r): r for r in load(nosimd_csv)}

    modes = ["native", "wasm_warm", "wasm_warm_hostdist", "wasm_cold"]
    keys = sorted(set(simd.keys()) & set(nosimd.keys()), key=lambda k: (k[0], k[1], modes.index(k[2]) if k[2] in modes else 99))

    print(
        "codec\tnq\tmode"
        "\tavg_ms(simd)\tavg_ms(nosimd)\tratio(nosimd/simd)"
        "\tcompute_ms(simd)\tcompute_ms(nosimd)\tratio_compute(nosimd/simd)"
        "\thost_copy_ms(simd)\thost_copy_ms(nosimd)"
    )
    for k in keys:
        r1 = simd[k]
        r2 = nosimd[k]
        avg1 = float(r1.get("avg_latency_ms", "nan"))
        avg2 = float(r2.get("avg_latency_ms", "nan"))
        ratio = (avg2 / avg1) if avg1 and avg1 > 0 else float("nan")
        c1 = float(r1.get("compute_time_ms", "nan"))
        c2 = float(r2.get("compute_time_ms", "nan"))
        ratio_c = (c2 / c1) if c1 and c1 > 0 else float("nan")
        h1 = float(r1.get("host_copy_time_ms", "nan"))
        h2 = float(r2.get("host_copy_time_ms", "nan"))
        print(
            f"{k[0]}\t{k[1]}\t{k[2]}"
            f"\t{avg1:.3f}\t{avg2:.3f}\t{ratio:.3f}"
            f"\t{c1:.3f}\t{c2:.3f}\t{ratio_c:.3f}"
            f"\t{h1:.3f}\t{h2:.3f}"
        )


if __name__ == "__main__":
    main()
