#!/usr/bin/env python3
import argparse
import json
import os
import struct
from typing import Any, Dict, List, Tuple


MAGIC = b"F3VIDX1\x00"
FOOTER_MAGIC = b"F3VIDXF\x00"


def read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"wanted {n} bytes, got {len(b)}")
    return b


def parse_footer(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        magic = read_exact(f, 8)
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        version = struct.unpack("<I", read_exact(f, 4))[0]
        footer_off = struct.unpack("<Q", read_exact(f, 8))[0]
        if footer_off == 0:
            raise ValueError("footer_offset=0")
        f.seek(footer_off)
        footer_magic = read_exact(f, 8)
        if footer_magic != FOOTER_MAGIC:
            raise ValueError(f"bad footer magic: {footer_magic!r}")
        footer_ver = struct.unpack("<I", read_exact(f, 4))[0]
        json_len = struct.unpack("<I", read_exact(f, 4))[0]
        footer = json.loads(read_exact(f, json_len))
        footer["_header_version"] = version
        footer["_footer_version"] = footer_ver
        return footer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_path", help="Path to *.ivf_flat.artifact")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    footer = parse_footer(args.artifact_path)
    chunks: List[Dict[str, Any]] = footer.get("chunks", [])

    totals: Dict[Tuple[str, str], Dict[str, int]] = {}
    for c in chunks:
        ct = str(c.get("chunk_type"))
        codec = str(c.get("codec"))
        key = (ct, codec)
        totals.setdefault(key, {"len": 0, "raw_len": 0, "count": 0})
        totals[key]["len"] += int(c.get("len", 0))
        totals[key]["raw_len"] += int(c.get("raw_len", 0))
        totals[key]["count"] += 1

    out = {
        "artifact_path": os.path.abspath(args.artifact_path),
        "index_name": footer.get("index_name"),
        "kind": footer.get("kind"),
        "dim": footer.get("dim"),
        "nlist": footer.get("nlist"),
        "base_path": footer.get("base", {}).get("path"),
        "totals": [
            {
                "chunk_type": k[0],
                "codec": k[1],
                **v,
                "ratio_len_over_raw": (v["len"] / v["raw_len"]) if v["raw_len"] else None,
            }
            for k, v in sorted(totals.items())
        ],
    }

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"artifact: {out['artifact_path']}")
    print(f"dim={out['dim']} nlist={out['nlist']} kind={out['kind']} name={out['index_name']}")
    for row in out["totals"]:
        ratio = row["ratio_len_over_raw"]
        ratio_s = "n/a" if ratio is None else f"{ratio:.4f}"
        print(
            f"{row['chunk_type']}/{row['codec']}: chunks={row['count']} "
            f"len={row['len']} raw_len={row['raw_len']} ratio={ratio_s}"
        )


if __name__ == "__main__":
    main()

