#!/usr/bin/env python
import argparse
import json
import os
import sys


def stem(fp: str) -> str:
    base = os.path.basename(fp)
    if base.lower().endswith(".png"):
        base = base[:-4]
    return base


def main():
    ap = argparse.ArgumentParser(description="Normalize transforms.json to transforms_train/test.json")
    ap.add_argument("--dataset_dir", required=True, help="Dataset directory containing transforms.json")
    ap.add_argument("--source", default="transforms.json", help="Source transforms file name")
    ap.add_argument("--train_out", default="transforms_train.json", help="Output train transforms file name")
    ap.add_argument("--test_out", default="transforms_test.json", help="Output test transforms file name")
    args = ap.parse_args()

    ds = args.dataset_dir
    src = os.path.join(ds, args.source)
    if not os.path.isfile(src):
        print(f"[warn] {src} not found; nothing to do.")
        sys.exit(0)

    with open(src, "r") as f:
        meta = json.load(f)

    for fr in meta.get("frames", []):
        if "file_path" in fr:
            fr["file_path"] = stem(fr["file_path"])

    for out_name in [args.train_out, args.test_out]:
        out_path = os.path.join(ds, out_name)
        with open(out_path, "w") as g:
            json.dump(meta, g)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
