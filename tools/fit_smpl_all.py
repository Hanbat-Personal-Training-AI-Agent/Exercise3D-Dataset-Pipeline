#!/usr/bin/env python3
"""Run fit_smpl_sequence.py for every sequence in a subject-anthropometry file.

The subject-anthropometry file is private (gender/height per sequence) and is
never distributed with this repository -- see
docs/design/subject_anthropometry.md and configs/subject_anthropometry.example.json.
Splits sequences across --num-gpus by index parity via CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject-config", type=Path, required=True,
                    help="JSON file: {\"sequences\": {seq: {gender, height_cm}, ...}}")
    p.add_argument("--triangulation-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "smpl_fit_full")
    p.add_argument("--smpl-model-root", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--steps", type=int, default=1200)
    return p


def main() -> int:
    args = build_parser().parse_args()
    subjects = json.loads(args.subject_config.read_text())["sequences"]
    sequences = sorted(subjects.keys())
    my_jobs = [s for i, s in enumerate(sequences) if i % args.num_gpus == args.gpu_index]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    failures = []
    for seq in my_jobs:
        info = subjects[seq]
        print(f"=== {seq} ({info['gender']}, {info['height_cm']}cm) on GPU {args.gpu_index} ===", flush=True)
        cmd = [
            args.python, str(PROJECT_ROOT / "tools" / "fit_smpl_sequence.py"),
            "--sequence", seq,
            "--gender", info["gender"],
            "--height-cm", str(info["height_cm"]),
            "--triangulation-root", str(args.triangulation_root),
            "--output-root", str(args.output_root),
            "--smpl-model-root", str(args.smpl_model_root),
            "--steps", str(args.steps),
        ]
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"!!! FAILED: {seq} exit={r.returncode}", flush=True)
            failures.append(seq)

    print(f"GPU {args.gpu_index} done ({len(my_jobs)} sequences, {len(failures)} failed)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
