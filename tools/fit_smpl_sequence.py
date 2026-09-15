#!/usr/bin/env python3
"""Fit per-frame SMPL pose to a sequence's canonical triangulated 3D joints.

Shape (beta) is solved first so the model's own rest-pose height matches a
real-world height supplied by the caller; this project does not maintain an
evidence-backed sequence-to-subject mapping, so gender/height must be passed
in explicitly from a source outside this repository (see
docs/design/subject_anthropometry.md).

The dataset's triangulated 3D is in sequence-local arbitrary units, so a
scalar `scale` (arbitrary units -> meters) is jointly optimized per sequence
alongside per-frame pose/global_orient/transl -- it is never assumed. A
temporal smoothness term on body_pose lets joints with no direct 3D
correspondence (spine/collar/hand) follow a continuous plausible motion
instead of collapsing to a static rest pose.

This is a pseudo-label fit, not ground truth. SMPL model files are gated
(https://smpl.is.tue.mpg.de) and are not distributed by this repository;
point --smpl-model-root at your own local copy.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import smplx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMPL_MODEL_ROOT = Path(
    os.environ.get("EXERCISE3D_SMPL_MODEL_ROOT", "/path/to/smpl/converted")
).expanduser()

CANONICAL_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel", "neck",
    "pelvis_center", "shoulder_center",
]
CANON_IDX = {n: i for i, n in enumerate(CANONICAL_NAMES)}

# (canonical joint name, SMPL joint index, weight) -- the only joints with a
# direct, unambiguous 3D correspondence between this dataset's canonical
# skeleton and SMPL's kinematic tree. Spine/collar/hand joints are left to
# the temporal smoothness term rather than guessed via an unverified
# cross-rig retarget.
CORRESPONDENCE = [
    ("pelvis_center", 0, 1.5),
    ("left_hip", 1, 1.0), ("right_hip", 2, 1.0),
    ("left_knee", 4, 1.0), ("right_knee", 5, 1.0),
    ("left_ankle", 7, 1.0), ("right_ankle", 8, 1.0),
    ("neck", 12, 1.0),
    ("left_shoulder", 16, 1.0), ("right_shoulder", 17, 1.0),
    ("left_elbow", 18, 1.0), ("right_elbow", 19, 1.0),
    ("left_wrist", 20, 0.8), ("right_wrist", 21, 0.8),
    ("nose", 15, 0.3),
]
UP_AXIS = 1  # SMPL rest pose: Y is vertical (verified via joint coordinates: head.y > ankle.y)


def model_height_m(model: smplx.SMPL, betas: torch.Tensor) -> torch.Tensor:
    out = model(betas=betas, body_pose=torch.zeros(1, 69, device=betas.device),
                global_orient=torch.zeros(1, 3, device=betas.device))
    v = out.vertices[0]
    return (v.max(dim=0).values - v.min(dim=0).values)[UP_AXIS]


def fit_beta(model: smplx.SMPL, target_height_m: float, device: torch.device,
             steps: int = 800, lr: float = 0.05) -> torch.Tensor:
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    opt = torch.optim.Adam([betas], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = (model_height_m(model, betas) - target_height_m) ** 2 + 1e-3 * (betas ** 2).sum()
        loss.backward()
        opt.step()
    return betas.detach()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequence", required=True)
    p.add_argument("--gender", required=True, choices=["male", "female", "neutral"])
    p.add_argument("--height-cm", required=True, type=float)
    p.add_argument("--triangulation-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "smpl_fit_full")
    p.add_argument("--smpl-model-root", type=Path, default=DEFAULT_SMPL_MODEL_ROOT)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--beta-steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pose-reg", type=float, default=3e-3)
    p.add_argument("--smooth-weight", type=float, default=0.15,
                    help="temporal smoothness on body_pose between adjacent frames")
    return p


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    tri_path = args.triangulation_root / args.sequence / "canonical_3d.npz"
    d = np.load(tri_path, allow_pickle=True)
    kp = d["keypoints_3d"]          # (N, 26, 3), arbitrary units, NaN where invalid
    valid = d["valid_mask"]         # (N, 26) bool
    frame_index = d["frame_index"]
    timestamps = d["timestamp_pts_seconds"]
    N = kp.shape[0]

    kp_t = torch.tensor(np.nan_to_num(kp, nan=0.0), dtype=torch.float32, device=device)
    valid_t = torch.tensor(valid, dtype=torch.float32, device=device)

    model_root = str(args.smpl_model_root)
    model = smplx.create(model_root, model_type="smpl", gender=args.gender, batch_size=N).to(device)
    beta_model = smplx.create(model_root, model_type="smpl", gender=args.gender, batch_size=1).to(device)
    betas_single = fit_beta(beta_model, args.height_cm / 100.0, device, steps=args.beta_steps)
    fitted_height_m = float(model_height_m(beta_model, betas_single))
    betas = betas_single.expand(N, -1).contiguous()

    global_orient = torch.zeros(N, 3, device=device, requires_grad=True)
    body_pose = torch.zeros(N, 69, device=device, requires_grad=True)
    transl = torch.zeros(N, 3, device=device, requires_grad=True)
    log_scale = torch.zeros(1, device=device, requires_grad=True)  # scale = exp(log_scale)

    with torch.no_grad():
        transl.copy_(kp_t[:, CANON_IDX["pelvis_center"], :])

    opt = torch.optim.Adam([global_orient, body_pose, transl, log_scale], lr=args.lr)

    smpl_idx = torch.tensor([s for _, s, _ in CORRESPONDENCE], device=device)
    canon_idx = torch.tensor([CANON_IDX[n] for n, _, _ in CORRESPONDENCE], device=device)
    weights = torch.tensor([w for _, _, w in CORRESPONDENCE], device=device).view(1, -1, 1)

    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        out = model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        scale = torch.exp(log_scale)
        pred = out.joints[:, smpl_idx, :] * scale
        target = kp_t[:, canon_idx, :]
        w = weights * valid_t[:, canon_idx].unsqueeze(-1)
        joint_loss = (((pred - target) ** 2) * w).sum() / w.sum().clamp(min=1.0)
        pose_reg = args.pose_reg * (body_pose ** 2).mean()
        smooth_loss = args.smooth_weight * ((body_pose[1:] - body_pose[:-1]) ** 2).mean()
        loss = joint_loss + pose_reg + smooth_loss
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"[{args.sequence}] step {step:4d} joint_loss={joint_loss.item():.6f} "
                  f"smooth={smooth_loss.item():.6f} scale={scale.item():.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    with torch.no_grad():
        out = model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        scale = torch.exp(log_scale)
        pred = out.joints[:, smpl_idx, :] * scale
        target = kp_t[:, canon_idx, :]
        w = valid_t[:, canon_idx].unsqueeze(-1)
        per_frame_err = ((pred - target) ** 2 * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp(min=1.0)
        per_frame_err_m = torch.sqrt(per_frame_err) * float(scale)

    out_dir = args.output_root / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "smpl_fit.npz",
        frame_index=frame_index,
        timestamp_pts_seconds=timestamps,
        betas=betas_single.cpu().numpy(),
        global_orient=global_orient.detach().cpu().numpy(),
        body_pose=body_pose.detach().cpu().numpy(),
        transl=transl.detach().cpu().numpy(),
        scale_arbitrary_to_meters=float(scale.item()),
        per_frame_joint_rmse_m=per_frame_err_m.cpu().numpy(),
    )
    meta = {
        "schema_version": 1,
        "sequence": args.sequence,
        "not_ground_truth": True,
        "method": "SMPL 3D joint fit (SMPLify-3D style): shared beta anchored to an externally "
                  "supplied real-world height, per-frame pose/global_orient/transl optimized "
                  "against canonical triangulated 3D joints; sequence-local arbitrary-unit -> "
                  "meter scale jointly optimized as a single scalar; temporal smoothness "
                  "regularizer on body_pose across adjacent frames so joints without direct 3D "
                  "correspondence (spine/collar/hand) follow a continuous plausible motion "
                  "instead of collapsing to a static rest pose.",
        "temporal_smoothness_weight": args.smooth_weight,
        "gender_source": "external subject anthropometry file (not in this repository)",
        "gender": args.gender,
        "height_source": "external subject anthropometry file (not in this repository)",
        "height_cm_target": args.height_cm,
        "height_cm_fitted_model": fitted_height_m * 100.0,
        "joint_correspondence": [{"canonical": n, "smpl_joint_index": s, "weight": w} for n, s, w in CORRESPONDENCE],
        "frame_count": int(N),
        "scale_arbitrary_to_meters": float(scale.item()),
        "joint_rmse_m_median": float(np.median(per_frame_err_m.cpu().numpy())),
        "joint_rmse_m_p95": float(np.percentile(per_frame_err_m.cpu().numpy(), 95)),
        "steps": args.steps,
        "pose_regularization_weight": args.pose_reg,
        "smpl_model": "SMPL v1.1.0 (300 shape PCs source; first 10 used)",
        "caveat": "spine/collar/hand joints have no direct 3D correspondence in this joint set "
                  "and are only weakly constrained by the temporal smoothness and pose "
                  "regularizer terms.",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"saved -> {out_dir}, median joint RMSE {meta['joint_rmse_m_median']*1000:.1f} mm, "
          f"p95 {meta['joint_rmse_m_p95']*1000:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
