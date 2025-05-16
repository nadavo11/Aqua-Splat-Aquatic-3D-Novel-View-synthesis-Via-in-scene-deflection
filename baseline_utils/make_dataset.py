#!/usr/bin/env python3
"""
Create a Gaussian‑Splatting‑ready dataset from a single video file.

Speed‑ups integrated:
  • JPEG frames (COLMAP‑friendly, smaller).
  • Sequential matcher with small overlap (default) instead of exhaustive.
  • Optional GPU SIFT / GPU matcher via --gpu flag (uses xvfb‑run).
  • Multi‑threaded CPU fall‑back otherwise.

Example (CPU only, fast):
  python make_dataset.py aquarium.mp4 --k 800 --fps 12

Example (GPU SIFT + matcher):
  python make_dataset.py aquarium.mp4 --k 800 --fps 12 --gpu --overlap 5
"""
import pycolmap  # lightweight wrapper

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Environment tweaks (head‑less Qt)
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # force Qt off‑screen

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def need_colmap():
    if shutil.which("colmap") is None:
        raise SystemExit(
            "❌  COLMAP executable not found. Install it first, e.g.\n"
            "    sudo apt install colmap"
        )


def run(cmd: str, desc: str | None = None, gpu: bool = False):
    """Execute *cmd* in the shell, streaming output. Uses xvfb‑run when *gpu*."""
    if desc:
        print(f"\n▶ {desc}")
    if gpu:
        cmd = f"xvfb-run -s \"-screen 0 640x480x24\" {cmd}"
    subprocess.check_call(cmd, shell=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Frame extraction (ffmpeg)
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(video: Path, out_dir: Path, k: int, fps: int):
    """Extract ≈k JPEG frames from *video* into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1‑A: count total frames robustly
    try:
        probe = subprocess.check_output(
            f"ffprobe -v error -count_frames -select_streams v:0 "
            f"-show_entries stream=nb_read_frames -of csv=p=0 {video}",
            shell=True,
        )
        total_frames = int(probe)
    except Exception:
        probe = subprocess.check_output(
            f"ffprobe -v error -count_frames -select_streams v:0 "
            f"-show_entries stream=nb_read_frames -of default=nokey=1 {video}",
            shell=True,
        )
        total_frames = int(next(l for l in probe.decode().splitlines() if l.isdigit()))

    step = max(total_frames // k, 1)
    print(
        f"\n▶ Extracting ≈{k} frames (total {total_frames}, keep 1 / {step}, cap {fps} fps)"
    )

    cmd = (
        f"ffmpeg -y -i {video} "
        f"-vf \"select='not(mod(n,{step}))',fps={fps}\" "
        f"-qscale:v 2 -vsync vfr {out_dir}/%06d.jpg "
        f"-hide_banner -loglevel error"
    )
    subprocess.check_call(cmd, shell=True)
    print("✓ frames extracted")


# ──────────────────────────────────────────────────────────────────────────────
# 2. COLMAP SfM pipeline (feature → match → map)
# ──────────────────────────────────────────────────────────────────────────────

def colmap_reconstruct(img_dir: Path, db_path: Path, sparse_dir: Path, *,
                       cam_model: str = "PINHOLE", gpu: bool = False,
                       overlap: int = 3, threads: int = os.cpu_count() or 4):
    need_colmap()
    sparse_dir.mkdir(exist_ok=True)

    # 2‑A: Feature extraction -------------------------------------------------
    cmd_feat = (
        f"colmap feature_extractor "
        f"--ImageReader.camera_model {cam_model} "
        f"--ImageReader.single_camera 1 "
        f"--SiftExtraction.num_threads {threads} "
        f"--SiftExtraction.use_gpu {int(gpu)} "
        f"--database_path {db_path} "
        f"--image_path {img_dir}"
    )

    # 2‑B: Matching (sequential for video) ------------------------------------
    cmd_match = (
        f"colmap sequential_matcher "
        f"--database_path {db_path} "
        f"--SequentialMatching.overlap {overlap} "
        f"--SiftMatching.num_threads {threads} "
        f"--SiftMatching.use_gpu {int(gpu)}"
    )

    # 2‑C: Mapper (bundle adjust) --------------------------------------------
    cmd_map = (
        f"colmap mapper "
        f"--database_path {db_path} --image_path {img_dir} "
        f"--output_path {sparse_dir} "
        f"--Mapper.num_threads {threads} "
        f"--Mapper.extract_colors 0"
    )

    run(cmd_feat, "COLMAP feature extraction", gpu)
    run(cmd_match, "COLMAP matching", gpu)
    run(cmd_map, "COLMAP mapping (SfM)", gpu)
    print("✓ sparse reconstruction complete")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Convert COLMAP -> transforms_*.json (Gaussian‑Splatting format)
# ──────────────────────────────────────────────────────────────────────────────

def convert_to_gs(sparse_model: Path, img_dir: Path, train_json: Path, test_json: Path,
                  test_ratio: float = 0.1):

    rec = pycolmap.Reconstruction(sparse_model)
    ims = list(rec.images.items())
    random.shuffle(ims)

    split = int(len(ims) * test_ratio)
    test_ims, train_ims = ims[:split], ims[split:]

    for subset, path in [(train_ims, train_json), (test_ims, test_json)]:
        frames = []
        for _, im in tqdm(subset, desc=f"Writing {path.name}"):
            cam = rec.cameras[im.camera_id]
            w, h = cam.width, cam.height
            fx = fy = cam.params[0]
            cx = cam.params[1] if len(cam.params) > 3 else w / 2
            cy = cam.params[2] if len(cam.params) > 3 else h / 2

            T_c2w = im.cam_from_world.matrix()      # (4, 4) numpy array
            frames.append({
                "file_path": f"images/{Path(im.name).name}",
                "transform_matrix": T_c2w.tolist(),
                "w": w,
                "h": h,
                "fl_x": fx,
                "fl_y": fy,
                "cx": cx,
                "cy": cy,
            })
        with open(path, "w") as f:
            json.dump({"frames": frames}, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Write split.txt (train=0 / test=1) for legacy loaders
# ──────────────────────────────────────────────────────────────────────────────

def write_split(img_dir: Path, train_json: Path, test_json: Path, split_path: Path):
    train_names = {Path(f["file_path"]).name for f in json.load(open(train_json))["frames"]}
    test_names = {Path(f["file_path"]).name for f in json.load(open(test_json))["frames"]}

    with open(split_path, "w") as f:
        for img in tqdm(sorted(img_dir.glob("*.jpg")), desc="Writing split.txt"):
            tag = 0 if img.name in train_names else 1
            f.write(f"{img.name} {tag}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry‑point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--k", type=int, default=600, help="#frames to keep after sampling")
    p.add_argument("--fps", type=int, default=15, help="cap FPS during extraction")
    p.add_argument("--dataset", default="aquarium_dataset", help="output folder name")
    p.add_argument("--overlap", type=int, default=3, help="sequential matcher overlap")
    p.add_argument("--gpu", action="store_true", help="use GPU SIFT + matcher via xvfb")
    args = p.parse_args()

    root = Path(args.dataset)
    img_dir = root / "images"
    sparse_dir = root / "sparse"
    db_path = root / "colmap.db"
    root.mkdir(exist_ok=True)

    extract_frames(args.video, img_dir, args.k, args.fps)

    colmap_reconstruct(
        img_dir=img_dir,
        db_path=db_path,
        sparse_dir=sparse_dir,
        gpu=args.gpu,
        overlap=args.overlap,
    )

    convert_to_gs(
        sparse_model=sparse_dir / "0",  # COLMAP mapper writes model 0
        img_dir=img_dir,
        train_json=root / "transforms_train.json",
        test_json=root / "transforms_test.json",
    )

    write_split(
        img_dir=img_dir,
        train_json=root / "transforms_train.json",
        test_json=root / "transforms_test.json",
        split_path=root / "split.txt",
    )

    print("\n✅  Dataset ready at", root.resolve())
