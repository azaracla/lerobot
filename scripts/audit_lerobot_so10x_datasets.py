"""
audit_lerobot_datasets.py
=========================
Explore, score and flag LeRobot datasets for SO-100/SO-101 training.

Workflow:
  1. Fetch all public LeRobot datasets via HF API
  2. Filter by robot_type (so100/so101 variants)
  3. Load meta/info.json for each → inspect features, cameras, resolution, fps
  4. Score quality (0-100) based on configurable criteria
  5. Flag "compatible" datasets matching your target format
  6. Export: flagged_datasets.json  +  normalization_map.json

Requirements:
  pip install huggingface_hub requests tqdm

Usage:
  python audit_lerobot_datasets.py
  python audit_lerobot_datasets.py --min-episodes 50 --target-res 640x480 --target-fps 30
"""

import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import requests
from tqdm import tqdm

try:
    from huggingface_hub import HfApi, hf_hub_download, list_datasets
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
except ImportError:
    raise SystemExit("pip install huggingface_hub tqdm")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

# Robot types to include (covers naming variants in the wild)
SO10X_TYPES = {
    "so100",
    "so101",
    "so100_follower",
    "so101_follower",
    "so100-red",
    "so100-blue",
    "so101_follower_bimanual",
    "so100_follower_bimanual",
    "bi_so100_follower",
    "bi_so101_follower",
    "dual_so101_follower",
    "dual_so100_follower",
}

# Camera key aliases → canonical name
# Keys: possible names found in the wild | Value: your canonical name
CAMERA_ALIASES: dict[str, str] = {
    "observation.image": "cam_main",
    "observation.images.main": "cam_main",
    "observation.images.front": "cam_main",
    "observation.images.cam_main": "cam_main",
    "observation.image2": "cam_wrist",
    "observation.images.wrist": "cam_wrist",
    "observation.images.cam_wrist": "cam_wrist",
    "observation.images.top": "cam_top",
    "observation.images.cam_top": "cam_top",
    "observation.images.overhead": "cam_top",
    "observation.images.side": "cam_side",
}

# ─── Data model ──────────────────────────────────────────────────────────────


@dataclass
class DatasetAudit:
    repo_id: str
    robot_type: str = "unknown"
    lerobot_version: str = "unknown"
    total_episodes: int = 0
    total_frames: int = 0
    fps: int = 0
    avg_episode_len_s: float = 0.0
    camera_keys: list[str] = field(default_factory=list)
    canonical_cameras: list[str] = field(default_factory=list)  # after alias mapping
    unknown_camera_keys: list[str] = field(default_factory=list)  # not in alias map
    resolutions: dict[str, str] = field(default_factory=dict)  # key → "WxH"
    action_dim: int = 0
    state_dim: int = 0
    tasks: list[str] = field(default_factory=list)
    score: int = 0  # 0-100
    flags: list[str] = field(default_factory=list)  # quality issues
    compatible: bool = False  # matches target spec exactly
    error: Optional[str] = None


# ─── Core logic ──────────────────────────────────────────────────────────────


def fetch_so10x_datasets(max_results: int = 5000) -> list[str]:
    """Return repo_ids of all public LeRobot datasets tagged with lerobot."""
    api = HfApi()
    log.info("Fetching dataset list from HuggingFace (this may take a minute)…")
    datasets = api.list_datasets(filter="lerobot", limit=max_results)
    return [d.id for d in datasets]


def load_info_json(repo_id: str) -> Optional[dict]:
    """Download meta/info.json from a LeRobot dataset repo."""
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/info.json",
            repo_type="dataset",
        )
        with open(path) as f:
            return json.load(f)
    except (EntryNotFoundError, RepositoryNotFoundError, Exception):
        return None


def parse_resolution(shape: list) -> str:
    """Convert HxWxC shape list to 'WxH' string."""
    if len(shape) >= 2:
        # shape is usually [H, W, C] or [C, H, W]
        if shape[0] in (1, 3, 4) and shape[1] > 10:  # CHW
            return f"{shape[2]}x{shape[1]}"
        elif shape[2] in (1, 3, 4):  # HWC
            return f"{shape[1]}x{shape[0]}"
    return "unknown"


def audit_dataset(repo_id: str) -> DatasetAudit:
    audit = DatasetAudit(repo_id=repo_id)
    info = load_info_json(repo_id)

    if info is None:
        audit.error = "meta/info.json not found"
        return audit

    audit.robot_type = info.get("robot_type", "unknown")
    audit.lerobot_version = info.get("codebase_version", info.get("version", "unknown"))
    audit.total_episodes = info.get("total_episodes", 0)
    audit.total_frames = info.get("total_frames", 0)
    audit.fps = info.get("fps", 0)

    if audit.fps > 0 and audit.total_episodes > 0:
        audit.avg_episode_len_s = round(audit.total_frames / (audit.fps * audit.total_episodes), 1)

    # Task descriptions
    tasks_info = info.get("tasks", {})
    if isinstance(tasks_info, dict):
        audit.tasks = [t.get("task", "") for t in tasks_info.values()][:5]
    elif isinstance(tasks_info, list):
        audit.tasks = [t if isinstance(t, str) else t.get("task", "") for t in tasks_info][:5]

    # Features → cameras, dims
    features = info.get("features", {})
    for key, feat in features.items():
        dtype = feat.get("dtype", "")
        shape = feat.get("shape", [])

        if dtype == "video" or (isinstance(shape, list) and len(shape) == 3):
            audit.camera_keys.append(key)
            canonical = CAMERA_ALIASES.get(key)
            if canonical:
                audit.canonical_cameras.append(canonical)
            else:
                audit.unknown_camera_keys.append(key)
            if shape:
                audit.resolutions[key] = parse_resolution(shape)

        if key == "action" and isinstance(shape, list):
            audit.action_dim = shape[0] if shape else 0
        if "observation.state" in key and isinstance(shape, list):
            audit.state_dim = shape[0] if shape else 0

    return audit


def score_and_flag(
    audit: DatasetAudit,
    target_res: str,
    target_fps: int,
    min_episodes: int,
    min_cameras: int,
    required_canonical: list[str],
) -> DatasetAudit:
    score = 0
    flags = []

    # Episodes
    if audit.total_episodes >= min_episodes:
        score += 25
    elif audit.total_episodes >= min_episodes // 2:
        score += 12
        flags.append(f"low_episodes:{audit.total_episodes}")
    else:
        flags.append(f"very_low_episodes:{audit.total_episodes}")

    # FPS
    if audit.fps == target_fps:
        score += 15
    elif audit.fps > 0:
        flags.append(f"fps_mismatch:{audit.fps}!={target_fps}")

    # Camera count
    if len(audit.camera_keys) >= min_cameras:
        score += 15
    else:
        flags.append(f"not_enough_cameras:{len(audit.camera_keys)}<{min_cameras}")

    # Resolutions
    target_W, target_H = target_res.split("x")
    bad_res = []
    for k, r in audit.resolutions.items():
        if r != target_res:
            bad_res.append(f"{k}:{r}")
    if not bad_res:
        score += 20
    else:
        flags.append(f"resolution_mismatch:{','.join(bad_res)}")
        score += 5

    # Camera name mapping
    unresolved = len(audit.unknown_camera_keys)
    if unresolved == 0:
        score += 15
    else:
        flags.append(f"unknown_camera_keys:{','.join(audit.unknown_camera_keys)}")
        score += max(0, 15 - unresolved * 5)

    # Required canonical cameras present
    missing = [c for c in required_canonical if c not in audit.canonical_cameras]
    if not missing:
        score += 10
    else:
        flags.append(f"missing_canonical:{','.join(missing)}")

    # Episode length (prefer 10-60s)
    if 10 <= audit.avg_episode_len_s <= 60:
        score += 0  # bonus already baked in
    elif audit.avg_episode_len_s > 0:
        flags.append(f"unusual_episode_len:{audit.avg_episode_len_s}s")

    # Version penalty
    if audit.lerobot_version not in ("v2.0", "v2.1", "2.0", "2.1"):
        flags.append(f"old_format:{audit.lerobot_version}")
        score = max(0, score - 10)

    audit.score = min(score, 100)
    audit.flags = flags
    audit.compatible = len(flags) == 0 and audit.score >= 70
    return audit


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Audit LeRobot SO-10x datasets")
    parser.add_argument("--min-episodes", type=int, default=30, help="Min episodes threshold")
    parser.add_argument("--target-res", type=str, default="640x480", help="Target camera resolution WxH")
    parser.add_argument("--target-fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--min-cameras", type=int, default=2, help="Min number of cameras")
    parser.add_argument(
        "--required-cams", type=str, default="cam_main", help="Comma-separated required canonical cameras"
    )
    parser.add_argument("--max-datasets", type=int, default=5000, help="Max datasets to fetch from HF")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for JSON files")
    parser.add_argument(
        "--filter-robot", action="store_true", help="Pre-filter by robot_type metadata (faster)"
    )
    args = parser.parse_args()

    required_canonical = [c.strip() for c in args.required_cams.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch dataset list
    all_repos = fetch_so10x_datasets(max_results=args.max_datasets)
    log.info(f"Found {len(all_repos)} LeRobot datasets total")

    # 2. Audit each
    results: list[DatasetAudit] = []
    skipped = 0

    for repo_id in tqdm(all_repos, desc="Auditing"):
        audit = audit_dataset(repo_id)

        # Filter by robot type
        rt = audit.robot_type.lower().replace("-", "_")
        if rt not in SO10X_TYPES and not any(s in rt for s in ("so100", "so101")):
            skipped += 1
            continue

        audit = score_and_flag(
            audit,
            target_res=args.target_res,
            target_fps=args.target_fps,
            min_episodes=args.min_episodes,
            min_cameras=args.min_cameras,
            required_canonical=required_canonical,
        )
        results.append(audit)

    log.info(f"Audited {len(results)} SO-10x datasets ({skipped} skipped / wrong robot_type)")

    # 3. Sort by score desc
    results.sort(key=lambda a: a.score, reverse=True)

    # 4. Export full audit
    full_path = output_dir / "lerobot_audit_full.json"
    with open(full_path, "w") as f:
        json.dump([asdict(a) for a in results], f, indent=2)
    log.info(f"Full audit → {full_path}")

    # 5. Export flagged (compatible) list
    compatible = [a for a in results if a.compatible]
    flagged_path = output_dir / "flagged_datasets.json"
    with open(flagged_path, "w") as f:
        json.dump(
            {
                "target_spec": {
                    "resolution": args.target_res,
                    "fps": args.target_fps,
                    "min_episodes": args.min_episodes,
                    "min_cameras": args.min_cameras,
                    "required_canonical_cameras": required_canonical,
                },
                "compatible_count": len(compatible),
                "datasets": [
                    {
                        "repo_id": a.repo_id,
                        "robot_type": a.robot_type,
                        "total_episodes": a.total_episodes,
                        "fps": a.fps,
                        "avg_episode_len_s": a.avg_episode_len_s,
                        "cameras": a.camera_keys,
                        "canonical_cameras": a.canonical_cameras,
                        "resolutions": a.resolutions,
                        "score": a.score,
                        "tasks": a.tasks,
                    }
                    for a in compatible
                ],
            },
            f,
            indent=2,
        )
    log.info(f"Compatible datasets ({len(compatible)}) → {flagged_path}")

    # 6. Build normalization map: for each camera key found → canonical name
    norm_map: dict[str, str] = {}
    for a in results:
        for k in a.camera_keys:
            if k not in norm_map:
                norm_map[k] = CAMERA_ALIASES.get(k, "UNMAPPED_" + k.replace(".", "_"))

    norm_path = output_dir / "normalization_map.json"
    with open(norm_path, "w") as f:
        json.dump(norm_map, f, indent=2)
    log.info(f"Camera normalization map → {norm_path}")

    # 7. Print summary
    print("\n" + "=" * 60)
    print(f"  SUMMARY — SO-100/SO-101 dataset audit")
    print("=" * 60)
    print(f"  Audited:     {len(results)} datasets")
    print(f"  Compatible:  {len(compatible)} datasets (score≥70, no flags)")
    print(f"  Score ≥ 60:  {sum(1 for a in results if a.score >= 60)}")
    print(f"  Score < 30:  {sum(1 for a in results if a.score < 30)}")

    print("\n  TOP 10 datasets by score:")
    for a in results[:10]:
        print(
            f"    [{a.score:3d}] {a.repo_id}  ({a.total_episodes} eps, {a.fps}fps, {list(a.resolutions.values())})"
        )

    print("\n  Most common camera key issues:")
    from collections import Counter

    all_flags = [f for a in results for f in a.flags]
    for flag, count in Counter(all_flags).most_common(10):
        print(f"    {count:4d}×  {flag}")

    print(f"\n  Outputs in: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
