"""Stage 2.3 — Standalone OCR CLI.

Usage:
    .venv\\Scripts\\python.exe scripts/run_ocr.py \\
        --track-dir outputs/simple_traffic_30s/track_002 \\
        [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_ocr")

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fmt_slot_table(slots) -> str:
    header = f"{'Slot':>4} {'Type':>4} {'Pipe A (char/conf/share)':>26} {'Pipe B (char/conf/share)':>26} {'Bucket':>14} {'Char':>4} {'Conf':>7} {'VLM Action':>24}"
    sep = "-" * len(header)
    rows = [header, sep]
    for s in slots:
        a_char = s.get("pipe_a_top", "?")
        a_conf = s.get("pipe_a_conf", 0.0)
        a_share = s.get("pipe_a_vote_share", 1.0)
        a_raw = s.get("pipe_a_char_conf", a_conf)
        pipe_a_str = f"{a_char} ({a_conf:.2f}/{a_share:.2f})"

        b_char = s.get("pipe_b_top", "?")
        b_conf = s.get("pipe_b_conf", 0.0)
        b_share = s.get("pipe_b_vote_share", 1.0)
        b_raw = s.get("pipe_b_char_conf", b_conf)
        pipe_b_str = f"{b_char} ({b_conf:.2f}/{b_share:.2f})"

        rows.append(
            f"{s['slot']:>4} {s.get('slot_type','?'):>4} "
            f"{pipe_a_str:>26} {pipe_b_str:>26} "
            f"{s['bucket']:>14} {s['char']:>4} "
            f"{s['combined_conf']:>7.3f} {s['vlm_action']:>24}"
        )
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description="Stage 2.3 OCR runner")
    p.add_argument("--track-dir", required=True, help="Path to track directory (contains plate_manifest.json)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    args = p.parse_args()

    track_dir = Path(args.track_dir)
    manifest_path = track_dir / "plate_manifest.json"
    if not manifest_path.exists():
        logger.error(f"plate_manifest.json not found in {track_dir}")
        sys.exit(1)

    logger.info(f"Running OCR on track: {track_dir}")
    t0 = time.time()

    from cv_pipeline.ocr_engine import run_ocr_track
    result = run_ocr_track(str(manifest_path), device=args.device)

    out_path = track_dir / "ocr_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"ocr_result.json written to {out_path}")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print(f"  TRACK {result['track_id']} — Stage 2.3 OCR Results")
    print("=" * 70)
    print(f"  Recovered plate : {result['recovered_plate']}")
    print(f"  Overall conf    : {result['overall_confidence']:.4f}")
    print(f"  State code valid: {result['state_code_valid']}")
    print(f"  Frames processed: {result['n_frames_processed']}")
    print(f"  Plate length    : {result['plate_length']}")
    ss = result["slot_summary"]
    print(f"  Slots  LOCKED={ss['locked']}  CONTRADICTORY={ss['contradictory']}  UNREADABLE={ss['unreadable']}")
    print(f"  Runtime         : {result['runtime_seconds']:.1f}s")
    print()
    print("Slot-by-Slot Table:")
    print(_fmt_slot_table(result["slots"]))
    print()
    print("Top-5 highest-weight frames:")
    for fr in result["top5_frames"]:
        print(f"  frame={fr['frame_idx']:>5}  pipe={fr['pipe']:<6}  text={fr['text']:<12}  weight={fr['weight']:.4f}")
    print()
    print(f"VLM Handoff: {result['vlm_handoff']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
