#!/usr/bin/env python3
"""CLI script for Stage 2.4 — Blind Multi-Tier VLM Triple-Check & Tie-Breaker.

Usage:
    python scripts/run_vlm.py --track-dir outputs/activa_signal/track_027
    python scripts/run_vlm.py --track-dir outputs/simple_traffic_30s/track_002 --provider groq
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cv_pipeline.vlm_verifier import verify_track_vlm


def print_banner(text: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n  {text}\n{line}")


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 2.4 Blind VLM Verification on an OCR-processed track directory."
    )
    parser.add_argument(
        "--track-dir",
        type=str,
        required=True,
        help="Path to track directory (e.g., outputs/activa_signal/track_027)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "groq", "nvidia", "ollama"],
        help="VLM provider to use (default: auto)",
    )
    args = parser.parse_args()

    track_dir = Path(args.track_dir).resolve()
    if not track_dir.is_dir():
        print(f"Error: Track directory does not exist: {track_dir}", file=sys.stderr)
        sys.exit(1)

    print_banner(f"Stage 2.4: Blind Multi-Tier VLM Verification | {track_dir.name}")
    print(f"Provider requested: {args.provider}")

    try:
        res = verify_track_vlm(track_dir=track_dir, provider=args.provider)
    except Exception as e:
        print(f"\nExecution failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. High-Level Summary
    print_banner("Verification & Arbitration Summary")
    print(f"  Track ID:          {res.get('track_id')}")
    print(f"  VLM Status:        {res.get('vlm_status')}")
    print(f"  Provider Used:     {res.get('provider_used')} ({res.get('model_used')})")
    print(f"  Runtime:           {res.get('runtime_seconds'):.2f}s")
    print(f"  Detected Layout:   {res.get('plate_layout')}")
    print(f"  Stage 2.3 OCR:     {res.get('stage_2_3_plate')}")
    print(f"  VLM Raw Text:      {res.get('vlm_raw_plate')}")
    print(f"  FINAL RECONCILED:  {res.get('reconciled_plate')} (Valid State Code: {res.get('state_code_valid')})")

    # 2. Slot-by-Slot Table
    slots = res.get("reconciled_slots", [])
    if slots:
        print("\n" + "=" * 105)
        header = (
            f"{'Slot':<4} | {'Tone':<7} | {'OCR':<4} | {'OCR Bucket':<13} | "
            f"{'VLM Raw':<7} | {'Disambig':<8} | {'Final':<5} | "
            f"{'Reconciliation Action':<24} | {'Ghost Stack'}"
        )
        print(header)
        print("-" * 105)
        for s in slots:
            idx = s.get("slot_idx", 0)
            tone = s.get("ui_tile_tone", "unknown")
            ocr_c = s.get("ocr_char", "?")
            bucket = s.get("ocr_bucket", "")
            raw_v = s.get("vlm_blind_raw_char") or s.get("vlm_char") or "?"
            dis_v = s.get("vlm_disambiguated_char") or raw_v
            final_c = s.get("final_char", "?")
            act = s.get("action", "")
            g_stack = s.get("ghost_stack", [])
            if g_stack:
                g_str = ", ".join(f"{g['char']}({g['soft_score']:.2f},{g['kind'][:4]})" for g in g_stack)
            else:
                g_str = "-"
            print(
                f"{idx:<4} | {tone:<7} | {ocr_c:<4} | {bucket:<13} | "
                f"{raw_v:<7} | {dis_v:<8} | {final_c:<5} | "
                f"{act:<24} | {g_str}"
            )
        print("=" * 105)

    # 3. Top-3 Ghost Predictions
    ghost_preds = res.get("ghost_predictions", [])
    if ghost_preds:
        print_banner("Ghost Predictions (Advisory / Re-Ranking Only)")
        for gp in ghost_preds:
            rank = gp.get("rank")
            plate = gp.get("plate")
            score = gp.get("soft_score", 0.0)
            valid = gp.get("state_code_valid")
            ghost_slots = gp.get("ghost_slot_indices", [])
            print(
                f"  Rank {rank}: {plate:<12} | Soft Score: {score:.3f} | "
                f"Valid State Code: {str(valid):<5} | Ghost Slots: {ghost_slots}"
            )

    # 4. Actions Breakdown
    summary = res.get("actions_summary", {})
    if summary:
        print("\nAction Breakdown:")
        for act, count in summary.items():
            print(f"  - {act}: {count}")

    # 4. Fallback Trace
    trace = res.get("fallback_trace", [])
    if trace:
        print("\nProvider Fallback Trace:")
        for t in trace:
            status = t.get("status")
            tier = t.get("tier")
            prov = t.get("provider")
            mod = t.get("model")
            lat = t.get("latency_sec", 0.0)
            err = t.get("error")
            err_str = f" | error: {err}" if err else ""
            print(f"  Tier {tier} [{prov} / {mod}]: {status} ({lat:.2f}s){err_str}")

    print(f"\nEvidence Grids generated at:")
    print(f"  - {track_dir / 'vlm_evidence_unenhanced_grid.png'}")
    print(f"  - {track_dir / 'vlm_evidence_enhanced_grid.png'}")
    print(f"Result written to: {track_dir / 'vlm_result.json'}")
    print(f"                 : {track_dir / 'final_plate_result.json'}\n")


if __name__ == "__main__":
    run()
