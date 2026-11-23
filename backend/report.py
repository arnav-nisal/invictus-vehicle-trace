"""Forensic Case Report Generator (backend/report.py).

Generates a clean, formal, self-contained HTML forensic case report (text/html)
strictly compliant with AIML-01 requirements:
- Case timestamp and tracking ID
- Exact keyframes and cropped plate images (embedded as base64 Data URIs with safe fallback)
- Multi-frame OCR recovery table with per-character confidence scores and ? unknown slot highlighting
- Detected vehicle colour and body type with confidence scores
- Ranked Top-5 shortlist with character-by-character diffs or high-visibility 'No Confident Match' banner
- Plain-English reasoning and forensic audit notes
- Print-optimized @media print CSS and built-in "Print / Save as PDF" toolbar
"""

from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from backend.schemas import (
    AnalysisResponse,
    FrameEvidence,
    MatchCandidate,
    VehicleEvidence,
)

BASE_DIR = Path(__file__).resolve().parent.parent

# CCTV OCR visual confusion pairs for diff labeling
CONFUSION_PAIRS = {
    ("8", "B"), ("B", "8"),
    ("0", "O"), ("O", "0"),
    ("0", "D"), ("D", "0"),
    ("1", "I"), ("I", "1"),
    ("5", "S"), ("S", "5"),
    ("2", "Z"), ("Z", "2"),
    ("6", "G"), ("G", "6"),
}


def _image_to_base64_uri(image_path: Optional[str], fallback_label: str = "Image") -> str:
    """Read an image from disk and return a self-contained base64 data URI or SVG fallback."""
    if not image_path:
        return _svg_placeholder(fallback_label)

    # Check relative to BASE_DIR or as absolute path
    path = Path(image_path)
    if not path.is_absolute():
        path = BASE_DIR / path

    if path.exists() and path.is_file():
        try:
            with open(path, "rb") as f:
                data = f.read()
            # Determine mime type
            ext = path.suffix.lower()
            mime = "image/jpeg"
            if ext == ".png":
                mime = "image/png"
            elif ext == ".webp":
                mime = "image/webp"
            b64_str = base64.b64encode(data).decode("utf-8")
            return f"data:{mime};base64,{b64_str}"
        except Exception:
            pass

    return _svg_placeholder(fallback_label)


def _svg_placeholder(label: str) -> str:
    """Generate an inline SVG fallback placeholder for missing/mock image paths."""
    safe_label = html.escape(label)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="300" height="180" viewBox="0 0 300 180" '
        f'style="background-color: #0f172a; border-radius: 6px; display: block; border: 1px solid #334155;">'
        f'<rect width="100%" height="100%" fill="#0f172a"/>'
        f'<circle cx="150" cy="70" r="28" fill="#1e293b" stroke="#475569" stroke-width="2"/>'
        f'<path d="M140 70 L160 70 M150 60 L150 80" stroke="#94a3b8" stroke-width="2" stroke-linecap="round"/>'
        f'<text x="150" y="125" font-family="monospace" font-size="12" fill="#94a3b8" text-anchor="middle" font-weight="bold">{safe_label}</text>'
        f'<text x="150" y="145" font-family="monospace" font-size="10" fill="#64748b" text-anchor="middle">FORENSIC CROP</text>'
        f'</svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def _compute_char_diffs(observed: str, candidate_plate: str) -> List[Tuple[str, str, str]]:
    """Compute character-by-character alignment and classification.
    
    Returns list of tuples: (candidate_char, observed_char, diff_type)
    where diff_type in ('exact', 'wildcard', 'confusion', 'mismatch', 'extra')
    """
    results: List[Tuple[str, str, str]] = []
    max_len = max(len(observed), len(candidate_plate))

    for i in range(max_len):
        obs_char = observed[i] if i < len(observed) else ""
        cand_char = candidate_plate[i] if i < len(candidate_plate) else ""

        if not obs_char:
            results.append((cand_char, "-", "extra"))
        elif not cand_char:
            results.append(("-", obs_char, "mismatch"))
        elif obs_char == "?":
            results.append((cand_char, "?", "wildcard"))
        elif obs_char == cand_char:
            results.append((cand_char, obs_char, "exact"))
        elif (obs_char, cand_char) in CONFUSION_PAIRS:
            results.append((cand_char, obs_char, "confusion"))
        else:
            results.append((cand_char, obs_char, "mismatch"))

    return results


def generate_forensic_html_report(
    analysis: AnalysisResponse,
    case_ref: Optional[str] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """Generate a comprehensive, formal, self-contained HTML forensic case report."""
    now = generated_at or datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    case_id = case_ref or f"INV-AIML01-{now.strftime('%Y%m%d')}-TRK{analysis.track_id:03d}"

    evidence = analysis.evidence
    consensus_plate = evidence.consensus_plate
    char_confs = evidence.char_confidences or [1.0] * len(consensus_plate)

    # Pad confidences if length mismatch
    while len(char_confs) < len(consensus_plate):
        char_confs.append(0.5)

    # Build consensus plate character tiles HTML
    plate_tiles_html = []
    for idx, (ch, conf) in enumerate(zip(consensus_plate, char_confs)):
        conf_pct = int(round(conf * 100))
        if ch == "?":
            tile_cls = "plate-char-tile tile-wildcard"
            badge = '<span class="tile-conf-badge conf-low">? UNKNOWN</span>'
        else:
            conf_cls = "conf-high" if conf >= 0.85 else ("conf-med" if conf >= 0.60 else "conf-low")
            tile_cls = f"plate-char-tile tile-known"
            badge = f'<span class="tile-conf-badge {conf_cls}">{conf_pct}%</span>'

        plate_tiles_html.append(
            f'<div class="{tile_cls}">'
            f'  <div class="char-pos">#{idx + 1}</div>'
            f'  <div class="char-val">{html.escape(ch)}</div>'
            f'  {badge}'
            f'</div>'
        )
    plate_tiles_rendered = "".join(plate_tiles_html)

    # Build Per-Frame Evidence Rows
    frame_rows_html = []
    for frame in evidence.frames_used:
        veh_uri = _image_to_base64_uri(frame.image_path, f"Vehicle Crop (F#{frame.frame_idx})")
        plate_uri = _image_to_base64_uri(frame.plate_crop_path, f"Plate Crop (F#{frame.frame_idx})")
        ocr_disp = html.escape(frame.ocr_text or "N/A")
        ts_disp = f"{frame.timestamp_sec:.2f}s" if frame.timestamp_sec is not None else "N/A"
        f_conf_pct = f"{frame.confidence * 100:.1f}%"

        frame_rows_html.append(
            f'<tr>'
            f'  <td class="font-mono"><strong>Frame #{frame.frame_idx}</strong><br><span class="text-muted">t = {ts_disp}</span></td>'
            f'  <td class="crop-cell"><img src="{veh_uri}" alt="Vehicle crop" class="report-crop-img veh-crop" /></td>'
            f'  <td class="crop-cell"><img src="{plate_uri}" alt="Plate crop" class="report-crop-img plate-crop" /></td>'
            f'  <td class="font-mono ocr-reading-cell"><code>{ocr_disp}</code></td>'
            f'  <td class="font-mono text-center"><strong>{f_conf_pct}</strong></td>'
            f'</tr>'
        )
    frame_rows_rendered = "".join(frame_rows_html) if frame_rows_html else (
        '<tr><td colspan="5" class="text-center text-muted">No frame crops recorded for this vehicle.</td></tr>'
    )

    # Build Shortlist or No-Confident-Match Section
    if analysis.no_confident_match:
        shortlist_header = (
            '<div class="alert-box alert-warning">'
            '  <div class="alert-title">⚠ NO CONFIDENT REGISTRY MATCH IDENTIFIED</div>'
            f'  <div class="alert-desc">{html.escape(analysis.explanation or "The observed evidence does not match any registered vehicle above the calibrated threshold (65.0%).")}</div>'
            '</div>'
        )
    else:
        shortlist_header = (
            '<div class="alert-box alert-success">'
            '  <div class="alert-title">✓ PRIMARY TARGET MATCH IDENTIFIED</div>'
            f'  <div class="alert-desc">{html.escape(analysis.explanation or "High-confidence multi-signal candidate confirmed.")}</div>'
            '</div>'
        )

    # Render Shortlist Candidate Cards
    candidates_html = []
    for cand in analysis.top_matches:
        rank_badge_cls = "rank-1" if cand.rank == 1 else "rank-other"
        composite_pct = f"{cand.composite_score * 100:.2f}%"
        plate_score_pct = f"{cand.score_breakdown.plate_score * 100:.1f}%"
        colour_score_pct = f"{cand.score_breakdown.colour_score * 100:.1f}%"
        body_score_pct = f"{cand.score_breakdown.body_type_score * 100:.1f}%"

        # Compute char diff badges
        diffs = _compute_char_diffs(consensus_plate, cand.plate_number)
        diff_chips = []
        for cand_ch, obs_ch, diff_type in diffs:
            if diff_type == "exact":
                diff_chips.append(f'<span class="diff-chip diff-exact" title="Exact match: {cand_ch}">{cand_ch}</span>')
            elif diff_type == "wildcard":
                diff_chips.append(f'<span class="diff-chip diff-wildcard" title="Wildcard slot (? -> {cand_ch})">{cand_ch}</span>')
            elif diff_type == "confusion":
                diff_chips.append(f'<span class="diff-chip diff-confusion" title="OCR confusion: observed {obs_ch} matches {cand_ch}">{cand_ch}</span>')
            else:
                diff_chips.append(f'<span class="diff-chip diff-mismatch" title="Mismatch: observed {obs_ch} vs {cand_ch}">{cand_ch}</span>')
        diff_chips_rendered = "".join(diff_chips)

        # Match reasons list
        reasons_list = "".join(f"<li>{html.escape(r)}</li>" for r in cand.match_reasons)

        candidates_html.append(
            f'<div class="candidate-card {"primary-candidate" if cand.rank == 1 else ""}">'
            f'  <div class="candidate-header">'
            f'    <div class="candidate-rank-badge {rank_badge_cls}">RANK #{cand.rank}</div>'
            f'    <div class="candidate-plate-group">'
            f'      <div class="diff-chips-row">{diff_chips_rendered}</div>'
            f'      <span class="candidate-plate-text">{cand.plate_number}</span>'
            f'    </div>'
            f'    <div class="candidate-score-block">'
            f'      <span class="score-label">Composite Match</span>'
            f'      <span class="score-value font-mono">{composite_pct}</span>'
            f'    </div>'
            f'  </div>'
            f'  <div class="candidate-body">'
            f'    <div class="candidate-meta-grid">'
            f'      <div class="meta-item"><span class="meta-k">Vehicle Make & Model:</span> <span class="meta-v"><strong>{html.escape(cand.make)} {html.escape(cand.model)}</strong></span></div>'
            f'      <div class="meta-item"><span class="meta-k">Body & Colour:</span> <span class="meta-v">{html.escape(cand.colour.capitalize())} {html.escape(cand.body_type)}</span></div>'
            f'      <div class="meta-item"><span class="meta-k">Registered Jurisdiction:</span> <span class="meta-v">{html.escape(cand.rto_city)}</span></div>'
            f'      <div class="meta-item"><span class="meta-k">Synthetic Owner ID:</span> <span class="meta-v font-mono">{html.escape(cand.synthetic_owner_id)}</span></div>'
            f'    </div>'
            f'    <div class="score-bars-container">'
            f'      <div class="score-bar-row">'
            f'        <span class="bar-title">Plate Similarity:</span>'
            f'        <div class="bar-track"><div class="bar-fill" style="width: {plate_score_pct};"></div></div>'
            f'        <span class="bar-pct font-mono">{plate_score_pct}</span>'
            f'      </div>'
            f'      <div class="score-bar-row">'
            f'        <span class="bar-title">Colour Match:</span>'
            f'        <div class="bar-track"><div class="bar-fill" style="width: {colour_score_pct};"></div></div>'
            f'        <span class="bar-pct font-mono">{colour_score_pct}</span>'
            f'      </div>'
            f'      <div class="score-bar-row">'
            f'        <span class="bar-title">Body Type:</span>'
            f'        <div class="bar-track"><div class="bar-fill" style="width: {body_score_pct};"></div></div>'
            f'        <span class="bar-pct font-mono">{body_score_pct}</span>'
            f'      </div>'
            f'    </div>'
            f'    <div class="reasons-block">'
            f'      <div class="reasons-heading">Forensic Match Rationale:</div>'
            f'      <ul class="reasons-list">{reasons_list}</ul>'
            f'    </div>'
            f'  </div>'
            f'</div>'
        )

    candidates_rendered = "".join(candidates_html) if candidates_html else (
        '<div class="no-records-notice">No candidate vehicles shortlisted.</div>'
    )

    colour_conf_pct = f"{evidence.colour_conf * 100:.1f}%"
    body_conf_pct = f"{evidence.body_type_conf * 100:.1f}%"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Forensic Case Report — {case_id} — Vehicle #{analysis.track_id}</title>
  <style>
    /* Reset & Base */
    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: #f8fafc;
      color: #0f172a;
      line-height: 1.5;
      font-size: 14px;
      -webkit-font-smoothing: antialiased;
    }}
    .font-mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}

    /* Toolbar (Screen Only) */
    .report-toolbar {{
      position: sticky;
      top: 0;
      z-index: 100;
      background-color: #0f172a;
      color: #f8fafc;
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      border-bottom: 1px solid #1e293b;
    }}
    .toolbar-brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
      font-size: 15px;
      letter-spacing: 0.5px;
    }}
    .toolbar-badge {{
      background-color: #0284c7;
      color: #ffffff;
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      font-weight: 700;
    }}
    .toolbar-actions {{
      display: flex;
      gap: 12px;
    }}
    .btn {{
      padding: 8px 16px;
      font-size: 13px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease-in-out;
      border: none;
      text-decoration: none;
    }}
    .btn-primary {{
      background-color: #2563eb;
      color: #ffffff;
    }}
    .btn-primary:hover {{
      background-color: #1d4ed8;
    }}
    .btn-secondary {{
      background-color: #334155;
      color: #f1f5f9;
      border: 1px solid #475569;
    }}
    .btn-secondary:hover {{
      background-color: #475569;
    }}

    /* Main Container */
    .report-wrapper {{
      max-width: 1020px;
      margin: 32px auto;
      background-color: #ffffff;
      padding: 48px;
      border-radius: 8px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
      border: 1px solid #e2e8f0;
    }}

    /* Header & Classification */
    .classification-bar {{
      text-align: center;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 2px;
      color: #475569;
      padding-bottom: 16px;
      border-bottom: 2px solid #0f172a;
      margin-bottom: 24px;
    }}
    .report-header-grid {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid #e2e8f0;
    }}
    .agency-title {{
      font-size: 20px;
      font-weight: 800;
      color: #0f172a;
      letter-spacing: -0.5px;
    }}
    .agency-subtitle {{
      font-size: 13px;
      color: #64748b;
      margin-top: 4px;
      font-weight: 500;
    }}
    .case-meta-box {{
      text-align: right;
      font-size: 12px;
      color: #475569;
      line-height: 1.6;
    }}
    .case-id-highlight {{
      font-size: 14px;
      font-weight: 700;
      color: #0f172a;
      font-family: monospace;
    }}

    /* Section Styling */
    .section-title {{
      font-size: 15px;
      font-weight: 800;
      letter-spacing: 0.5px;
      color: #0f172a;
      text-transform: uppercase;
      margin-top: 32px;
      margin-bottom: 16px;
      padding-bottom: 6px;
      border-bottom: 2px solid #cbd5e1;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }}
    .section-title .section-tag {{
      font-size: 11px;
      font-weight: 600;
      color: #64748b;
      text-transform: none;
    }}

    /* Target Vehicle Overview Card */
    .target-overview-box {{
      background-color: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .plate-tiles-wrapper {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 16px 0;
      justify-content: flex-start;
    }}
    .plate-char-tile {{
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 58px;
      padding: 8px 4px;
      border-radius: 6px;
      border: 1.5px solid #334155;
      background-color: #0f172a;
      color: #f8fafc;
    }}
    .plate-char-tile.tile-wildcard {{
      background-color: #451a03;
      border-color: #d97706;
      color: #fef3c7;
    }}
    .plate-char-tile .char-pos {{
      font-size: 10px;
      font-family: monospace;
      color: #94a3b8;
      margin-bottom: 2px;
    }}
    .plate-char-tile.tile-wildcard .char-pos {{
      color: #fcd34d;
    }}
    .plate-char-tile .char-val {{
      font-size: 26px;
      font-weight: 900;
      font-family: monospace;
      line-height: 1;
      margin: 4px 0;
    }}
    .tile-conf-badge {{
      font-size: 10px;
      font-family: monospace;
      font-weight: 700;
      padding: 2px 4px;
      border-radius: 3px;
    }}
    .conf-high {{
      background-color: #065f46;
      color: #a7f3d0;
    }}
    .conf-med {{
      background-color: #854d0e;
      color: #fef08a;
    }}
    .conf-low {{
      background-color: #991b1b;
      color: #fecaca;
    }}

    .target-attributes-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid #e2e8f0;
    }}
    .attr-box {{
      background-color: #ffffff;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
    }}
    .attr-label {{
      font-size: 11px;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .attr-val {{
      font-size: 15px;
      font-weight: 800;
      color: #0f172a;
      margin-top: 4px;
    }}
    .attr-conf {{
      font-size: 11px;
      color: #059669;
      font-family: monospace;
      font-weight: 600;
    }}

    /* Multi-frame Evidence Table */
    .table-responsive {{
      width: 100%;
      overflow-x: auto;
      margin-top: 12px;
    }}
    .evidence-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .evidence-table th {{
      background-color: #f1f5f9;
      color: #334155;
      font-weight: 700;
      text-align: left;
      padding: 10px 12px;
      border: 1px solid #cbd5e1;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .evidence-table td {{
      padding: 10px 12px;
      border: 1px solid #e2e8f0;
      vertical-align: middle;
    }}
    .evidence-table tr:nth-child(even) {{
      background-color: #f8fafc;
    }}
    .crop-cell {{
      text-align: center;
      width: 180px;
    }}
    .report-crop-img {{
      max-width: 160px;
      max-height: 90px;
      border-radius: 4px;
      border: 1px solid #cbd5e1;
      display: inline-block;
      object-fit: cover;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    .plate-crop {{
      max-height: 48px;
    }}
    .ocr-reading-cell code {{
      font-size: 15px;
      font-weight: 800;
      background-color: #0f172a;
      color: #38bdf8;
      padding: 4px 8px;
      border-radius: 4px;
    }}

    /* Alert Boxes */
    .alert-box {{
      padding: 16px 20px;
      border-radius: 8px;
      margin-bottom: 24px;
      border-left: 6px solid;
    }}
    .alert-success {{
      background-color: #ecfdf5;
      border-color: #10b981;
      color: #065f46;
    }}
    .alert-warning {{
      background-color: #fef2f2;
      border-color: #ef4444;
      color: #991b1b;
    }}
    .alert-title {{
      font-size: 15px;
      font-weight: 800;
      letter-spacing: 0.5px;
    }}
    .alert-desc {{
      font-size: 13px;
      margin-top: 4px;
      color: #334155;
    }}

    /* Candidate Cards */
    .candidate-card {{
      background-color: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 16px;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
      break-inside: avoid;
      page-break-inside: avoid;
    }}
    .candidate-card.primary-candidate {{
      border: 2px solid #2563eb;
      background-color: #f8faff;
    }}
    .candidate-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid #e2e8f0;
    }}
    .candidate-rank-badge {{
      font-size: 12px;
      font-weight: 800;
      padding: 4px 10px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }}
    .rank-1 {{
      background-color: #2563eb;
      color: #ffffff;
    }}
    .rank-other {{
      background-color: #64748b;
      color: #ffffff;
    }}
    .candidate-plate-group {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .diff-chips-row {{
      display: flex;
      gap: 2px;
    }}
    .diff-chip {{
      font-family: monospace;
      font-size: 14px;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 3px;
      display: inline-block;
    }}
    .diff-exact {{
      background-color: #d1fae5;
      color: #065f46;
      border: 1px solid #a7f3d0;
    }}
    .diff-wildcard {{
      background-color: #fef3c7;
      color: #92400e;
      border: 1px solid #fde68a;
    }}
    .diff-confusion {{
      background-color: #ede9fe;
      color: #5b21b6;
      border: 1px solid #ddd6fe;
    }}
    .diff-mismatch {{
      background-color: #fee2e2;
      color: #991b1b;
      border: 1px solid #fecaca;
    }}
    .candidate-plate-text {{
      font-family: monospace;
      font-size: 16px;
      font-weight: 800;
      color: #0f172a;
    }}
    .candidate-score-block {{
      text-align: right;
    }}
    .candidate-score-block .score-label {{
      font-size: 10px;
      text-transform: uppercase;
      color: #64748b;
      font-weight: 700;
      display: block;
    }}
    .candidate-score-block .score-value {{
      font-size: 18px;
      font-weight: 900;
      color: #0284c7;
    }}

    .candidate-meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-top: 14px;
      font-size: 13px;
    }}
    .meta-item .meta-k {{
      color: #64748b;
    }}
    .meta-item .meta-v {{
      color: #0f172a;
    }}

    .score-bars-container {{
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      background-color: #f1f5f9;
      padding: 10px 14px;
      border-radius: 6px;
    }}
    .score-bar-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
    }}
    .bar-title {{
      font-weight: 600;
      color: #475569;
      width: 90px;
    }}
    .bar-track {{
      flex: 1;
      height: 8px;
      background-color: #cbd5e1;
      border-radius: 4px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background-color: #0284c7;
      border-radius: 4px;
    }}
    .bar-pct {{
      font-size: 11px;
      font-weight: 700;
      color: #0f172a;
      width: 44px;
      text-align: right;
    }}

    .reasons-block {{
      margin-top: 14px;
      background-color: #fafaf9;
      border-left: 3px solid #0284c7;
      padding: 10px 14px;
      border-radius: 0 4px 4px 0;
    }}
    .reasons-heading {{
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      color: #475569;
      margin-bottom: 6px;
      letter-spacing: 0.5px;
    }}
    .reasons-list {{
      padding-left: 18px;
      font-size: 12px;
      color: #334155;
      line-height: 1.5;
    }}

    /* Forensic Audit Footnotes */
    .audit-section {{
      margin-top: 40px;
      padding-top: 20px;
      border-top: 2px dashed #cbd5e1;
      font-size: 11px;
      color: #64748b;
      line-height: 1.6;
    }}
    .audit-title {{
      font-weight: 800;
      text-transform: uppercase;
      color: #334155;
      margin-bottom: 6px;
      letter-spacing: 0.5px;
    }}

    /* Print Optimization Rules */
    @media print {{
      .no-print, .report-toolbar {{
        display: none !important;
      }}
      body {{
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 11pt;
      }}
      .report-wrapper {{
        box-shadow: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
      }}
      .candidate-card {{
        break-inside: avoid !important;
        page-break-inside: avoid !important;
      }}
      .evidence-table {{
        break-inside: auto;
      }}
      .evidence-table tr {{
        break-inside: avoid;
        page-break-inside: avoid;
      }}
      @page {{
        size: A4 portrait;
        margin: 12mm 15mm 15mm 15mm;
      }}
    }}
  </style>
</head>
<body>

  <!-- Sticky Non-Printable Investigation Toolbar -->
  <header class="report-toolbar no-print">
    <div class="toolbar-brand">
      <span class="toolbar-badge">AIML-01</span>
      <span>INVICTUS FORENSIC SURVEILLANCE REPORT</span>
    </div>
    <div class="toolbar-actions">
      <button class="btn btn-secondary" onclick="saveAsHtmlFile()">
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/><path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/></svg>
        Download HTML Report
      </button>
      <button class="btn btn-primary" onclick="window.print()">
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M2.5 8a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1z"/><path d="M5 1a2 2 0 0 0-2 2v2H2a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1v1a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-1h1a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-1V3a2 2 0 0 0-2-2H5zm1 2a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v2H6V3zm7 8h1a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1v3a1 1 0 0 0 1 1h1v-1a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v1zm-8 1a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v1a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-1z"/></svg>
        Print / Save as PDF
      </button>
    </div>
  </header>

  <!-- Printable Forensic Case Dossier -->
  <main class="report-wrapper">
    <div class="classification-bar">
      OFFICIAL FORENSIC RECORD // CONFIDENTIAL SURVEILLANCE AUDIT // HACK-A-SOL 5.0
    </div>

    <div class="report-header-grid">
      <div>
        <div class="agency-title">FORENSIC VEHICLE TRACEABILITY REPORT</div>
        <div class="agency-subtitle">Team Invictus — Automated Multi-Signal CCTV Vehicle Re-Identification Engine (AIML-01)</div>
      </div>
      <div class="case-meta-box">
        <div>Case Reference: <span class="case-id-highlight">{case_id}</span></div>
        <div>Target Track ID: <strong>#{analysis.track_id}</strong></div>
        <div>Generated: {timestamp_str}</div>
        <div>Surveillance Stream: Active Surveillance Node #01</div>
      </div>
    </div>

    <!-- Section 1: Target Overview & Consensus Plate -->
    <section>
      <div class="section-title">
        <span>Target Vehicle Identification Summary</span>
        <span class="section-tag">Multi-Frame Temporal Consensus</span>
      </div>

      <div class="target-overview-box">
        <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">
          Recovered Consensus License Plate (? denotes unreadable / motion-blurred slot)
        </div>
        <div class="plate-tiles-wrapper">
          {plate_tiles_rendered}
        </div>

        <div class="target-attributes-grid">
          <div class="attr-box">
            <div class="attr-label">Observed Colour</div>
            <div class="attr-val">{html.escape(evidence.colour.capitalize())}</div>
            <div class="attr-conf">Conf: {colour_conf_pct}</div>
          </div>
          <div class="attr-box">
            <div class="attr-label">Body Classification</div>
            <div class="attr-val">{html.escape(evidence.body_type.value.capitalize())}</div>
            <div class="attr-conf">Conf: {body_conf_pct}</div>
          </div>
          <div class="attr-box">
            <div class="attr-label">Keyframes Processed</div>
            <div class="attr-val">{len(evidence.frames_used)} Keyframes</div>
            <div class="attr-conf">High Sharpness Selection</div>
          </div>
          <div class="attr-box">
            <div class="attr-label">Decision Boundary</div>
            <div class="attr-val">65.0% Composite</div>
            <div class="attr-conf">Calibrated Gating</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Section 2: Multi-Frame Evidence Strip -->
    <section>
      <div class="section-title">
        <span>Frame-by-Frame Evidence Breakdown</span>
        <span class="section-tag">Keyframe Crops & OCR Extractions</span>
      </div>

      <div class="table-responsive">
        <table class="evidence-table">
          <thead>
            <tr>
              <th style="width: 140px;">Keyframe & Time</th>
              <th>Vehicle Bounding Box</th>
              <th>Cropped License Plate</th>
              <th>Raw Frame OCR</th>
              <th style="width: 90px; text-align: center;">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {frame_rows_rendered}
          </tbody>
        </table>
      </div>
    </section>

    <!-- Section 3: Ranked Top-5 Shortlist or No-Confident-Match Notice -->
    <section style="margin-top: 32px;">
      <div class="section-title">
        <span>Registry Match Evaluation & Shortlist</span>
        <span class="section-tag">Probabilistic Multi-Signal Scoring Engine</span>
      </div>

      {shortlist_header}

      <div class="candidates-container">
        {candidates_rendered}
      </div>
    </section>

    <!-- Section 4: Forensic Audit & Methodology -->
    <footer class="audit-section">
      <div class="audit-title">Forensic Chain-of-Custody & Algorithmic Notes</div>
      <p>1. <strong>Synthetic Registry Privacy Guarantee:</strong> All registered identities shown in this report utilize synthetic, non-PII test records (prefixed <code>SYN-OWN-xxxx</code>) generated strictly for algorithm benchmarking and verification.</p>
      <p>2. <strong>Temporal Consensus Rationale:</strong> Individual CCTV frames suffer from shutter roll, glare, and vehicle velocity distortion. The multi-frame consensus engine fuses character distributions across high-sharpness keyframes, reserving '?' wildcards for unresolved characters rather than hallucinating alphanumeric values.</p>
      <p>3. <strong>OCR Visual Confusion Tolerance:</strong> Domain-specific similarity matrices (e.g., 8↔B, 0↔O, 1↔I, 2↔Z) are applied when evaluating plate character matches, penalizing decoy registrations that lack RTO syntactic validity.</p>
      <p>4. <strong>Calibrated Verification Gate:</strong> If composite confidence fails to exceed 65.0%, the system strictly flags "No Confident Match", preventing false-positive identification in law enforcement applications.</p>
    </footer>
  </main>

  <script>
    function saveAsHtmlFile() {{
      const htmlContent = "<!DOCTYPE html>\\n" + document.documentElement.outerHTML;
      const blob = new Blob([htmlContent], {{ type: "text/html;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "invictus-forensic-report-track-{analysis.track_id:03d}.html";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}
  </script>
</body>
</html>"""
    return html_content
