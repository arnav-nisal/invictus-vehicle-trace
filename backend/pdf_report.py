"""Publication-Grade Forensic PDF Summary Report Generator for Invictus.

Generates a forensic instrument report using ReportLab:
- Warm alabaster (#F7F7F5), obsidian (#111111), deep emerald (#047857), muted amber (#B45309).
- Scannable Executive Summary Box with key takeaways.
- Visual Evidence Gallery embedding real vehicle and license plate crops.
- Hard Plate Vector Tile Strip and Top-3 Advisory Ghost Predictions.
- Detailed Slot-by-Slot Forensic Audit Table.
- Vehicle Intelligence and Ranked Registry Candidates (focused on identification & sightings).
- Integrity and Audit Trail Footer with Page X of Y numbering.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("backend.pdf_report")

# Theme Palette
C_OBSIDIAN = colors.HexColor("#111111")
C_ALABASTER = colors.HexColor("#F7F7F5")
C_EMERALD = colors.HexColor("#047857")
C_EMERALD_BG = colors.HexColor("#ECFDF5")
C_EMERALD_BORDER = colors.HexColor("#A7F3D0")
C_AMBER = colors.HexColor("#B45309")
C_AMBER_BG = colors.HexColor("#FFFBEB")
C_CARD_BG = colors.HexColor("#F9F9F8")
C_BORDER_LIGHT = colors.HexColor("#E4E4E7")
C_TEXT_MUTED = colors.HexColor("#52525B")
C_TEXT_DARK = colors.HexColor("#18181B")


class NumberedCanvas(canvas.Canvas):
    """Canvas that performs a two-pass render to stamp 'Page X of Y' in the running footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        page_w, page_h = A4

        # Top thin accent line
        self.setStrokeColor(C_EMERALD)
        self.setLineWidth(2.0)
        self.line(36, page_h - 24, page_w - 36, page_h - 24)

        # Footer divider line
        self.setStrokeColor(C_BORDER_LIGHT)
        self.setLineWidth(0.75)
        self.line(36, 32, page_w - 36, 32)

        # Footer text
        self.setFont("Helvetica-Bold", 7)
        self.setFillColor(C_TEXT_MUTED)
        self.drawString(36, 20, "INVICTUS FORENSIC VEHICLE TRACEABILITY SYSTEM · AI-ASSISTED EVIDENCE REPORT")

        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.setFont("Helvetica", 7)
        self.drawRightString(page_w - 36, 20, page_str)

        self.restoreState()


def _safe_image(path: Optional[Union[str, Path]], max_w: float, max_h: float) -> Optional[Image]:
    """Safely construct a ReportLab Image maintaining aspect ratio, or None if missing."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None

    try:
        with PILImage.open(p) as img:
            iw, ih = img.size
        if iw <= 0 or ih <= 0:
            return None

        aspect = float(ih) / float(iw)
        target_w = min(max_w, float(iw))
        target_h = target_w * aspect
        if target_h > max_h:
            target_h = max_h
            target_w = target_h / aspect

        return Image(str(p), width=target_w, height=target_h)
    except Exception as e:
        logger.warning(f"Could not load image {p} for report: {e}")
        return None


def _resolve_image_path(
    img_ref: Optional[Union[str, Path]],
    t_dir: Path,
    base_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Resolve an image reference (URL, relative path, or absolute path) to a local Path."""
    if not img_ref:
        return None
    p = Path(img_ref)
    if p.is_file() and p.stat().st_size > 0:
        return p

    # 1. Check directly relative to t_dir or by filename inside t_dir
    if (t_dir / p.name).is_file():
        return t_dir / p.name
    clean_ref = str(img_ref).replace("\\", "/").lstrip("/")
    if (t_dir / clean_ref).is_file():
        return t_dir / clean_ref

    # 2. Check under subdirectories inside t_dir (e.g. plate_crops_all/unenhanced/...)
    parts = clean_ref.split("/")
    for i in range(len(parts)):
        sub = Path(*parts[i:])
        if (t_dir / sub).is_file():
            return t_dir / sub

    # 3. Check relative to base_dir or project root
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
    if (root / clean_ref).is_file():
        return root / clean_ref

    # 4. Strip outputs/ or test_outputs/ prefix and check root / outputs / rel or root / test_outputs / rel
    for prefix in ("outputs/", "test_outputs/"):
        if clean_ref.startswith(prefix):
            rel = clean_ref[len(prefix):]
            for folder in ("outputs", "test_outputs"):
                cand = root / folder / rel
                if cand.is_file():
                    return cand
            break

    return None


def generate_forensic_pdf(
    data: Dict[str, Any],
    output_pdf_path: Union[str, Path],
    track_dir: Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Generate a publication-grade forensic dossier PDF from verification_result data."""
    out_pdf = Path(output_pdf_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    t_dir = Path(track_dir) if track_dir else out_pdf.parent

    # Document geometry: A4 (595.27 x 841.89 points), 36pt (0.5 in) margins
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )
    content_width = A4[0] - 72  # 523.27 points

    # Typography Styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=C_OBSIDIAN,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=C_EMERALD,
        spaceAfter=8,
    )
    sec_heading = ParagraphStyle(
        "SecHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=C_OBSIDIAN,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=C_TEXT_DARK,
    )
    body_bold = ParagraphStyle(
        "BodyDarkBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=C_TEXT_DARK,
    )
    meta_label = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=C_TEXT_MUTED,
    )
    meta_val = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=C_TEXT_DARK,
    )
    exec_bullet = ParagraphStyle(
        "ExecBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=C_TEXT_DARK,
        leftIndent=12,
        spaceAfter=3,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=C_TEXT_DARK,
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=C_TEXT_DARK,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    story: List[Any] = []

    # -----------------------------------------------------------------------
    # 1. Header & Case Metadata Strip
    # -----------------------------------------------------------------------
    case_id = data.get("case_id") or f"INV-{data.get('job_id', 'UNKNOWN').upper()}-{data.get('track_key', 'TRK').upper()}"
    story.append(Paragraph("INVICTUS · FORENSIC VEHICLE IDENTIFICATION DOSSIER", title_style))
    story.append(Paragraph("HIGH-INTEGRITY COMPUTER VISION & PROBABILISTIC MATCHING REPORT", subtitle_style))

    # Metadata Grid
    is_merged = data.get("is_merged_track", False)
    target_id = data.get("target_id") or data.get("track_id", "Unknown")
    merged_ids = data.get("merged_track_ids", [target_id])
    target_str = f"Track #{target_id} (Merged IDs: {', '.join(str(m) for m in merged_ids)})" if is_merged else f"Track #{target_id}"

    now_iso = data.get("generated_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elapsed_sec = data.get("processing_time_sec", 0.0)

    meta_cells = [
        [
            Paragraph("CASE IDENTIFIER", meta_label),
            Paragraph("SOURCE VIDEO CLIP", meta_label),
            Paragraph("TARGET VEHICLE", meta_label),
            Paragraph("PIPELINE RUNTIME", meta_label),
        ],
        [
            Paragraph(f"<b>{case_id}</b>", meta_val),
            Paragraph(str(data.get("filename") or f"{data.get('job_id')}.mp4"), meta_val),
            Paragraph(f"<b>{target_str}</b>", meta_val),
            Paragraph(f"{elapsed_sec:.2f}s (Stages 1–4)", meta_val),
        ],
        [
            Paragraph("TIMESTAMP (UTC)", meta_label),
            Paragraph("INVESTIGATION STATUS", meta_label),
            Paragraph("ZERO-HALLUCINATION GUARD", meta_label),
            Paragraph("VLM PROVIDER / MODEL", meta_label),
        ],
        [
            Paragraph(str(now_iso)[:19], meta_val),
            Paragraph("<b>EVIDENCE RECONSTRUCTED</b>", meta_val),
            Paragraph("<font color='#047857'><b>ACTIVE (Hard-Blocked '?')</b></font>", meta_val),
            Paragraph(str(data.get("plate", {}).get("provider_used") or "Open-Source Cascade"), meta_val),
        ],
    ]

    col_w = content_width / 4.0
    meta_table = Table(meta_cells, colWidths=[col_w] * 4)
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDEDEC")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # 2. Executive Summary Callout Box
    # -----------------------------------------------------------------------
    plate_data = data.get("plate", {})
    final_plate = plate_data.get("final_plate") or data.get("final_plate", "??????????")
    overall_conf = plate_data.get("overall_confidence", data.get("overall_confidence", 0.0))
    state_valid = plate_data.get("valid_state_code", plate_data.get("state_code_valid", False))

    slots = plate_data.get("reconciled_slots", [])
    n_confirmed = sum(1 for s in slots if s.get("action") == "TRIPLE_CHECK_CONFIRMED")
    n_tie_break = sum(1 for s in slots if "TIE_BREAK" in s.get("action", "") or s.get("action") == "UNANIMOUS_LOW_CONF_RESCUE")
    n_unreadable = sum(1 for s in slots if s.get("final_char") == "?" or s.get("action") == "FORBIDDEN_KEEP_UNKNOWN")

    ghost_preds = plate_data.get("ghost_predictions", [])
    top_ghost = ghost_preds[0] if ghost_preds else {}
    top_ghost_str = f"<b>{top_ghost.get('plate')}</b> (Soft Score: {top_ghost.get('soft_score', 0.0):.3f})" if top_ghost else "None (High-confidence hard plate locked)"

    veh_data = data.get("vehicle", {})
    colour_str = str(veh_data.get("colour", "Unknown")).capitalize()
    body_str = str(veh_data.get("body_type", "Unknown")).replace("-", " ").capitalize()
    make_str = str(veh_data.get("make", "Unknown"))
    model_str = str(veh_data.get("model", "Unknown"))
    veh_profile = f"{colour_str} {make_str} {model_str} ({body_str})" if make_str != "Unknown" else f"{colour_str} {body_str}"

    reg_matches = data.get("registry_matches", [])
    top_match = reg_matches[0] if reg_matches else None
    if top_match:
        top_reg_str = f"Candidate #1: <b>{top_match.get('plate_number')}</b> ({top_match.get('make', '')} {top_match.get('model', '')}, Owner: {top_match.get('owner_name', 'Protected')}, RTO: {top_match.get('registered_rto', 'Unknown')}) with <b>{top_match.get('total_confidence', 0.0):.1%}</b> forensic match confidence."
    else:
        top_reg_str = "No close match identified in local SQLite registry (vehicle may be registered in another jurisdiction or unregistered)."

    exec_items = [
        Paragraph("<b>EXECUTIVE FORENSIC SUMMARY & KEY FINDINGS</b>", ParagraphStyle(
            "ExecHead", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=C_EMERALD
        )),
        Spacer(1, 3),
        Paragraph(
            f"• <b>Reconstructed Hard Plate:</b> Exact forensic reading <b>{final_plate}</b> at <b>{overall_conf:.1%}</b> overall confidence "
            f"({n_confirmed} Triple-Confirmed · {n_tie_break} Tie-Broken/Rescued · {n_unreadable} Unreadable slots). "
            f"Valid Indian RTO Code: <b>{'YES' if state_valid else 'NO'}</b>.",
            exec_bullet,
        ),
        Paragraph(
            f"• <b>Advisory Ghost Hypothesis:</b> Top beam-search re-ranking: {top_ghost_str}. "
            "<i>(Provides advisory candidate guidance for human operators; hard plate preserves strict zero-hallucination).</i>",
            exec_bullet,
        ),
        Paragraph(
            f"• <b>Extracted Vehicle Profile:</b> Visual intelligence confirms <b>{veh_profile}</b> "
            f"(Body: {veh_data.get('body_type_conf', 0.85):.0%}, Colour: {veh_data.get('colour_conf', 0.85):.0%}, Make: {veh_data.get('make_conf', 0.0):.0%}).",
            exec_bullet,
        ),
        Paragraph(
            f"• <b>Registry Traceability:</b> {top_reg_str}",
            exec_bullet,
        ),
    ]

    exec_table = Table([[exec_items]], colWidths=[content_width])
    exec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_EMERALD_BG),
        ("BOX", (0, 0), (-1, -1), 1.0, C_EMERALD_BORDER),
        ("LINEBEFORE", (0, 0), (0, -1), 3.5, C_EMERALD),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # 3. Visual Evidence Gallery (Embedded Real Crops)
    # -----------------------------------------------------------------------
    story.append(Paragraph("SECTION 1 — VISUAL EVIDENCE GALLERY", sec_heading))

    # Resolve image paths
    # Search in track_dir or data["evidence_urls"]
    ev_urls = data.get("evidence_urls") or {}

    veh_ref = (
        ev_urls.get("vehicle_contact_strip_url")
        or ev_urls.get("best_vehicle_crop_url")
        or (t_dir / "stage3_vehicle_crops_top3.png")
    )
    crop_vehicle = _safe_image(_resolve_image_path(veh_ref, t_dir, base_dir), content_width * 0.48, 120)
    if not crop_vehicle:
        # Fallback to single vehicle crop
        for v_sub in ("vehicle_crops_unenhanced", "vehicle_crops_raw", "vehicle_crops_enhanced"):
            raw_crop_dir = t_dir / v_sub
            if raw_crop_dir.is_dir():
                first_c = next(raw_crop_dir.glob("*.png"), None)
                if first_c:
                    crop_vehicle = _safe_image(first_c, content_width * 0.48, 120)
                    if crop_vehicle:
                        break

    plate_ref = (
        ev_urls.get("plate_grid_unenhanced_url")
        or ev_urls.get("plate_grid_enhanced_url")
        or ev_urls.get("clearest_plate_crop_url")
        or (t_dir / "vlm_evidence_unenhanced_grid.png")
    )
    crop_plate_grid = _safe_image(_resolve_image_path(plate_ref, t_dir, base_dir), content_width * 0.48, 120)
    if not crop_plate_grid:
        for p_sub in (
            t_dir / "plate_crops_selected" / "unenhanced",
            t_dir / "plate_crops_all" / "unenhanced",
            t_dir / "plate_crops_selected" / "enhanced",
        ):
            if p_sub.is_dir():
                first_p = next(p_sub.glob("*.png"), None)
                if first_p:
                    crop_plate_grid = _safe_image(first_p, content_width * 0.48, 120)
                    if crop_plate_grid:
                        break

    gallery_cells = [
        [
            Paragraph("<b>EVIDENCE A: Multi-Angle Vehicle Contact Strip</b>", meta_label),
            Paragraph("<b>EVIDENCE B: 2×2 License Plate Inspection Grid</b>", meta_label),
        ],
        [
            crop_vehicle if crop_vehicle else Paragraph("<i>Vehicle contact strip not available on disk.</i>", meta_val),
            crop_plate_grid if crop_plate_grid else Paragraph("<i>Plate inspection grid not available on disk.</i>", meta_val),
        ],
        [
            Paragraph("<font size=6.5>Top-3 sharpest vehicle observations extracted across CCTV track duration.</font>", meta_val),
            Paragraph("<font size=6.5>Multi-frame license plate captures (temporal consensus & readability-scored).</font>", meta_val),
        ],
    ]

    half_w = content_width / 2.0
    gallery_table = Table(gallery_cells, colWidths=[half_w, half_w])
    gallery_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDEDEC")),
        ("ALIGN", (0, 1), (-1, 1), "CENTER"),
        ("VALIGN", (0, 1), (-1, 1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(gallery_table)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # 4. Hard Plate Tile Strip & Ghost Predictions
    # -----------------------------------------------------------------------
    story.append(Paragraph("SECTION 2 — RECONSTRUCTED LICENSE PLATE & ADVISORY RE-RANKING", sec_heading))

    # Construct vector tile strip for plate characters
    tile_cells: List[List[Any]] = [[], []]
    tile_styles: List[Tuple] = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
    ]

    tile_w = min(28.0, (content_width - 20) / max(len(final_plate), 10))
    plate_len = len(final_plate)

    for idx in range(plate_len):
        ch = final_plate[idx]
        slot_info = slots[idx] if idx < len(slots) else {}
        tone = slot_info.get("ui_tile_tone", "unknown")

        if tone == "green":
            bg_c = C_EMERALD_BG
            bdr_c = C_EMERALD
            txt_c = "#047857"
            sub_lbl = "LOCKED"
        elif tone == "black":
            bg_c = colors.HexColor("#F4F4F5")
            bdr_c = C_OBSIDIAN
            txt_c = "#111111"
            sub_lbl = "ARBITR."
        else:
            bg_c = colors.HexColor("#FAFAFA")
            bdr_c = colors.HexColor("#D4D4D8")
            txt_c = "#71717A"
            sub_lbl = "UNKNOWN"

        tile_p = Paragraph(
            f"<font size=12 color='{txt_c}'><b>{ch}</b></font>",
            ParagraphStyle(f"Tile_{idx}", parent=styles["Normal"], alignment=1, leading=14),
        )
        idx_p = Paragraph(
            f"<font size=5.5 color='#71717A'>{idx}<br/>{sub_lbl}</font>",
            ParagraphStyle(f"Idx_{idx}", parent=styles["Normal"], alignment=1, leading=6.5),
        )

        tile_cells[0].append(tile_p)
        tile_cells[1].append(idx_p)

        tile_styles.extend([
            ("BACKGROUND", (idx, 0), (idx, 0), bg_c),
            ("BOX", (idx, 0), (idx, 0), 1.25, bdr_c),
        ])

    tile_table = Table(tile_cells, colWidths=[tile_w] * plate_len)
    tile_table.setStyle(TableStyle(tile_styles))

    # Top-3 Ghost Table
    ghost_rows = [
        [
            Paragraph("Rank", table_header),
            Paragraph("Advisory Hypothesis", table_header),
            Paragraph("Soft Score", table_header),
            Paragraph("Valid RTO", table_header),
            Paragraph("Resolved Slots", table_header),
        ]
    ]

    for gp in ghost_preds[:3]:
        r = gp.get("rank", 1)
        gp_plate = gp.get("plate", "??????????")
        sc = gp.get("soft_score", 0.0)
        vr = "YES" if gp.get("state_code_valid", False) else "NO"
        g_slots = ", ".join(str(i) for i in gp.get("ghost_slot_indices", [])) or "None (Exact)"

        ghost_rows.append([
            Paragraph(f"#{r}", table_cell_bold),
            Paragraph(f"<b>{gp_plate}</b>", table_cell_bold),
            Paragraph(f"{sc:.3f}", table_cell),
            Paragraph(vr, table_cell),
            Paragraph(f"Slot(s) {g_slots}", table_cell),
        ])

    if len(ghost_rows) == 1:
        ghost_rows.append([
            Paragraph("-", table_cell),
            Paragraph("No alternative ghost hypothesis needed (plate locked)", table_cell),
            Paragraph("--", table_cell),
            Paragraph("--", table_cell),
            Paragraph("--", table_cell),
        ])

    gh_table = Table(ghost_rows, colWidths=[36, 120, 65, 60, content_width - 281])
    gh_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_OBSIDIAN),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_CARD_BG]),
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDEDEC")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    story.append(tile_table)
    story.append(Spacer(1, 6))
    story.append(gh_table)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # 5. Slot-by-Slot Forensic Audit Table
    # -----------------------------------------------------------------------
    story.append(Paragraph("SECTION 3 — SLOT-BY-SLOT AUDIT & ARBITRATION TRACE", sec_heading))

    audit_headers = [
        Paragraph("Slot", table_header),
        Paragraph("Type", table_header),
        Paragraph("RapidOCR (A)", table_header),
        Paragraph("PARSeq (B)", table_header),
        Paragraph("Blind VLM", table_header),
        Paragraph("Arbitration Action", table_header),
        Paragraph("Final", table_header),
        Paragraph("Ghost Alternatives", table_header),
    ]
    audit_rows = [audit_headers]

    for idx in range(plate_len):
        s_info = slots[idx] if idx < len(slots) else {}
        stype = s_info.get("slot_type", "D" if idx >= 2 and idx not in (4, 5) else "L")
        pa_top = s_info.get("pipe_a_candidate", "--")
        pb_top = s_info.get("pipe_b_candidate", "--")
        v_top = s_info.get("vlm_disambiguated_char") or s_info.get("vlm_blind_raw_char") or "--"
        v_conf = s_info.get("vlm_conf", 0.0)
        action = s_info.get("action", "UNRESOLVED")
        fc = s_info.get("final_char", "?")
        tone = s_info.get("ui_tile_tone", "unknown")

        ghost_list = s_info.get("ghost_stack", [])
        g_str = ", ".join(f"{g['char']} ({g['soft_score']:.2f})" for g in ghost_list) if ghost_list else "--"

        # Action tag styling
        if action == "TRIPLE_CHECK_CONFIRMED":
            act_p = Paragraph("<font color='#047857'><b>TRIPLE_CONFIRMED</b></font>", table_cell)
        elif "TIE_BREAK" in action:
            act_p = Paragraph(f"<font color='#B45309'><b>{action}</b></font>", table_cell)
        elif action == "UNANIMOUS_LOW_CONF_RESCUE":
            act_p = Paragraph("<font color='#047857'><b>UNANIMOUS_RESCUE</b></font>", table_cell)
        elif action == "FORBIDDEN_KEEP_UNKNOWN":
            act_p = Paragraph("<font color='#71717A'><b>HARD_BLOCKED '?'</b></font>", table_cell)
        else:
            act_p = Paragraph(action, table_cell)

        final_color = "#047857" if tone == "green" else ("#111111" if tone == "black" else "#DC2626")

        audit_rows.append([
            Paragraph(str(idx), table_cell_bold),
            Paragraph(stype, table_cell),
            Paragraph(str(pa_top), table_cell),
            Paragraph(str(pb_top), table_cell),
            Paragraph(f"{v_top} ({v_conf:.2f})" if v_top != "--" else "--", table_cell),
            act_p,
            Paragraph(f"<font color='{final_color}'><b>{fc}</b></font>", table_cell_bold),
            Paragraph(g_str, table_cell),
        ])

    # Widths: Slot (24), Type (26), Pipe A (55), Pipe B (55), Blind VLM (58), Action (120), Final (28), Ghost (rest)
    col_w_audit = [24, 26, 56, 56, 62, 115, 28, content_width - 367]
    audit_table = Table(audit_rows, colWidths=col_w_audit)
    audit_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_OBSIDIAN),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_CARD_BG]),
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDEDEC")),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("ALIGN", (6, 0), (6, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))

    story.append(audit_table)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # 6. Stage 3 Vehicle Intelligence & Stage 4 Registry Candidates
    # -----------------------------------------------------------------------
    story.append(Paragraph("SECTION 4 — VEHICLE ATTRIBUTES & REGISTRY TRACEABILITY", sec_heading))

    # Two column: Left = Stage 3 Attributes, Right = Stage 4 Match Summary
    left_items = [
        [Paragraph("EXTRACTED VISUAL ATTRIBUTES", meta_label), Paragraph("", meta_label)],
        [Paragraph("Body Type:", table_cell), Paragraph(f"<b>{body_str}</b> ({veh_data.get('body_type_conf', 0.85):.0%})", table_cell)],
        [Paragraph("Exterior Colour:", table_cell), Paragraph(f"<b>{colour_str}</b> ({veh_data.get('colour_conf', 0.85):.0%})", table_cell)],
        [Paragraph("Make / Company:", table_cell), Paragraph(f"<b>{make_str}</b> ({veh_data.get('make_conf', 0.0):.0%})", table_cell)],
        [Paragraph("Model / Series:", table_cell), Paragraph(f"<b>{model_str}</b> ({veh_data.get('model_conf', 0.0):.0%})", table_cell)],
        [Paragraph("Vision Provider:", table_cell), Paragraph(f"{veh_data.get('provider_used', 'NVIDIA Build NIM')}", table_cell)],
    ]
    w_attr_col = (content_width - 8) / 2.0
    left_table = Table(left_items, colWidths=[w_attr_col * 0.40, w_attr_col * 0.60])
    left_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDEDEC")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    # Ranked Registry Table
    reg_rows = [
        [
            Paragraph("Rank", table_header),
            Paragraph("Plate Number", table_header),
            Paragraph("Confidence", table_header),
            Paragraph("Plate %", table_header),
            Paragraph("Visual %", table_header),
            Paragraph("Registered Owner / Jurisdiction", table_header),
        ]
    ]

    for cand in reg_matches[:4]:
        r_num = cand.get("rank", 1)
        p_num = cand.get("plate_number", "Unknown")
        tot_c = cand.get("total_confidence", cand.get("composite_score", 0.0))
        p_sc = cand.get("plate_score", 0.0)
        v_sc = cand.get("visual_score", 0.0)
        owner = cand.get("owner_name", "Protected")
        rto = cand.get("registered_rto") or cand.get("rto_city", "Unknown")
        veh_ident = f"{cand.get('colour', '')} {cand.get('make', '')} {cand.get('model', '')}".strip()

        reg_rows.append([
            Paragraph(f"#{r_num}", table_cell_bold),
            Paragraph(f"<b>{p_num}</b>", table_cell_bold),
            Paragraph(f"<b>{tot_c:.1%}</b>", table_cell_bold),
            Paragraph(f"{p_sc:.1%}", table_cell),
            Paragraph(f"{v_sc:.1%}", table_cell),
            Paragraph(f"<b>{owner}</b> · {rto} ({veh_ident})", table_cell),
        ])

    if len(reg_rows) == 1:
        reg_rows.append([
            Paragraph("-", table_cell),
            Paragraph("No candidate found", table_cell),
            Paragraph("--", table_cell),
            Paragraph("--", table_cell),
            Paragraph("--", table_cell),
            Paragraph("No matching record in current registry slice", table_cell),
        ])

    reg_table = Table(reg_rows, colWidths=[28, 75, 55, 45, 45, content_width - 248])
    reg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_OBSIDIAN),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_CARD_BG]),
        ("BOX", (0, 0), (-1, -1), 0.75, C_BORDER_LIGHT),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#EDEDEC")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    story.append(left_table)
    story.append(Spacer(1, 6))
    story.append(reg_table)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # 7. System Integrity & Audit Trail Footer
    # -----------------------------------------------------------------------
    integ_p = [
        Paragraph("<b>SYSTEM INTEGRITY & CHAIN-OF-CUSTODY AUDIT NOTES</b>", ParagraphStyle(
            "IntegHead", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=C_TEXT_MUTED
        )),
        Spacer(1, 2),
        Paragraph(
            "1. <b>Zero-Hallucination Protocol:</b> Characters categorized as unreadable by dual OCR engines are strictly hard-blocked as '?' placeholders in the Hard Reconciled Plate unless all 3 independent visual engines unanimously verify the character at high confidence.",
            meta_val,
        ),
        Paragraph(
            "2. <b>Independent Triple-Check Architecture:</b> RapidOCR (PaddleOCR weights) and PARSeq (Permuted Autoregressive Sequence) operate blindly across unenhanced and enhanced crops. The Vision-Language Model performs blind inference without prior OCR text leakage.",
            meta_val,
        ),
        Paragraph(
            "3. <b>Multi-Key Fallback & Fault Tolerance:</b> API requests utilize a multi-key pool with automated sequential failover across NVIDIA Build NIM, Groq Cloud, and local Ollama offline inference. All secret tokens are scrubbed from exported audit traces.",
            meta_val,
        ),
    ]

    integ_table = Table([[integ_p]], colWidths=[content_width])
    integ_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, C_BORDER_LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether(integ_table))

    # Build PDF with custom NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    logger.info(f"Forensic PDF report generated successfully: {out_pdf} ({out_pdf.stat().st_size} bytes)")
    return out_pdf
