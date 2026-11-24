import React, { useState } from 'react';
import { Eye, X, CheckCircle2 } from 'lucide-react';
import { EvidenceUrls, VerificationResult } from '../types';

interface PlateGhostStackProps {
  plate: VerificationResult['plate'];
  evidenceUrls: EvidenceUrls;
  isMergedTrack: boolean;
  mergedTrackIds: number[];
  hoveredCandidatePlate: string | null;
}

export const PlateGhostStack: React.FC<PlateGhostStackProps> = ({
  plate,
  evidenceUrls,
  isMergedTrack,
  mergedTrackIds,
  hoveredCandidatePlate,
}) => {
  const [hoveredSlotIdx, setHoveredSlotIdx] = useState<number | null>(null);
  const [isEvidenceModalOpen, setIsEvidenceModalOpen] = useState<boolean>(false);

  const slots = plate.reconciled_slots || [];
  const numSlots = slots.length;

  // Up to Top 3 ghost rows
  const ghostPredictions = plate.ghost_predictions || [];
  const topGhosts = ghostPredictions.slice(0, 3);

  // We want to visually stack Row #3 (top, 0.25 opacity), Row #2 (middle, 0.52 opacity), Row #1 (bottom, 0.88 opacity)
  // Let's create an ordered array for top-to-bottom rendering
  // If topGhosts has [rank1, rank2, rank3], in top-to-bottom order it should be: [rank3, rank2, rank1]
  const renderedGhostRows = [...topGhosts].reverse();

  // Find Ghost #1 prediction string for comparison
  const ghost1 = topGhosts.find((g) => g.rank === 1) || topGhosts[0];
  const ghost1Plate = ghost1?.plate || '';

  // Two-beat stagger timing
  const beat2StartDelay = numSlots * 68 + 60;

  return (
    <div className="bg-white rounded-2xl border border-[#E6E6E2] p-6 shadow-[0_1px_3px_rgba(0,0,0,0.02)] flex flex-col gap-5">
      {/* 1. Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#E6E6E2] pb-4">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs font-semibold tracking-wider text-[#111111] uppercase">
            Reconstructed License Plate
          </span>

          {isMergedTrack && (
            <span className="font-mono text-[11px] font-medium bg-[#F1F1EE] text-[#111111] px-2.5 py-0.5 rounded-full border border-[#E6E6E2]">
              Merged IDs: {mergedTrackIds.join(', ')}
            </span>
          )}
        </div>

        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#F0FDF4] border border-[#A7F3D0] text-[#047857] text-xs font-mono font-medium">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>{Math.round(plate.overall_confidence * 100)}% Overall Confidence</span>
        </div>
      </div>

      {/* 2. Pixel-Locked CSS Grid Structure */}
      <div className="w-full overflow-x-auto pb-2">
        <div
          className="min-w-[620px] flex flex-col gap-2"
          style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          {/* A. Rising Ghost Prediction Stack (Stacked Upward: Row #3 -> Row #2 -> Row #1) */}
          <div className="flex flex-col gap-1.5 pb-1">
            {renderedGhostRows.map((gp) => {
              const rank = gp.rank;
              // Base opacities per specification: Row #3: 0.25, Row #2: 0.52, Row #1: 0.88
              const baseOpacity = rank === 3 ? 0.25 : rank === 2 ? 0.52 : 0.88;
              const isSyncHighlighted = hoveredCandidatePlate === gp.plate;
              const effectiveOpacity = isSyncHighlighted ? 1.0 : baseOpacity;

              return (
                <div
                  key={gp.rank}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: `56px repeat(${numSlots}, minmax(0, 1fr))`,
                    columnGap: '8px',
                    opacity: effectiveOpacity,
                    transition: 'opacity 200ms cubic-bezier(0.16, 1, 0.3, 1), transform 200ms cubic-bezier(0.16, 1, 0.3, 1)',
                    animation: `ghostRowRise 260ms cubic-bezier(0.16, 1, 0.3, 1) ${
                      beat2StartDelay + (3 - rank) * 60
                    }ms both`,
                  }}
                  className={`items-center py-1 px-1 rounded-lg ${
                    isSyncHighlighted ? 'bg-emerald-50/50 ring-1 ring-[#10B981]' : ''
                  }`}
                >
                  {/* Left 56px track: quiet monospace rank + score */}
                  <div className="text-[11px] text-[#6B6B66] font-mono select-none truncate">
                    #{gp.rank} · {Math.round(gp.soft_score * 100)}%
                  </div>

                  {/* Character columns aligned with hard plate slots */}
                  {slots.map((slot, i) => {
                    const ghostChar = gp.plate ? gp.plate[i] || ' ' : ' ';
                    const isGreen = slot.ui_tile_tone === 'green';
                    const isUnknown = slot.ui_tile_tone === 'unknown' || slot.final_char === '?';
                    const isBlack = slot.ui_tile_tone === 'black';

                    let charClass = 'text-[#111111]';
                    if (isGreen) {
                      charClass = 'text-[#111111]/[0.14]';
                    } else if (isUnknown) {
                      charClass = 'text-[#111111] font-semibold';
                    } else if (isBlack && ghostChar !== slot.final_char) {
                      charClass = 'text-[#B45309] font-bold';
                    }

                    return (
                      <div
                        key={i}
                        className={`text-center text-sm sm:text-base font-bold select-none ${charClass}`}
                      >
                        {ghostChar}
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>

          {/* B. Giant Hard Plate Tile Row (Bottom Anchor) */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: `56px repeat(${numSlots}, minmax(0, 1fr))`,
              columnGap: '8px',
            }}
            className="items-stretch"
          >
            {/* Left 56px track: Muted IND Anchor Badge */}
            <div className="h-16 rounded-xl bg-[#F7F7F5] border border-[#E6E6E2] flex flex-col items-center justify-center text-[#111111] select-none shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
              <span className="text-[9px] text-[#6B6B66] font-semibold tracking-wider">IND</span>
              <div className="w-3.5 h-3.5 rounded-full border border-[#111111]/30 flex items-center justify-center text-[7px] text-[#111111] my-0.5">
                ☸
              </div>
            </div>

            {/* Hard Plate Character Tiles */}
            {slots.map((slot, idx) => {
              const isGreen = slot.ui_tile_tone === 'green';
              const isBlack = slot.ui_tile_tone === 'black';
              const isUnknown = slot.ui_tile_tone === 'unknown' || slot.final_char === '?';

              // Ghost #1 mismatch check for black tiles
              const ghost1Char = ghost1Plate[idx];
              const hasGhost1Conflict = isBlack && ghost1Char && ghost1Char !== slot.final_char;

              let tileStyle = 'bg-[#F1F1EE] text-[#111111] border-b-[3px] border-b-[#111111]';
              if (isGreen) {
                tileStyle = 'bg-[#F1F1EE] text-[#047857] border-b-[3px] border-b-[#047857]';
              } else if (isBlack && hasGhost1Conflict) {
                tileStyle = 'bg-[#F1F1EE] text-[#111111] border-b-[3px] border-b-[#B45309]';
              } else if (isUnknown) {
                tileStyle = 'bg-[#F7F7F5] text-[#111111] border border-dashed border-[#A1A1AA]';
              }

              return (
                <div
                  key={slot.slot_idx}
                  onMouseEnter={() => setHoveredSlotIdx(idx)}
                  onMouseLeave={() => setHoveredSlotIdx(null)}
                  style={{
                    animation: `hardTileReveal 220ms cubic-bezier(0.16, 1, 0.3, 1) ${idx * 68}ms both`,
                  }}
                  className={`relative h-16 rounded-xl flex flex-col items-center justify-center cursor-pointer transition-transform duration-150 hover:scale-[1.04] shadow-sm ${tileStyle}`}
                >
                  <span className="text-2xl sm:text-[30px] font-bold leading-none select-none">
                    {slot.final_char}
                  </span>

                  {/* Per-Slot Hover Tooltip */}
                  {hoveredSlotIdx === idx && (
                    <div className="absolute bottom-full mb-3 left-1/2 -translate-x-1/2 w-48 bg-[#111111] text-white p-3 rounded-xl shadow-2xl z-50 text-[11px] font-mono pointer-events-none flex flex-col gap-1.5 animate-in fade-in zoom-in-95">
                      <div className="flex justify-between border-b border-white/10 pb-1 text-[#E6E6E2]">
                        <span className="font-semibold text-white">Slot #{slot.slot_idx}</span>
                        <span className="text-emerald-400">{Math.round(slot.final_confidence * 100)}%</span>
                      </div>
                      <div className="flex justify-between text-white/80">
                        <span>Expected:</span>
                        <span className="text-white">{slot.expected_type}</span>
                      </div>
                      <div className="flex justify-between text-white/80">
                        <span>Pipe A:</span>
                        <span className="text-white">{slot.pipe_a_char || '—'}</span>
                      </div>
                      <div className="flex justify-between text-white/80">
                        <span>Pipe B:</span>
                        <span className="text-white">{slot.pipe_b_char || '—'}</span>
                      </div>
                      <div className="flex justify-between text-white/80">
                        <span>Blind VLM:</span>
                        <span className="text-white">{slot.vlm_char || '—'}</span>
                      </div>
                      <div className="flex justify-between text-white/80">
                        <span>Action:</span>
                        <span className="text-white truncate max-w-[100px]">{slot.action}</span>
                      </div>

                      {slot.ghost_stack && slot.ghost_stack.length > 0 && (
                        <div className="mt-1 pt-1 border-t border-white/10 flex flex-col gap-0.5">
                          <span className="text-[10px] text-white/50">Alternatives:</span>
                          <div className="flex flex-wrap gap-1">
                            {slot.ghost_stack.map((g, gi) => (
                              <span key={gi} className="px-1 py-0.5 rounded bg-white/10 text-[10px]">
                                {g.char} ({Math.round(g.soft_score * 100)}%)
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* 3. Subtle Card Footer */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-3 border-t border-[#E6E6E2] text-xs font-mono">
        {/* Left: Minimal Tone Legend */}
        <div className="flex items-center gap-4 text-[#6B6B66] text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-[#047857]" />
            <span>Green: Triple-Confirmed</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-[#111111]" />
            <span>Black: Rescued / Tie-Broken</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm border border-dashed border-[#A1A1AA] bg-[#F7F7F5]" />
            <span>Dashed: Unreadable</span>
          </div>
        </div>

        {/* Right: Inspect Plate Evidence Button */}
        <button
          onClick={() => setIsEvidenceModalOpen(true)}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-[#E6E6E2] bg-[#F1F1EE] hover:bg-[#EAEAE6] text-[#111111] transition-colors focus:outline-none"
        >
          <Eye className="w-3.5 h-3.5" />
          <span>Inspect Plate Evidence</span>
        </button>
      </div>

      {/* 4. Evidence Lightbox Modal */}
      {isEvidenceModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-3xl bg-white rounded-2xl border border-[#E6E6E2] p-6 shadow-2xl flex flex-col gap-4 animate-in fade-in zoom-in-95">
            <div className="flex items-center justify-between border-b border-[#E6E6E2] pb-3">
              <span className="font-mono text-xs font-semibold uppercase text-[#111111]">
                Forensic Plate Crops &amp; Resolution Evidence
              </span>
              <button
                onClick={() => setIsEvidenceModalOpen(false)}
                className="p-1 rounded-lg hover:bg-[#F1F1EE] text-[#6B6B66] transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {evidenceUrls.clearest_plate_crop_url && (
                <div className="flex flex-col gap-1.5">
                  <span className="font-mono text-[11px] text-[#6B6B66]">Clearest Plate Crop</span>
                  <div className="aspect-[3/1] bg-[#111111] rounded-xl overflow-hidden border border-[#E6E6E2] flex items-center justify-center">
                    <img
                      src={evidenceUrls.clearest_plate_crop_url}
                      alt="Clearest Plate Crop"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
              )}

              {evidenceUrls.plate_grid_unenhanced_url && (
                <div className="flex flex-col gap-1.5">
                  <span className="font-mono text-[11px] text-[#6B6B66]">Raw Keyframe Grid</span>
                  <div className="aspect-[3/1] bg-[#111111] rounded-xl overflow-hidden border border-[#E6E6E2] flex items-center justify-center">
                    <img
                      src={evidenceUrls.plate_grid_unenhanced_url}
                      alt="Unenhanced Grid"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
              )}

              {evidenceUrls.plate_grid_enhanced_url && (
                <div className="flex flex-col gap-1.5">
                  <span className="font-mono text-[11px] text-[#6B6B66]">Enhanced Ensemble Grid</span>
                  <div className="aspect-[3/1] bg-[#111111] rounded-xl overflow-hidden border border-[#E6E6E2] flex items-center justify-center">
                    <img
                      src={evidenceUrls.plate_grid_enhanced_url}
                      alt="Enhanced Grid"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setIsEvidenceModalOpen(false)}
                className="px-4 py-2 rounded-xl text-xs font-medium bg-[#111111] text-white hover:bg-[#262626]"
              >
                Close Evidence
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Animation Keyframe Style Block */}
      <style>{`
        @keyframes hardTileReveal {
          0% {
            opacity: 0;
            transform: translateY(6px);
          }
          100% {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes ghostRowRise {
          0% {
            transform: translateY(8px);
          }
          100% {
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
};
