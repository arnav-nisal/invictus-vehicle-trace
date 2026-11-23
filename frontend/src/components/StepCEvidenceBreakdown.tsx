import React, { useState } from 'react';
import { VehicleEvidence } from '../types';
import { resolveImageUrl } from '../api';

interface StepCEvidenceBreakdownProps {
  evidence: VehicleEvidence | null;
}

// Visual color map for swatch indicators
const COLOUR_HEX_MAP: Record<string, string> = {
  white: '#f8fafc',
  silver: '#94a3b8',
  grey: '#64748b',
  gray: '#64748b',
  black: '#18181b',
  red: '#ef4444',
  blue: '#3b82f6',
  yellow: '#eab308',
  orange: '#f97316',
  green: '#10b981',
  brown: '#78350f',
};

export const StepCEvidenceBreakdown: React.FC<StepCEvidenceBreakdownProps> = ({ evidence }) => {
  const [imageErrorMap, setImageErrorMap] = useState<Record<string, boolean>>({});

  if (!evidence) {
    return null;
  }

  const consensus = evidence?.consensus_plate || '';
  const confidences = Array.isArray(evidence?.char_confidences) ? evidence.char_confidences : [];
  const colourRaw = evidence?.colour || 'grey';
  const colourHex = COLOUR_HEX_MAP[colourRaw.toLowerCase()] || '#64748b';
  const trackId = evidence?.track_id ?? 0;
  const frames = Array.isArray(evidence?.frames_used) ? evidence.frames_used : [];
  const colourConfPct = ((evidence?.colour_conf ?? 0) * 100).toFixed(1);
  const bodyTypeStr = String(evidence?.body_type || 'vehicle');
  const bodyConfPct = ((evidence?.body_type_conf ?? 0) * 100).toFixed(1);

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 mb-6 gap-2">
        <div className="flex items-center gap-3">
          <span className="w-7 h-7 rounded bg-indigo-950 text-indigo-400 border border-indigo-800/80 font-mono text-xs font-bold flex items-center justify-center">
            C
          </span>
          <div>
            <h2 className="font-mono text-sm font-bold tracking-wider uppercase text-white">
              Multi-Frame Visual Evidence & Consensus OCR
            </h2>
            <p className="text-xs text-slate-400">
              Temporal sharpness fusion across high-speed surveillance keyframes (reserves ? for unreadable slots)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
            Target Track: <strong className="text-cyan-400">#{trackId}</strong>
          </span>
          <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
            Keyframes: <strong className="text-cyan-400">{frames.length}</strong>
          </span>
        </div>
      </div>

      {/* Part 1: Consensus Plate Character Tiles */}
      <div className="mb-6 bg-slate-950/70 border border-slate-800 rounded-xl p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
          <div className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            Recovered Consensus License Plate
          </div>
          <div className="text-[11px] font-mono text-amber-400 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-400 inline-block"></span>
            <span>? indicates unreadable / motion-blurred character (never invented)</span>
          </div>
        </div>

        {/* Tiles row */}
        <div className="flex flex-wrap gap-2.5 items-stretch py-2">
          {consensus.split('').map((char, index) => {
            const conf = confidences[index] !== undefined ? confidences[index] : 0.8;
            const confPct = Math.round(conf * 100);
            const isUnknown = char === '?';

            return (
              <div
                key={index}
                className={`flex flex-col items-center justify-between w-14 sm:w-16 py-2.5 px-1.5 rounded-lg border transition-all ${
                  isUnknown
                    ? 'bg-amber-950/40 border-amber-500/80 text-amber-300 ring-1 ring-amber-500/30'
                    : 'bg-slate-900 border-slate-700 text-slate-100 hover:border-slate-500'
                }`}
              >
                {/* Character index */}
                <span className="font-mono text-[10px] text-slate-500 font-semibold mb-1">
                  #{index + 1}
                </span>

                {/* Character Glyph */}
                <span
                  className={`font-mono text-2xl sm:text-3xl font-black tracking-tight leading-none my-1 ${
                    isUnknown ? 'text-amber-400 animate-pulse' : 'text-white'
                  }`}
                >
                  {char}
                </span>

                {/* Confidence Bar & Percentage */}
                <div className="w-full mt-2">
                  <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden mb-1">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        isUnknown
                          ? 'bg-amber-500'
                          : conf >= 0.85
                          ? 'bg-emerald-400'
                          : conf >= 0.60
                          ? 'bg-cyan-400'
                          : 'bg-rose-400'
                      }`}
                      style={{ width: `${confPct}%` }}
                    />
                  </div>
                  <span
                    className={`font-mono text-[10px] font-bold block text-center ${
                      isUnknown
                        ? 'text-amber-400'
                        : conf >= 0.85
                        ? 'text-emerald-400'
                        : 'text-slate-400'
                    }`}
                  >
                    {isUnknown ? 'UNKNOWN' : `${confPct}%`}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Part 2: Detected Physical Attributes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Colour Detection Card */}
        <div className="bg-slate-850/60 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
          <div className="flex items-center gap-3.5">
            <div
              className="w-10 h-10 rounded-lg border-2 border-slate-700 shadow-inner flex items-center justify-center relative"
              style={{ backgroundColor: colourHex }}
            >
              {colourRaw.toLowerCase() === 'white' && (
                <div className="w-2 h-2 rounded-full bg-slate-400" />
              )}
            </div>
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 block">
                DETECTED COLOUR
              </span>
              <span className="font-mono text-sm font-bold text-slate-200 capitalize">
                {colourRaw}
              </span>
            </div>
          </div>
          <div className="text-right">
            <span className="font-mono text-xs font-bold text-emerald-400">
              {colourConfPct}%
            </span>
            <span className="text-[10px] font-mono text-slate-500 block">CONFIDENCE</span>
          </div>
        </div>

        {/* Body Type Detection Card */}
        <div className="bg-slate-850/60 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-lg bg-slate-900 border border-slate-700 flex items-center justify-center font-mono font-bold text-cyan-400 text-xs">
              {bodyTypeStr.slice(0, 3).toUpperCase()}
            </div>
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 block">
                BODY CLASSIFICATION
              </span>
              <span className="font-mono text-sm font-bold text-slate-200 capitalize">
                {bodyTypeStr}
              </span>
            </div>
          </div>
          <div className="text-right">
            <span className="font-mono text-xs font-bold text-emerald-400">
              {bodyConfPct}%
            </span>
            <span className="text-[10px] font-mono text-slate-500 block">CONFIDENCE</span>
          </div>
        </div>
      </div>

      {/* Part 3: Horizontal Frame-by-Frame Evidence Strip */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-purple-400"></span>
            Frame-by-Frame Evidence Strip ({frames.length} Keyframes)
          </span>
          <span className="text-[11px] font-mono text-slate-500">
            Scroll horizontally to inspect crops →
          </span>
        </div>

        <div className="flex gap-4 overflow-x-auto pb-3 pt-1">
          {frames.map((frame, idx) => {
            if (!frame) return null;
            const frameIdx = frame.frame_idx ?? idx;
            const vehUrl = resolveImageUrl(frame.image_path);
            const plateUrl = resolveImageUrl(frame.plate_crop_path);
            const vehErrKey = `veh-${frameIdx}`;
            const plateErrKey = `plate-${frameIdx}`;
            const tsDisp = frame.timestamp_sec !== null && frame.timestamp_sec !== undefined
              ? `${frame.timestamp_sec.toFixed(2)}s`
              : 'N/A';
            const frameConfPct = ((frame.confidence ?? 0) * 100).toFixed(1);

            return (
              <div
                key={frameIdx}
                className="flex-shrink-0 w-72 bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 flex flex-col justify-between"
              >
                {/* Frame Header */}
                <div className="flex items-center justify-between mb-2.5 font-mono text-[11px]">
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-bold border border-slate-700">
                    FRAME #{frameIdx}
                  </span>
                  <span className="text-slate-400">
                    t = {tsDisp}
                  </span>
                </div>

                {/* Vehicle Bounding Box Crop */}
                <div className="w-full h-28 rounded-lg bg-slate-900 border border-slate-800 overflow-hidden relative mb-2 flex items-center justify-center">
                  {vehUrl && !imageErrorMap[vehErrKey] ? (
                    <img
                      src={vehUrl}
                      alt={`Vehicle crop ${frameIdx}`}
                      onError={() =>
                        setImageErrorMap((prev) => ({ ...prev, [vehErrKey]: true }))
                      }
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="font-mono text-[10px] text-slate-500">
                      VEHICLE CROP #{frameIdx}
                    </div>
                  )}
                  <div className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-slate-950/80 backdrop-blur-sm text-[9px] font-mono text-slate-400 border border-slate-800">
                    VEHICLE CROP
                  </div>
                </div>

                {/* License Plate Crop */}
                <div className="w-full h-14 rounded-lg bg-slate-900 border border-slate-800 overflow-hidden relative mb-2 flex items-center justify-center">
                  {plateUrl && !imageErrorMap[plateErrKey] ? (
                    <img
                      src={plateUrl}
                      alt={`Plate crop ${frameIdx}`}
                      onError={() =>
                        setImageErrorMap((prev) => ({ ...prev, [plateErrKey]: true }))
                      }
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="font-mono text-[10px] text-slate-500">
                      PLATE CROP #{frameIdx}
                    </div>
                  )}
                  <div className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-slate-950/80 backdrop-blur-sm text-[9px] font-mono text-slate-400 border border-slate-800">
                    PLATE CROP
                  </div>
                </div>

                {/* Raw Frame OCR Reading & Confidence */}
                <div className="flex items-center justify-between border-t border-slate-800/80 pt-2 font-mono text-xs">
                  <div>
                    <span className="text-[10px] text-slate-500 block">FRAME OCR</span>
                    <span className="font-bold text-cyan-300">
                      {frame.ocr_text || 'N/A'}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 block">CONFIDENCE</span>
                    <span className="font-bold text-emerald-400">
                      {frameConfPct}%
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
