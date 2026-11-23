import React, { useState, useRef, useEffect } from 'react';
import { VehicleEvidence, FrameEvidence } from '../types';
import { resolveImageUrl } from '../api';
import { FrameLightboxModal } from './FrameLightboxModal';
import { FilmstripSkeleton, PlateConsensusSkeleton } from './Skeletons';

interface MainStageProps {
  evidence: VehicleEvidence | null;
  isAnalyzing: boolean;
  knownHint?: string;
  onHypothesisChange?: (newHint: string) => void;
  onResetHypothesis?: () => void;
}

const COLOUR_HEX_MAP: Record<string, string> = {
  white: '#FFFFFF',
  silver: '#CBD5E1',
  grey: '#64748B',
  gray: '#64748B',
  black: '#0F172A',
  red: '#EF4444',
  blue: '#3B82F6',
  yellow: '#EAB308',
  orange: '#F97316',
  green: '#10B981',
  brown: '#78350F',
};

export const MainStage: React.FC<MainStageProps> = ({
  evidence,
  isAnalyzing,
  knownHint = '',
  onHypothesisChange,
  onResetHypothesis,
}) => {
  const [imageErrorMap, setImageErrorMap] = useState<Record<string, boolean>>({});
  const [selectedLightboxFrame, setSelectedLightboxFrame] = useState<FrameEvidence | null>(null);

  // 3D Tilt State for HSRP Number Plate
  const plateContainerRef = useRef<HTMLDivElement>(null);
  const [tilt, setTilt] = useState({ rotateX: 0, rotateY: 0, glareX: 50, glareY: 50 });
  const [isHovered, setIsHovered] = useState(false);

  // In-place slot editing state
  const [editingSlotIdx, setEditingSlotIdx] = useState<number | null>(null);
  const [slotValue, setSlotValue] = useState<string>('');
  const slotInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingSlotIdx !== null && slotInputRef.current) {
      slotInputRef.current.focus();
      slotInputRef.current.select();
    }
  }, [editingSlotIdx]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!plateContainerRef.current) return;
    const rect = plateContainerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const normX = x / rect.width - 0.5;
    const normY = y / rect.height - 0.5;

    // Constrain tilt to smooth, elegant range (-14 deg to 14 deg)
    const rotateX = -normY * 18;
    const rotateY = normX * 22;
    const glareX = (x / rect.width) * 100;
    const glareY = (y / rect.height) * 100;

    setTilt({ rotateX, rotateY, glareX, glareY });
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setTilt({ rotateX: 0, rotateY: 0, glareX: 50, glareY: 50 });
  };

  if (isAnalyzing && !evidence) {
    return (
      <div className="space-y-6">
        <PlateConsensusSkeleton />
        <FilmstripSkeleton />
      </div>
    );
  }

  if (!evidence) {
    return (
      <div className="rounded-3xl border border-[rgba(15,23,42,0.06)] bg-white p-12 text-center text-slate-400 font-sans shadow-bento">
        <p className="text-sm font-medium">Select a suspect track to inspect multi-signal forensic evidence.</p>
      </div>
    );
  }

  const consensus = evidence.consensus_plate || '';
  const confidences = Array.isArray(evidence.char_confidences) ? evidence.char_confidences : [];
  const colourRaw = evidence.colour || 'grey';
  const colourHex = COLOUR_HEX_MAP[colourRaw.toLowerCase()] || '#64748B';
  const trackId = evidence.track_id ?? 1;
  const frames = Array.isArray(evidence.frames_used) ? evidence.frames_used : [];
  const colourConfPct = ((evidence.colour_conf ?? 0) * 100).toFixed(1);
  const bodyTypeStr = String(evidence.body_type || 'vehicle');
  const bodyConfPct = ((evidence.body_type_conf ?? 0) * 100).toFixed(1);

  // Derive current display plate incorporating any active investigator hypothesis
  // If knownHint exists and matches length or replaces parts, we compute the active display plate
  const consensusChars = consensus.split('');
  const hintChars = knownHint.split('');
  const activePlateChars = consensusChars.map((c, i) => {
    if (hintChars[i] && hintChars[i] !== '?') {
      return hintChars[i];
    }
    return c;
  });

  const hasManualHypothesis = Boolean(
    knownHint && knownHint.replace(/\?/g, '').length > 0 && knownHint !== consensus
  );

  const startEditingSlot = (idx: number, currentChar: string) => {
    setEditingSlotIdx(idx);
    setSlotValue(currentChar === '?' ? '' : currentChar);
  };

  const commitSlotEdit = (idx: number, newChar: string) => {
    const sanitized = (newChar || '?').toUpperCase().slice(-1) || '?';
    const updatedChars = [...activePlateChars];
    updatedChars[idx] = sanitized;
    const newPlateHypothesis = updatedChars.join('');

    setEditingSlotIdx(null);
    if (onHypothesisChange) {
      onHypothesisChange(newPlateHypothesis);
    }
  };

  const handleSlotKeyDown = (e: React.KeyboardEvent, idx: number) => {
    if (e.key === 'Enter') {
      commitSlotEdit(idx, slotValue);
    } else if (e.key === 'Escape') {
      setEditingSlotIdx(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Deep Obsidian Optical Viewport Card */}
      <div className="rounded-4xl bg-obsidian border border-white/10 shadow-obsidian p-6 sm:p-8 text-white relative overflow-hidden">
        {/* Subtle ambient lighting backdrop */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-80 h-80 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none" />

        {/* Viewport Top Header & Floating Frosted HUD Strip */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-6 border-b border-white/10 relative z-10">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse shadow-sm shadow-emerald-400" />
              <h2 className="text-lg font-bold font-sans text-white tracking-snugger">
                Optical Viewport & Consensus Fusion
              </h2>
              <span className="px-3 py-0.5 rounded-full bg-white/10 text-white text-xs font-mono font-semibold border border-white/10">
                Track #{trackId}
              </span>
            </div>
            <p className="text-xs text-slate-400 font-sans mt-1">
              Temporal multi-frame sharpness synthesis across {frames.length} CCTV surveillance keyframes
            </p>
          </div>

          {/* Floating Frosted-Glass HUD Badges */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Color HUD */}
            <div className="frosted-glass-pill px-3.5 py-1.5 rounded-2xl flex items-center gap-2 text-xs font-sans">
              <span
                className="w-3.5 h-3.5 rounded-full border border-white/30 shadow-sm flex-shrink-0"
                style={{ backgroundColor: colourHex }}
                title={colourRaw}
              />
              <span className="text-slate-400 text-[11px] font-medium">Color:</span>
              <span className="text-white font-semibold capitalize">{colourRaw}</span>
              <span className="text-emerald-400 font-mono text-[11px] font-medium">({colourConfPct}%)</span>
            </div>

            {/* Body Type HUD */}
            <div className="frosted-glass-pill px-3.5 py-1.5 rounded-2xl flex items-center gap-2 text-xs font-sans">
              <svg className="w-3.5 h-3.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              <span className="text-slate-400 text-[11px] font-medium">Body:</span>
              <span className="text-white font-semibold uppercase">{bodyTypeStr}</span>
              <span className="text-blue-400 font-mono text-[11px] font-medium">({bodyConfPct}%)</span>
            </div>

            {/* Reset to AI Reading Pill Button */}
            {hasManualHypothesis && onResetHypothesis && (
              <button
                type="button"
                onClick={onResetHypothesis}
                className="tactile-button px-3 py-1.5 rounded-2xl bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 font-sans text-xs font-semibold flex items-center gap-1.5 shadow-sm active:scale-95 transition"
                title="Reset custom hypothesis back to AI consensus reading"
              >
                <svg className="w-3.5 h-3.5 text-amber-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span>Reset to AI Reading</span>
              </button>
            )}
          </div>
        </div>

        {/* 3D-Perspective Interactive HSRP Number Plate Centerpiece */}
        <div className="py-8 sm:py-10 flex flex-col items-center justify-center relative perspective-container">
          <div className="text-center mb-3">
            <span className="text-[11px] font-sans font-medium uppercase tracking-wider text-slate-400 inline-flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
              </svg>
              Tactile 3D HSRP Plate • Click any ? slot to type hypothesis in real-time
            </span>
          </div>

          {/* 3D Tilting Card Container */}
          <div
            ref={plateContainerRef}
            onMouseMove={handleMouseMove}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            className="w-full max-w-xl cursor-pointer preserve-3d transition-transform ease-out"
            style={{
              transform: `perspective(1000px) rotateX(${tilt.rotateX}deg) rotateY(${tilt.rotateY}deg) scale3d(${
                isHovered ? 1.02 : 1
              }, ${isHovered ? 1.02 : 1}, 1)`,
              transitionDuration: isHovered ? '80ms' : '500ms',
            }}
          >
            {/* Authentic HSRP Number Plate Physical Card */}
            <div className="hsrp-plate-surface rounded-2xl p-2 sm:p-2.5 relative border border-slate-300 shadow-hsrp overflow-hidden flex items-stretch gap-2.5 select-none">
              {/* Dynamic Glare Specular Highlight */}
              <div
                className="absolute inset-0 pointer-events-none transition-opacity duration-300"
                style={{
                  background: `radial-gradient(circle 320px at ${tilt.glareX}% ${tilt.glareY}%, rgba(255,255,255,0.45) 0%, rgba(255,255,255,0) 70%)`,
                  opacity: isHovered ? 1 : 0.2,
                }}
              />

              {/* Blue IND Hologram Strip (Left Side) */}
              <div className="w-10 sm:w-12 rounded-xl bg-[#003893] text-white flex flex-col items-center justify-between py-2 px-1 relative overflow-hidden flex-shrink-0 shadow-sm border border-[#002B7F]">
                {/* Hologram Chakra Chrome Emblem */}
                <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-amber-200/90 via-slate-100 to-cyan-200/90 flex items-center justify-center p-0.5 shadow-inner border border-white/40">
                  <svg className="w-4 h-4 text-[#003893]" viewBox="0 0 24 24" fill="currentColor">
                    <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
                    <circle cx="12" cy="12" r="2.5" fill="currentColor" />
                    {Array.from({ length: 12 }).map((_, i) => (
                      <line
                        key={i}
                        x1="12"
                        y1="3"
                        x2="12"
                        y2="21"
                        stroke="currentColor"
                        strokeWidth="0.8"
                        transform={`rotate(${i * 15} 12 12)`}
                      />
                    ))}
                  </svg>
                </div>

                {/* IND Authentic Text */}
                <span className="font-sans font-black text-sm tracking-widest text-white drop-shadow-sm">
                  IND
                </span>

                {/* Laser etched micro-PIN */}
                <span className="font-mono text-[7px] text-white/60 tracking-tighter">
                  AA20
                </span>
              </div>

              {/* Outer Embossed Rim Border Line */}
              <div className="flex-1 flex items-center justify-between border-2 border-black/80 rounded-xl px-2 sm:px-3 py-2 bg-gradient-to-b from-white via-[#FAFAFA] to-[#EAECEF] relative">
                {/* Embossed Characters Row */}
                <div className="w-full flex items-center justify-center gap-1 sm:gap-2">
                  {activePlateChars.map((char, index) => {
                    const conf = confidences[index] !== undefined ? confidences[index] : 0.85;
                    const confPct = Math.round(conf * 100);
                    const isUnknown = char === '?';
                    const isManual = hintChars[index] && hintChars[index] !== '?' && hintChars[index] !== consensusChars[index];
                    const isEditing = editingSlotIdx === index;

                    return (
                      <div
                        key={index}
                        onClick={() => startEditingSlot(index, char)}
                        className={`relative flex flex-col items-center justify-center rounded-xl transition-all cursor-pointer ${
                          isEditing
                            ? 'ring-2 ring-blue-500 bg-blue-50/80 shadow-md scale-105 z-20'
                            : isManual
                            ? 'bg-emerald-50/90 border border-emerald-500/80 shadow-sm'
                            : isUnknown
                            ? 'bg-amber-100/90 border-2 border-dashed border-amber-500 animate-pulse hover:bg-amber-200/90'
                            : 'hover:bg-black/5'
                        } px-1.5 sm:px-2.5 py-1 sm:py-1.5`}
                        title={
                          isUnknown
                            ? `Slot #${index + 1}: Unresolved. Click to test witness hypothesis.`
                            : `Slot #${index + 1}: '${char}' (${confPct}% confidence). Click to edit.`
                        }
                      >
                        {/* Interactive In-Place Input */}
                        {isEditing ? (
                          <input
                            ref={slotInputRef}
                            type="text"
                            maxLength={1}
                            value={slotValue}
                            onChange={(e) => setSlotValue(e.target.value.toUpperCase())}
                            onBlur={() => commitSlotEdit(index, slotValue)}
                            onKeyDown={(e) => handleSlotKeyDown(e, index)}
                            className="w-7 sm:w-9 h-10 sm:h-12 text-center font-mono text-2xl sm:text-3xl font-black bg-white border border-blue-400 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-600 shadow-inner"
                          />
                        ) : (
                          <span
                            className={`font-mono text-2xl sm:text-3xl lg:text-4xl font-black tracking-tight leading-none hsrp-char-embossed ${
                              isManual
                                ? 'text-emerald-700 font-extrabold'
                                : isUnknown
                                ? 'text-amber-600'
                                : 'text-slate-950'
                            }`}
                          >
                            {char}
                          </span>
                        )}

                        {/* Confidence / State Micro-Indicator */}
                        <div className="flex items-center gap-0.5 mt-1">
                          <span
                            className={`text-[9px] font-mono font-bold tabular-nums ${
                              isManual
                                ? 'text-emerald-700'
                                : isUnknown
                                ? 'text-amber-700 font-black'
                                : conf >= 0.85
                                ? 'text-slate-500'
                                : 'text-rose-600'
                            }`}
                          >
                            {isManual ? 'HYPO' : isUnknown ? '?' : `${confPct}%`}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Laser Security Watermark Badge (Right Bottom) */}
                <div className="absolute bottom-1 right-2 pointer-events-none opacity-40">
                  <span className="font-mono text-[8px] tracking-wider text-slate-800 uppercase font-semibold">
                    SEC-HSRP
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Guidance Pill */}
          <div className="mt-4 flex items-center gap-3">
            <span className="text-xs text-slate-400 font-sans">
              Hypothesis testing: Click any slot to override OCR blur with witness intelligence.
            </span>
          </div>
        </div>

        {/* Clickable Horizontal Keyframe & Plate-Crop Filmstrip */}
        <div className="pt-6 border-t border-white/10 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              <h3 className="font-sans font-bold text-xs uppercase tracking-wider text-white">
                Keyframe Evidence Filmstrip
              </h3>
              <span className="px-2 py-0.5 rounded-full bg-white/10 text-slate-300 text-[10px] font-mono border border-white/10">
                {frames.length} FRAMES
              </span>
            </div>
            <span className="text-[11px] text-slate-400 font-sans">
              Click any frame to inspect side-by-side zoomed lightbox ↗
            </span>
          </div>

          {frames.length === 0 ? (
            <div className="p-8 text-center text-slate-500 font-sans text-xs">
              No keyframes recorded for this track
            </div>
          ) : (
            <div className="flex gap-4 overflow-x-auto pb-3 pt-1 dark-scroll">
              {frames.map((frame, idx) => {
                const frameIdx = frame.frame_idx ?? idx;
                const vehUrl = resolveImageUrl(frame.image_path);
                const plateUrl = resolveImageUrl(frame.plate_crop_path);
                const vehErrKey = `veh-${frameIdx}`;
                const plateErrKey = `plate-${frameIdx}`;
                const tsDisp =
                  frame.timestamp_sec !== null && frame.timestamp_sec !== undefined
                    ? `${frame.timestamp_sec.toFixed(2)}s`
                    : 'N/A';
                const frameConfPct = ((frame.confidence ?? 0) * 100).toFixed(1);

                return (
                  <div
                    key={frameIdx}
                    onClick={() => setSelectedLightboxFrame(frame)}
                    className="flex-shrink-0 w-64 rounded-2xl bg-obsidian-surface border border-white/10 p-3 flex flex-col justify-between gap-2.5 hover:border-white/30 hover:scale-[1.01] transition duration-150 cursor-pointer shadow-md group"
                    title="Click to open full zoomed comparison modal"
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between font-mono text-[11px]">
                      <span className="px-2 py-0.5 rounded-lg bg-white/10 text-white font-medium border border-white/10">
                        Frame #{frameIdx}
                      </span>
                      <span className="text-slate-400 font-mono">t = {tsDisp}</span>
                    </div>

                    {/* Vehicle Crop Thumbnail */}
                    <div className="w-full h-28 rounded-xl bg-obsidian-deep border border-white/5 overflow-hidden relative flex items-center justify-center">
                      {vehUrl && !imageErrorMap[vehErrKey] ? (
                        <img
                          src={vehUrl}
                          alt={`Vehicle ${frameIdx}`}
                          onError={() =>
                            setImageErrorMap((prev) => ({ ...prev, [vehErrKey]: true }))
                          }
                          className="w-full h-full object-cover group-hover:scale-105 transition duration-200"
                        />
                      ) : (
                        <span className="font-mono text-[10px] text-slate-600">
                          VEHICLE CROP #{frameIdx}
                        </span>
                      )}
                      <div className="absolute top-1.5 left-1.5 px-2 py-0.5 rounded-md bg-obsidian/80 backdrop-blur-md text-[9px] font-sans font-medium text-slate-300 border border-white/10">
                        Vehicle Crop
                      </div>
                    </div>

                    {/* Plate Crop Thumbnail */}
                    <div className="w-full h-12 rounded-xl bg-obsidian-deep border border-white/5 overflow-hidden relative flex items-center justify-center">
                      {plateUrl && !imageErrorMap[plateErrKey] ? (
                        <img
                          src={plateUrl}
                          alt={`Plate ${frameIdx}`}
                          onError={() =>
                            setImageErrorMap((prev) => ({ ...prev, [plateErrKey]: true }))
                          }
                          className="w-full h-full object-contain filter contrast-125"
                        />
                      ) : (
                        <span className="font-mono text-[10px] text-slate-600">
                          PLATE CROP #{frameIdx}
                        </span>
                      )}
                      <div className="absolute top-1 left-1 px-1.5 py-0.2 rounded bg-obsidian/80 backdrop-blur-md text-[8px] font-sans font-medium text-slate-300 border border-white/10">
                        Plate
                      </div>
                    </div>

                    {/* OCR Text and Confidence */}
                    <div className="flex items-center justify-between border-t border-white/10 pt-2 text-xs">
                      <div>
                        <span className="text-[10px] text-slate-400 block font-sans">Raw OCR</span>
                        <span className="font-mono font-bold text-white tracking-wider">
                          {frame.ocr_text || 'N/A'}
                        </span>
                      </div>
                      <div className="text-right">
                        <span className="text-[10px] text-slate-400 block font-sans">Confidence</span>
                        <span className="font-mono font-bold text-emerald-400">
                          {frameConfPct}%
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Side-by-Side Zoomed Comparison Lightbox Modal */}
      {selectedLightboxFrame && (
        <FrameLightboxModal
          frame={selectedLightboxFrame}
          trackId={trackId}
          allFrames={frames}
          onClose={() => setSelectedLightboxFrame(null)}
          onSelectFrame={(f) => setSelectedLightboxFrame(f)}
        />
      )}
    </div>
  );
};
