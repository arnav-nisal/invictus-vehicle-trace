import React, { useEffect } from 'react';
import { FrameEvidence } from '../types';
import { resolveImageUrl } from '../api';

interface FrameLightboxModalProps {
  frame: FrameEvidence | null;
  trackId?: number;
  allFrames: FrameEvidence[];
  onClose: () => void;
  onSelectFrame: (frame: FrameEvidence) => void;
}

export const FrameLightboxModal: React.FC<FrameLightboxModalProps> = ({
  frame,
  trackId = 1,
  allFrames,
  onClose,
  onSelectFrame,
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowRight' && frame) {
        const currIdx = allFrames.findIndex((f) => f.frame_idx === frame.frame_idx);
        if (currIdx < allFrames.length - 1) {
          onSelectFrame(allFrames[currIdx + 1]);
        }
      } else if (e.key === 'ArrowLeft' && frame) {
        const currIdx = allFrames.findIndex((f) => f.frame_idx === frame.frame_idx);
        if (currIdx > 0) {
          onSelectFrame(allFrames[currIdx - 1]);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [frame, allFrames, onClose, onSelectFrame]);

  if (!frame) return null;

  const vehUrl = resolveImageUrl(frame.image_path);
  const plateUrl = resolveImageUrl(frame.plate_crop_path);
  const tsDisp =
    frame.timestamp_sec !== null && frame.timestamp_sec !== undefined
      ? `${frame.timestamp_sec.toFixed(2)}s`
      : 'N/A';
  const confPct = ((frame.confidence ?? 0) * 100).toFixed(1);
  const currIdx = allFrames.findIndex((f) => f.frame_idx === frame.frame_idx);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-obsidian/90 backdrop-blur-xl p-4 sm:p-6 lg:p-8 animate-fadeIn"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Frame Zoom Comparison Lightbox"
    >
      <div
        className="w-full max-w-5xl bg-obsidian-surface border border-white/10 rounded-3xl shadow-obsidian overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Lightbox Header */}
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between bg-obsidian-deep/80">
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-sans font-bold text-base text-white tracking-snugger">
                  High-Resolution Keyframe Forensic Lightbox
                </h3>
                <span className="px-2.5 py-0.5 rounded-full bg-white/10 text-white/90 text-xs font-mono font-medium border border-white/10">
                  Target #{trackId}
                </span>
                <span className="px-2.5 py-0.5 rounded-full bg-blue-500/20 text-blue-300 text-xs font-mono font-medium border border-blue-500/30">
                  Frame #{frame.frame_idx}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5 font-sans">
                Side-by-side zoomed optical crop vs localized plate extraction at t = {tsDisp}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500 font-mono hidden sm:inline-block">
              Use ← / → keys to navigate
            </span>
            <button
              type="button"
              onClick={onClose}
              className="w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 text-slate-300 hover:text-white flex items-center justify-center transition border border-white/10 active:scale-95"
              title="Close [Esc]"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Side-by-Side Zoomed Comparison */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6 bg-obsidian/60">
          {/* Left: Full Vehicle Crop */}
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs font-sans">
              <span className="text-white/80 font-semibold tracking-wide flex items-center gap-1.5">
                <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 17a2 2 0 11-4 0 2 2 0 014 0zM19 17a2 2 0 11-4 0 2 2 0 014 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13 16V6a1 1 0 00-1-1H4a1 1 0 00-1 1v10a1 1 0 001 1h1m8-1a1 1 0 01-1 1H9m4-1V8a1 1 0 011-1h2.586a1 1 0 01.707.293l3.414 3.414a1 1 0 01.293.707V16a1 1 0 01-1 1h-1m-6-1a1 1 0 001 1h1M5 17a2 2 0 104 0m-4 0a2 2 0 114 0m6 0a2 2 0 104 0m-4 0a2 2 0 114 0" />
                </svg>
                Optical Vehicle Crop
              </span>
              <span className="text-slate-400 font-mono text-[11px]">CCTV Aspect Ratio</span>
            </div>

            <div className="w-full h-72 sm:h-80 rounded-2xl bg-obsidian-deep border border-white/10 overflow-hidden flex items-center justify-center relative shadow-inner">
              {vehUrl ? (
                <img
                  src={vehUrl}
                  alt={`Vehicle Crop Frame ${frame.frame_idx}`}
                  className="w-full h-full object-contain"
                />
              ) : (
                <span className="font-mono text-xs text-slate-500">No Image Available</span>
              )}
              <div className="absolute top-3 left-3 px-3 py-1 rounded-full bg-obsidian/80 backdrop-blur-md text-[11px] font-mono text-white/90 border border-white/10">
                Tracking Box Active
              </div>
            </div>
          </div>

          {/* Right: Zoomed Plate Crop & OCR Data */}
          <div className="space-y-3 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-sans mb-3">
                <span className="text-white/80 font-semibold tracking-wide flex items-center gap-1.5">
                  <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  Plate Region Zoom & OCR
                </span>
                <span className="text-emerald-400 font-mono text-xs font-bold">
                  {confPct}% Confidence
                </span>
              </div>

              <div className="w-full h-44 rounded-2xl bg-obsidian-deep border border-white/10 overflow-hidden flex items-center justify-center p-4 relative shadow-inner">
                {plateUrl ? (
                  <img
                    src={plateUrl}
                    alt={`Plate Crop Frame ${frame.frame_idx}`}
                    className="w-full h-full object-contain filter contrast-125"
                  />
                ) : (
                  <span className="font-mono text-xs text-slate-500">Plate Crop Unavailable</span>
                )}
                <div className="absolute top-2 right-2 px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 text-[10px] font-mono border border-amber-500/30">
                  Contrast Boosted
                </div>
              </div>
            </div>

            {/* OCR Metacard */}
            <div className="p-4 rounded-2xl bg-obsidian-surface border border-white/10 space-y-3 mt-4">
              <div className="flex items-center justify-between pb-2 border-b border-white/10">
                <span className="text-xs font-sans text-slate-400">Raw Optical Character Reading</span>
                <span className="text-[11px] font-mono text-slate-500">Confidence: {confPct}%</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="font-mono text-2xl font-bold tracking-wider text-white">
                  {frame.ocr_text || 'UNREADABLE'}
                </div>
                <div className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-mono text-xs font-semibold border border-emerald-500/30">
                  PASSED GATE
                </div>
              </div>

              <p className="text-xs text-slate-400 font-sans leading-relaxed">
                Extracted during temporal sharp-frame sampling. Multi-frame consensus weights this crop with neighboring frames to resolve high-frequency noise.
              </p>
            </div>
          </div>
        </div>

        {/* Lightbox Footer & Frame Carousel Selector */}
        <div className="px-6 py-3.5 border-t border-white/10 bg-obsidian-deep/90 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={currIdx <= 0}
              onClick={() => currIdx > 0 && onSelectFrame(allFrames[currIdx - 1])}
              className="px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-sans font-medium transition disabled:opacity-30 disabled:pointer-events-none flex items-center gap-1.5 border border-white/10 active:scale-95"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
              </svg>
              <span>Previous Frame</span>
            </button>

            <button
              type="button"
              disabled={currIdx >= allFrames.length - 1}
              onClick={() => currIdx < allFrames.length - 1 && onSelectFrame(allFrames[currIdx + 1])}
              className="px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-sans font-medium transition disabled:opacity-30 disabled:pointer-events-none flex items-center gap-1.5 border border-white/10 active:scale-95"
            >
              <span>Next Frame</span>
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>

          <span className="text-xs text-slate-400 font-mono">
            {currIdx + 1} of {allFrames.length} keyframes
          </span>
        </div>
      </div>
    </div>
  );
};
