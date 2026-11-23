import React, { useState } from 'react';
import { TrackSummary, VideoScanResponse } from '../types';
import { resolveImageUrl } from '../api';

interface StepBVehicleSelectorProps {
  scanResult: VideoScanResponse | null;
  selectedTrackId: number | null;
  onSelectTrack: (track: TrackSummary) => void;
  isAnalyzing: boolean;
}

export const StepBVehicleSelector: React.FC<StepBVehicleSelectorProps> = ({
  scanResult,
  selectedTrackId,
  onSelectTrack,
  isAnalyzing,
}) => {
  const [imageErrorMap, setImageErrorMap] = useState<Record<number, boolean>>({});

  if (!scanResult) {
    return null;
  }

  const tracks = Array.isArray(scanResult?.tracks) ? scanResult.tracks : [];
  const durationSec = scanResult?.duration_seconds !== undefined && scanResult?.duration_seconds !== null
    ? scanResult.duration_seconds.toFixed(1)
    : '0.0';
  const totalFrames = scanResult?.total_frames ?? 0;
  const fps = scanResult?.fps ?? 30;

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 mb-6 gap-3">
        <div className="flex items-center gap-3">
          <span className="w-7 h-7 rounded bg-blue-950 text-blue-400 border border-blue-800/80 font-mono text-xs font-bold flex items-center justify-center">
            B
          </span>
          <div>
            <h2 className="font-mono text-sm font-bold tracking-wider uppercase text-white">
              Detected Vehicle Tracks ({tracks.length} Tracks Identified)
            </h2>
            <p className="text-xs text-slate-400">
              Click a suspect vehicle track card to execute multi-frame consensus and registry match
            </p>
          </div>
        </div>

        {/* Video stream statistics */}
        <div className="flex items-center gap-2 font-mono text-xs text-slate-400">
          <span className="px-2 py-1 rounded bg-slate-800 border border-slate-700">
            Duration: <strong className="text-slate-200">{durationSec}s</strong>
          </span>
          <span className="px-2 py-1 rounded bg-slate-800 border border-slate-700">
            Frames: <strong className="text-slate-200">{totalFrames}</strong>
          </span>
          <span className="px-2 py-1 rounded bg-slate-800 border border-slate-700">
            FPS: <strong className="text-slate-200">{fps}</strong>
          </span>
        </div>
      </div>

      {/* Gallery of Detected Tracks */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {tracks.map((track, idx) => {
          if (!track) return null;
          const trackId = track.track_id ?? idx + 1;
          const isSelected = selectedTrackId === trackId;
          const imageUrl = resolveImageUrl(track.thumbnail_url);
          const hasImageError = imageErrorMap[trackId];
          const firstSec = track.first_sec ?? 0;
          const lastSec = track.last_sec ?? 0;
          const timeSpan = Math.max(0, lastSec - firstSec).toFixed(2);
          const coarseClass = track.coarse_class || 'vehicle';
          const framesSeen = track.frames_seen ?? 0;

          return (
            <div
              key={trackId}
              onClick={() => onSelectTrack(track)}
              className={`group cursor-pointer rounded-xl border p-4 transition-all duration-200 relative flex flex-col justify-between ${
                isSelected
                  ? 'bg-blue-950/40 border-cyan-400 ring-2 ring-cyan-500/30 shadow-lg shadow-cyan-950/50'
                  : 'bg-slate-850/70 border-slate-700/80 hover:border-slate-500 hover:bg-slate-800/80 hover:shadow-md'
              }`}
            >
              {/* Selected Badge */}
              {isSelected && (
                <div className="absolute top-2 right-2 z-10">
                  <span className="px-2 py-0.5 rounded bg-cyan-500 text-slate-950 font-mono text-[10px] font-extrabold tracking-wider uppercase shadow">
                    ACTIVE TARGET
                  </span>
                </div>
              )}

              {/* Thumbnail Container */}
              <div className="w-full h-36 rounded-lg bg-slate-950 border border-slate-800 overflow-hidden relative mb-3.5 flex items-center justify-center">
                {imageUrl && !hasImageError ? (
                  <img
                    src={imageUrl}
                    alt={`Vehicle Track ${trackId}`}
                    onError={() =>
                      setImageErrorMap((prev) => ({ ...prev, [trackId]: true }))
                    }
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center p-4 text-center">
                    <svg className="w-8 h-8 text-slate-600 mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 17a2 2 0 11-4 0 2 2 0 014 0zM19 17a2 2 0 11-4 0 2 2 0 014 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13 16V6a1 1 0 00-1-1H4a1 1 0 00-1 1v10a1 1 0 001 1h1m8-1a1 1 0 01-1 1H9m4-1V8a1 1 0 011-1h2.586a1 1 0 01.707.293l3.414 3.414a1 1 0 01.293.707V16a1 1 0 01-1 1h-1m-6-1a1 1 0 001 1h1M5 17a2 2 0 104 0m-4 0a2 2 0 114 0m6 0a2 2 0 104 0m-4 0a2 2 0 114 0" />
                    </svg>
                    <span className="font-mono text-[10px] text-slate-500 uppercase tracking-wider">
                      TRACK #{trackId} CROP
                    </span>
                  </div>
                )}
                {/* Surveillance time overlay */}
                <div className="absolute bottom-1.5 left-1.5 px-1.5 py-0.5 rounded bg-slate-900/80 backdrop-blur-sm border border-slate-700/60 font-mono text-[10px] text-slate-300">
                  {firstSec.toFixed(1)}s - {lastSec.toFixed(1)}s
                </div>
              </div>

              {/* Track Metadata */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-mono text-xs font-bold text-white tracking-wider">
                    TRACK #{trackId}
                  </span>
                  <span className="font-mono text-[11px] px-2 py-0.5 rounded uppercase font-semibold bg-slate-800 text-cyan-400 border border-slate-700">
                    {coarseClass}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400 border-t border-slate-800/80 pt-2 mb-3">
                  <div>
                    <span className="text-slate-500 block text-[10px]">TIME WINDOW</span>
                    <span>{timeSpan}s span</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">FRAMES SEEN</span>
                    <span>{framesSeen} frames</span>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectTrack(track);
                }}
                disabled={isAnalyzing}
                className={`w-full py-2 px-3 rounded-lg font-mono text-xs font-bold uppercase tracking-wider transition flex items-center justify-center gap-1.5 ${
                  isSelected
                    ? 'bg-cyan-500 text-slate-950 hover:bg-cyan-400'
                    : 'bg-slate-800 text-slate-200 hover:bg-cyan-600 hover:text-white border border-slate-700'
                }`}
              >
                {isSelected && isAnalyzing ? (
                  <>
                    <svg className="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                    </svg>
                    <span>{isSelected ? 'Analyzed' : 'Analyze Track'}</span>
                  </>
                )}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
