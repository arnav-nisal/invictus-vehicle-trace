import React, { useRef } from 'react';
import { DemoScenario, TrackSummary, VideoScanResponse } from '../types';
import { resolveImageUrl } from '../api';
import { TrackSkeleton } from './Skeletons';

interface LeftRailProps {
  scenarios: DemoScenario[];
  selectedScenarioId: string | null;
  onSelectScenario: (scenario: DemoScenario) => void;
  uploadedFile: File | null;
  onFileUpload: (file: File | null) => void;
  knownHint: string;
  onKnownHintChange: (hint: string) => void;
  onScanVideo: () => void;
  isScanning: boolean;
  scanResult: VideoScanResponse | null;
  selectedTrackId: number | null;
  onSelectTrack: (track: TrackSummary) => void;
  isAnalyzing: boolean;
}

export const LeftRail: React.FC<LeftRailProps> = ({
  scenarios = [],
  selectedScenarioId,
  onSelectScenario,
  uploadedFile,
  onFileUpload,
  knownHint = '',
  onKnownHintChange,
  onScanVideo,
  isScanning,
  scanResult,
  selectedTrackId,
  onSelectTrack,
  isAnalyzing,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [imageErrorMap, setImageErrorMap] = React.useState<Record<number, boolean>>({});

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const tracks = Array.isArray(scanResult?.tracks) ? scanResult.tracks : [];
  const durationSec =
    scanResult?.duration_seconds !== undefined && scanResult?.duration_seconds !== null
      ? scanResult.duration_seconds.toFixed(1)
      : '0.0';
  const totalFrames = scanResult?.total_frames ?? 0;

  return (
    <div className="rounded-3xl bg-white border border-[rgba(15,23,42,0.06)] shadow-bento p-6 space-y-6">
      {/* 1. Header & Benchmark Clips Pill Strip */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-900" />
            <h2 className="text-base font-bold font-sans text-slate-900 tracking-snugger">
              CCTV Surveillance Intake & Calibration Clips
            </h2>
          </div>
          <span className="text-xs font-sans text-slate-500 font-medium">
            4 Evaluation Presets • Zero-Hallucination
          </span>
        </div>

        {/* Horizontal Scenario Pills */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-2.5">
          {scenarios.map((scenario, idx) => {
            const isSelected = selectedScenarioId === scenario.id;
            const isBlur = scenario.id === 'scenario-1';
            const isConfusion = scenario.id === 'scenario-2';
            const isPartial = scenario.id === 'scenario-3';
            const isOutOfReg = scenario.id === 'scenario-4';
            const titleClean = (scenario.title || `Scenario ${idx + 1}`).replace(/Scenario \d+: /, '');
            const plate = scenario.evidence?.consensus_plate || 'UNKNOWN';

            return (
              <button
                key={scenario.id || idx}
                type="button"
                onClick={() => onSelectScenario(scenario)}
                className={`p-3 rounded-2xl border text-left transition-all flex flex-col justify-between gap-2 active:scale-[0.98] ${
                  isSelected
                    ? 'border-slate-900 bg-slate-900 text-white shadow-md'
                    : 'border-slate-200/80 bg-slate-50/60 hover:bg-slate-100/80 hover:border-slate-300 text-slate-800'
                }`}
              >
                <div className="flex items-center justify-between gap-1 w-full">
                  <span className="text-xs font-bold font-sans truncate">
                    {titleClean}
                  </span>
                  <span
                    className={`text-[9px] font-sans font-semibold px-2 py-0.5 rounded-full uppercase flex-shrink-0 ${
                      isSelected
                        ? 'bg-white/20 text-white'
                        : isOutOfReg
                        ? 'bg-rose-100 text-rose-700'
                        : isConfusion
                        ? 'bg-amber-100 text-amber-700'
                        : isBlur
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-emerald-100 text-emerald-700'
                    }`}
                  >
                    {isOutOfReg ? 'Out-Reg' : isConfusion ? 'Confusion' : isBlur ? 'Blurry' : 'Multi-Attr'}
                  </span>
                </div>

                <div className="flex items-center justify-between text-[11px] font-mono">
                  <span className={`font-bold ${isSelected ? 'text-amber-300' : 'text-slate-900'}`}>
                    {plate}
                  </span>
                  <span className={`capitalize text-[10px] font-sans ${isSelected ? 'text-slate-300' : 'text-slate-500'}`}>
                    {scenario.evidence?.colour} {scenario.evidence?.body_type}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. Custom Footage Dropzone & Witness Clue in Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
        {/* Dropzone & Scan Button */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-4 text-center cursor-pointer transition-all flex flex-col justify-between gap-2.5 ${
            uploadedFile
              ? 'border-slate-900 bg-slate-50'
              : 'border-slate-200 hover:border-slate-400 bg-slate-50/40 hover:bg-slate-50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/avi,.mp4,.avi"
            onChange={handleFileChange}
            className="hidden"
          />

          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-white shadow-sm border border-slate-200 flex items-center justify-center flex-shrink-0 text-slate-700">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div className="text-left overflow-hidden">
              {uploadedFile ? (
                <>
                  <p className="text-xs font-bold text-slate-900 font-sans truncate">
                    {uploadedFile.name}
                  </p>
                  <p className="text-[11px] text-slate-500 font-mono">
                    {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready
                  </p>
                </>
              ) : (
                <>
                  <p className="text-xs font-semibold text-slate-900 font-sans">
                    Drop custom CCTV clip (.mp4, .avi)
                  </p>
                  <p className="text-[11px] text-slate-500 font-sans">
                    or click to browse local video file
                  </p>
                </>
              )}
            </div>
          </div>

          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onScanVideo();
            }}
            disabled={isScanning}
            className="tactile-button w-full py-2 px-3 rounded-xl bg-slate-900 hover:bg-slate-800 text-white font-sans font-semibold text-xs transition duration-150 flex items-center justify-center gap-2 shadow-sm disabled:opacity-50"
          >
            {isScanning ? (
              <>
                <svg className="animate-spin h-3.5 w-3.5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Scanning Video Pipeline...</span>
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Process CCTV Footage</span>
              </>
            )}
          </button>
        </div>

        {/* Witness Plate Clue Input Card */}
        <div className="border border-slate-200/80 bg-slate-50/50 rounded-2xl p-4 flex flex-col justify-between gap-2.5">
          <div>
            <div className="flex items-center justify-between mb-1">
              <label htmlFor="witness-input" className="text-xs font-bold text-slate-900 font-sans">
                Witness Plate Clue / Partial
              </label>
              {knownHint && (
                <button
                  type="button"
                  onClick={() => onKnownHintChange('')}
                  className="text-[11px] font-sans text-slate-500 hover:text-slate-800"
                >
                  Clear
                </button>
              )}
            </div>
            <p className="text-[11px] text-slate-500 font-sans leading-relaxed">
              Provides Bayesian prior to resolve unreadable/blurred slots (e.g. MH12DE).
            </p>
          </div>

          <div className="relative">
            <input
              id="witness-input"
              type="text"
              placeholder="e.g. MH12DE or ??"
              value={knownHint}
              onChange={(e) => onKnownHintChange(e.target.value.toUpperCase())}
              className="w-full bg-white border border-slate-200 rounded-xl px-3 py-2 font-mono text-sm font-semibold text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-900 tracking-wider shadow-inner"
            />
          </div>
        </div>
      </div>

      {/* 3. Horizontal Suspect Vehicle Tracks */}
      <div className="space-y-3 pt-2 border-t border-slate-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500" />
            <h3 className="text-xs font-bold font-sans uppercase tracking-wider text-slate-900">
              Suspect Vehicle Tracks
            </h3>
            <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 font-mono text-[11px] font-semibold border border-slate-200">
              {tracks.length} Detected
            </span>
          </div>
          <span className="text-[11px] font-mono text-slate-500">
            {durationSec}s duration • {totalFrames} frames
          </span>
        </div>

        {isScanning ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            <TrackSkeleton />
            <TrackSkeleton />
            <TrackSkeleton />
          </div>
        ) : tracks.length === 0 ? (
          <div className="p-6 text-center rounded-2xl bg-slate-50 border border-slate-200/60 text-slate-500 text-xs font-sans">
            No vehicle tracks detected yet. Select a calibration preset above or scan footage.
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {tracks.map((track, idx) => {
              const trackId = track.track_id ?? idx + 1;
              const isSelected = selectedTrackId === trackId;
              const imageUrl = resolveImageUrl(track.thumbnail_url);
              const hasErr = imageErrorMap[trackId];
              const firstSec = track.first_sec ?? 0;
              const lastSec = track.last_sec ?? 0;

              return (
                <div
                  key={trackId}
                  onClick={() => onSelectTrack(track)}
                  className={`rounded-2xl border p-3 transition-all cursor-pointer flex flex-col gap-2.5 active:scale-[0.98] ${
                    isSelected
                      ? 'border-slate-900 bg-slate-900 text-white shadow-md'
                      : 'border-slate-200/80 bg-slate-50/50 hover:bg-white hover:border-slate-300 hover:shadow-sm text-slate-800'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-xs font-bold">
                        Track #{trackId}
                      </span>
                      <span className={`text-[10px] font-sans uppercase font-medium px-2 py-0.5 rounded-full ${
                        isSelected ? 'bg-white/20 text-white' : 'bg-slate-200/70 text-slate-700'
                      }`}>
                        {track.coarse_class || 'Vehicle'}
                      </span>
                    </div>

                    {isSelected && (
                      <span className="text-[9px] font-sans font-bold uppercase px-2 py-0.5 rounded-full bg-emerald-500 text-white">
                        ACTIVE
                      </span>
                    )}
                  </div>

                  {/* Thumbnail */}
                  <div className="w-full h-24 rounded-xl bg-slate-200/70 overflow-hidden relative flex items-center justify-center border border-black/5">
                    {imageUrl && !hasErr ? (
                      <img
                        src={imageUrl}
                        alt={`Track ${trackId}`}
                        onError={() => setImageErrorMap((prev) => ({ ...prev, [trackId]: true }))}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span className="font-mono text-[10px] text-slate-400">
                        CROP #{trackId}
                      </span>
                    )}
                    <div className="absolute bottom-1.5 left-1.5 px-2 py-0.5 rounded-md bg-black/70 backdrop-blur-md text-[9px] font-mono text-white">
                      {firstSec.toFixed(1)}s - {lastSec.toFixed(1)}s
                    </div>
                    <div className="absolute bottom-1.5 right-1.5 px-2 py-0.5 rounded-md bg-black/70 backdrop-blur-md text-[9px] font-mono text-white">
                      {track.frames_seen ?? 0} frames
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
