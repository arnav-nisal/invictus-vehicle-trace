import React, { useState, useEffect, useRef } from 'react';
import { Check, Film, RefreshCw } from 'lucide-react';
import { ClipItem, LocalClipsResponse } from '../types';

interface IntakePageProps {
  selectedClip: ClipItem | null;
  onSelectClip: (clip: ClipItem) => void;
  onContinue: () => void;
  onClipsLoaded?: (count: number) => void;
}

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(1)} MB`;
}

function formatDuration(sec?: number): string {
  if (sec === undefined || isNaN(sec)) return '--.-s';
  return `${sec.toFixed(1)}s`;
}

interface ClipCardProps {
  clip: ClipItem;
  isSelected: boolean;
  onSelect: (clip: ClipItem) => void;
  onMetadataUpdate: (id: string, metadata: Partial<ClipItem>) => void;
}

const ClipCard: React.FC<ClipCardProps> = ({
  clip,
  isSelected,
  onSelect,
  onMetadataUpdate,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const handleLoadedMetadata = () => {
    const el = videoRef.current;
    if (!el) return;

    // Gently seek to 0.5s so initial black frames show active traffic/scene
    if (el.duration > 0.5) {
      el.currentTime = 0.5;
    }

    onMetadataUpdate(clip.id, {
      width: el.videoWidth,
      height: el.videoHeight,
      duration_sec: el.duration,
    });
  };

  const handleMouseEnter = () => {
    const el = videoRef.current;
    if (!el) return;
    el.play()
      .then(() => setIsPlaying(true))
      .catch(() => {});
  };

  const handleMouseLeave = () => {
    const el = videoRef.current;
    if (!el) return;
    el.pause();
    setIsPlaying(false);
    if (el.duration > 0.5) {
      el.currentTime = 0.5;
    }
  };

  return (
    <div
      onClick={() => onSelect(clip)}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect(clip);
        }
      }}
      className={`group relative flex flex-col bg-white rounded-2xl p-2.5 transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] cursor-pointer text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-[#111111] ${
        isSelected
          ? 'ring-2 ring-[#047857] shadow-[0_2px_8px_rgba(4,120,87,0.12)]'
          : 'border border-[#E6E6E2] hover:border-[#111111]/30 hover:shadow-[0_2px_6px_rgba(0,0,0,0.04)]'
      }`}
    >
      {/* 16:9 Live Preview */}
      <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-[#111111] flex items-center justify-center">
        <video
          ref={videoRef}
          src={clip.video_url}
          muted
          playsInline
          preload="metadata"
          onLoadedMetadata={handleLoadedMetadata}
          className="w-full h-full object-cover rounded-xl"
        />

        {/* Selected State Checkmark Indicator */}
        {isSelected && (
          <div className="absolute top-2.5 right-2.5 w-6 h-6 rounded-full bg-[#047857] text-white flex items-center justify-center shadow-md ring-2 ring-white">
            <Check className="w-3.5 h-3.5 stroke-[2.5]" />
          </div>
        )}

        {/* Subtle hover state play badge indicator */}
        <div
          className={`absolute bottom-2 right-2 px-1.5 py-0.5 rounded bg-black/70 backdrop-blur-sm text-[10px] font-mono text-white/90 tabular-nums transition-opacity duration-150 ${
            isPlaying ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
          }`}
        >
          {isPlaying ? 'PLAYING' : 'HOVER PREVIEW'}
        </div>
      </div>

      {/* Metadata Strip: Minimal & Clean (filename, duration, size only) */}
      <div className="pt-2.5 pb-1 px-1 flex flex-col gap-1">
        <div className="flex items-center justify-between gap-2">
          <span
            className="font-mono text-xs font-semibold text-[#111111] truncate tracking-tight"
            title={clip.filename}
          >
            {clip.filename}
          </span>
          <span className="font-mono text-[11px] text-[#6B6B66] shrink-0 tabular-nums">
            {formatBytes(clip.size_bytes)}
          </span>
        </div>

        <div className="flex items-center text-[11px] text-[#6B6B66] font-mono tabular-nums">
          <span>{formatDuration(clip.duration_sec)}</span>
        </div>
      </div>
    </div>
  );
};

export const IntakePage: React.FC<IntakePageProps> = ({
  selectedClip,
  onSelectClip,
  onContinue,
  onClipsLoaded,
}) => {
  const [clips, setClips] = useState<ClipItem[]>([]);
  const [directory, setDirectory] = useState<string>('data/Final');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchClips = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch('/local-clips');
      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}: ${res.statusText}`);
      }
      const data: LocalClipsResponse = await res.json();
      setClips(data.clips || []);
      setDirectory(data.directory || 'data/Final');
      if (onClipsLoaded) {
        onClipsLoaded(data.clips ? data.clips.length : 0);
      }
    } catch (err) {
      setError((err as Error).message || 'Failed to scan footage library');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchClips();
  }, []);

  const handleMetadataUpdate = (id: string, metadata: Partial<ClipItem>) => {
    setClips((prev) =>
      prev.map((c) => (c.id === id ? { ...c, ...metadata } : c))
    );

    if (selectedClip && selectedClip.id === id) {
      onSelectClip({ ...selectedClip, ...metadata });
    }
  };

  return (
    <div className="min-h-[calc(100vh-3.5rem)] bg-[#F7F7F5] pb-28 pt-8 px-6">
      <main className="max-w-5xl mx-auto flex flex-col gap-8">
        {/* Header Section */}
        <div className="flex flex-col gap-1.5 border-b border-[#E6E6E2] pb-6">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[11px] text-[#6B6B66] uppercase tracking-wider">
              Library Source:
            </span>
            <span className="font-mono text-[11px] text-[#111111] bg-[#F1F1EE] px-2 py-0.5 rounded border border-[#E6E6E2]">
              {directory}
            </span>
          </div>

          <h1 className="text-2xl font-semibold text-[#111111] tracking-tighter">
            Welcome to Invictus
          </h1>

          <p className="text-sm text-[#6B6B66] max-w-2xl leading-relaxed">
            Select CCTV footage from your local library to isolate a vehicle and begin verification.
          </p>
        </div>

        {/* Loading Skeletons */}
        {isLoading && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, idx) => (
              <div
                key={idx}
                className="bg-white rounded-2xl p-2.5 border border-[#E6E6E2] flex flex-col gap-2.5 animate-pulse"
              >
                <div className="aspect-video w-full rounded-xl bg-[#F1F1EE]" />
                <div className="flex items-center justify-between px-1">
                  <div className="h-3.5 w-24 bg-[#F1F1EE] rounded" />
                  <div className="h-3 w-12 bg-[#F1F1EE] rounded" />
                </div>
                <div className="h-3 w-16 bg-[#F1F1EE] rounded mx-1" />
              </div>
            ))}
          </div>
        )}

        {/* Error State */}
        {!isLoading && error && (
          <div className="bg-white rounded-2xl border border-red-200 p-8 text-center flex flex-col items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-red-50 text-red-600 flex items-center justify-center">
              <Film className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-semibold text-[#111111]">
              Unable to load clips from {directory}
            </h3>
            <p className="text-xs text-[#6B6B66] max-w-md">{error}</p>
            <button
              onClick={fetchClips}
              className="mt-2 inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-medium bg-[#111111] text-white hover:bg-[#262626] transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Retry Scan
            </button>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && !error && clips.length === 0 && (
          <div className="bg-white rounded-2xl border border-[#E6E6E2] p-12 text-center flex flex-col items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-[#F1F1EE] text-[#6B6B66] flex items-center justify-center">
              <Film className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-[#111111]">
              No video footage found in {directory}
            </h3>
            <p className="text-xs text-[#6B6B66] max-w-md leading-relaxed">
              Place surveillance recordings (.mp4, .mov, or .mkv) in your local{' '}
              <code className="font-mono bg-[#F1F1EE] px-1 py-0.5 rounded text-[#111111]">
                {directory}
              </code>{' '}
              folder to begin target detection and validation.
            </p>
            <button
              onClick={fetchClips}
              className="mt-2 inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-medium bg-[#111111] text-white hover:bg-[#262626] transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Refresh Directory
            </button>
          </div>
        )}

        {/* Footage Grid */}
        {!isLoading && !error && clips.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {clips.map((clip) => (
              <ClipCard
                key={clip.id}
                clip={clip}
                isSelected={selectedClip?.id === clip.id}
                onSelect={onSelectClip}
                onMetadataUpdate={handleMetadataUpdate}
              />
            ))}
          </div>
        )}
      </main>

      {/* Anchored Bottom Action Bar: Filename, Duration, Size Only */}
      <div className="fixed bottom-6 left-0 right-0 z-40 px-6 pointer-events-none">
        <div className="max-w-5xl mx-auto pointer-events-auto bg-white rounded-2xl border border-[#E6E6E2] shadow-[0_4px_24px_rgba(0,0,0,0.06)] px-5 py-3.5 flex items-center justify-between gap-4">
          {/* Left: Clean metadata info */}
          <div className="flex items-center gap-2 min-w-0">
            {selectedClip ? (
              <div className="flex items-center gap-2 truncate">
                <span className="text-xs text-[#6B6B66]">Selected:</span>
                <span className="font-mono text-xs text-[#111111] font-medium truncate">
                  {selectedClip.filename}
                </span>
                <span className="text-[#E6E6E2]">·</span>
                <span className="font-mono text-xs text-[#6B6B66] tabular-nums shrink-0">
                  {formatDuration(selectedClip.duration_sec)}
                </span>
                <span className="text-[#E6E6E2]">·</span>
                <span className="font-mono text-xs text-[#6B6B66] tabular-nums shrink-0">
                  {formatBytes(selectedClip.size_bytes)}
                </span>
              </div>
            ) : (
              <span className="text-xs text-[#6B6B66]">
                Select a footage clip from the library above to proceed
              </span>
            )}
          </div>

          {/* Right: Obsidian Primary CTA */}
          <button
            disabled={!selectedClip}
            onClick={onContinue}
            className={`shrink-0 px-5 py-2.5 rounded-xl text-xs font-medium transition-all duration-150 flex items-center gap-2 select-none ${
              selectedClip
                ? 'bg-[#111111] hover:bg-[#262626] text-white shadow-sm cursor-pointer'
                : 'bg-[#F1F1EE] text-[#6B6B66]/50 cursor-not-allowed border border-[#E6E6E2]'
            }`}
          >
            <span>Continue to Target Selection</span>
            <span className="font-mono text-sm leading-none">→</span>
          </button>
        </div>
      </div>
    </div>
  );
};
