import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  ArrowLeft,
  AlertCircle,
  RefreshCw,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  ListFilter,
  X,
  Check,
} from 'lucide-react';
import { ClipItem, Stage1Result, Stage1Track, JobStatus } from '../types';
import { selectLocalVideo, pollStage1Job } from '../services/api';

interface TrackSelectPageProps {
  selectedClip: ClipItem;
  onBackToIntake: () => void;
  onContinueToVerify: (trackIds: number[], stage1Result: Stage1Result) => void;
}

export const TrackSelectPage: React.FC<TrackSelectPageProps> = ({
  selectedClip,
  onBackToIntake,
  onContinueToVerify,
}) => {
  const [jobStatus, setJobStatus] = useState<JobStatus>('processing');
  const [statusMessage, setStatusMessage] = useState<string>('Initializing YOLO tracking pipeline...');
  const [progress, setProgress] = useState<number>(15);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [stage1Result, setStage1Result] = useState<Stage1Result | null>(null);

  // Video playback controls state
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);

  // Track ID input state & tracks popover
  const [trackIdInput, setTrackIdInput] = useState<string>('');
  const [showTracksPopover, setShowTracksPopover] = useState<boolean>(false);

  const activeAbortController = useRef<AbortController | null>(null);

  // Stage 1 live pipeline execution with cancellation guard
  const runStage1Pipeline = async () => {
    if (activeAbortController.current) {
      activeAbortController.current.abort();
    }
    const controller = new AbortController();
    activeAbortController.current = controller;
    const signal = controller.signal;

    setJobStatus('processing');
    setErrorMessage(null);
    setProgress(15);
    setStatusMessage(`Requesting Stage 1 CV processing for ${selectedClip.filename}...`);

    try {
      const initResp = await selectLocalVideo(selectedClip.filename, false, signal);
      if (signal.aborted) return;

      const jobId = initResp.job_id || (initResp as any)?.result?.job_id;
      if (!jobId) {
        throw new Error('No job identifier received from server for Stage 1 processing.');
      }

      let finalResult: Stage1Result;

      if (initResp.status === 'ready' && initResp.result) {
        finalResult = initResp.result;
        setProgress(100);
        setStatusMessage('Stage 1 tracking and web video ready (cached)');
      } else {
        finalResult = await pollStage1Job(
          jobId,
          (statusResp) => {
            if (signal.aborted) return;

            if (statusResp.status === 'error') {
              setJobStatus('error');
            } else if (statusResp.status === 'queued') {
              setJobStatus('queued');
            } else {
              setJobStatus('running');
            }

            const rawProg = Number(statusResp.progress);
            if (!isNaN(rawProg) && isFinite(rawProg)) {
              const normalized = rawProg <= 1 && rawProg > 0 ? rawProg * 100 : rawProg;
              setProgress(Math.min(100, Math.max(0, Math.round(normalized))));
            }

            if (statusResp.message) {
              setStatusMessage(statusResp.message);
            }
          },
          signal
        );
      }

      if (signal.aborted) return;

      // Commit result directly without fetching boxes.json
      setStage1Result(finalResult);
      setProgress(100);
      setStatusMessage('Ready');
      setJobStatus('ready');
    } catch (err: any) {
      if (signal.aborted || err?.name === 'AbortError') {
        return;
      }
      setJobStatus('error');
      setErrorMessage(err?.message || 'Stage 1 processing failed on server.');
    }
  };

  useEffect(() => {
    runStage1Pipeline();

    return () => {
      if (activeAbortController.current) {
        activeAbortController.current.abort();
        activeAbortController.current = null;
      }
    };
  }, [selectedClip.filename]);

  // Video playback callbacks
  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    const dur = Number(video.duration);
    const validDur = isNaN(dur) || !isFinite(dur) ? 0 : dur;
    setDuration(validDur);
    if (validDur > 0.5 && currentTime === 0) {
      video.currentTime = 0.5;
      setCurrentTime(0.5);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const cur = Number(videoRef.current.currentTime);
      if (!isNaN(cur) && isFinite(cur)) {
        setCurrentTime(cur);
      }
    }
  };

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().then(() => setIsPlaying(true)).catch(() => {});
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  const handleReset = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    setCurrentTime(0);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const targetTime = parseFloat(e.target.value);
    const video = videoRef.current;
    if (video && !isNaN(targetTime) && isFinite(targetTime)) {
      video.currentTime = targetTime;
      setCurrentTime(targetTime);
    }
  };

  const handleSpeedChange = (speed: number) => {
    if (videoRef.current && isFinite(speed) && speed > 0) {
      videoRef.current.playbackRate = speed;
      setPlaybackSpeed(speed);
    }
  };

  // Spacebar toggle shortcut (inactive when focused in text input)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeTag = document.activeElement?.tagName?.toLowerCase();
      if (activeTag === 'input' || activeTag === 'textarea') {
        return;
      }
      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const formatTime = (sec: number) => {
    if (isNaN(sec) || !isFinite(sec) || sec < 0) return '00:00.0';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms}`;
  };

  // Parse and validate typed Track ID(s)
  const availableTracks: Stage1Track[] = useMemo(() => {
    if (!stage1Result || !Array.isArray(stage1Result.tracks)) return [];
    return stage1Result.tracks;
  }, [stage1Result]);

  const availableTrackMap = useMemo(() => {
    const map = new Map<number, Stage1Track>();
    for (const t of availableTracks) {
      map.set(Number(t.track_id), t);
    }
    return map;
  }, [availableTracks]);

  const { parsedTrackIds, validationError, trackBadgeInfo } = useMemo<{
    parsedTrackIds: number[];
    validationError: string | null;
    trackBadgeInfo: string | null;
  }>(() => {
    const trimmed = trackIdInput.trim();
    if (!trimmed) {
      return { parsedTrackIds: [], validationError: null, trackBadgeInfo: null };
    }

    const rawParts = trimmed.split(',').map((p) => p.trim()).filter(Boolean);
    const ids: number[] = [];
    const missing: number[] = [];
    const foundTracks: Stage1Track[] = [];

    for (const part of rawParts) {
      const num = parseInt(part, 10);
      if (isNaN(num)) {
        return { parsedTrackIds: [], validationError: `Invalid ID "${part}"`, trackBadgeInfo: null };
      }
      if (!ids.includes(num)) {
        ids.push(num);
      }
      const trk = availableTrackMap.get(num);
      if (!trk) {
        missing.push(num);
      } else {
        foundTracks.push(trk);
      }
    }

    if (missing.length > 0) {
      return {
        parsedTrackIds: ids,
        validationError: missing.length === 1 ? `#${missing[0]} not in clip` : `IDs ${missing.map((m) => `#${m}`).join(', ')} not in clip`,
        trackBadgeInfo: null,
      };
    }

    let badgeText = '';
    if (foundTracks.length === 1) {
      const t = foundTracks[0];
      badgeText = `✓ ${t.category || t.vehicle_class} · ${t.start_sec}s–${t.end_sec}s`;
    } else if (foundTracks.length > 1) {
      badgeText = `✓ ${foundTracks.length} tracks selected (merged)`;
    }

    return { parsedTrackIds: ids, validationError: null, trackBadgeInfo: badgeText };
  }, [trackIdInput, availableTrackMap]);

  const isValid = parsedTrackIds.length > 0 && !validationError;

  const handleContinue = () => {
    if (isValid && stage1Result) {
      onContinueToVerify(parsedTrackIds, stage1Result);
    }
  };

  const handleSelectTrackFromTable = (track: Stage1Track) => {
    setTrackIdInput(String(track.track_id));
    setShowTracksPopover(false);
    if (videoRef.current && track.start_sec !== undefined) {
      videoRef.current.currentTime = Math.max(0, track.start_sec);
      setCurrentTime(Math.max(0, track.start_sec));
    }
  };

  const isReady = jobStatus === 'ready' && !!stage1Result;
  const isError = jobStatus === 'error';
  const isProcessing = !isReady && !isError;

  const trackedVideoUrl = stage1Result?.tracked_video_url || stage1Result?.video_url || selectedClip.video_url;

  return (
    <div className="h-[calc(100vh-56px)] w-full max-w-full overflow-hidden flex flex-col px-5 py-3 gap-3 bg-[#F7F7F5] box-border">
      {/* 1. Slim Header Strip (h-9 shrink-0) */}
      <div className="h-9 shrink-0 flex items-center justify-between gap-3">
        {/* Left: Footage Library back button & clip filename pill */}
        <div className="flex items-center gap-2.5 min-w-0">
          <button
            onClick={onBackToIntake}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-[#E6E6E2] bg-white hover:bg-[#F1F1EE] text-xs font-medium text-[#111111] transition-colors focus:outline-none shadow-sm shrink-0"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Footage Library</span>
          </button>

          <span className="font-mono text-xs text-[#111111] bg-white px-2.5 py-1 rounded-lg border border-[#E6E6E2] shadow-sm truncate">
            {selectedClip.filename}
          </span>
        </div>

        {/* Right: Stage 2 badge, vehicle count & Tracks List Popover */}
        <div className="flex items-center gap-2.5 shrink-0 relative">
          <span className="font-mono text-[11px] text-[#6B6B66] bg-[#F1F1EE] px-2.5 py-1 rounded-md border border-[#E6E6E2]">
            Stage 2 • Target Selection
          </span>

          {availableTracks.length > 0 && (
            <>
              <span className="font-mono text-xs text-[#047857] bg-[#ECFDF5] px-2.5 py-1 rounded-lg border border-[#A7F3D0] flex items-center gap-1.5 shadow-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-[#10B981]" />
                <span>{availableTracks.length} Vehicles Tracked</span>
              </span>

              {/* Tracks List popover toggle button */}
              <div className="relative">
                <button
                  onClick={() => setShowTracksPopover(!showTracksPopover)}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-[#E6E6E2] bg-white hover:bg-[#F1F1EE] text-xs font-medium text-[#111111] transition-colors focus:outline-none shadow-sm"
                >
                  <ListFilter className="w-3.5 h-3.5 text-[#6B6B66]" />
                  <span>Tracks List</span>
                </button>

                {/* Popover Table Card */}
                {showTracksPopover && (
                  <div className="absolute right-0 top-10 w-96 max-w-[90vw] bg-white rounded-2xl border border-[#E6E6E2] shadow-2xl p-3 z-50 flex flex-col gap-2">
                    <div className="flex items-center justify-between pb-1.5 border-b border-[#E6E6E2]">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-xs text-[#111111]">Detected Vehicles</span>
                        <span className="font-mono text-[10px] bg-[#F1F1EE] text-[#6B6B66] px-1.5 py-0.5 rounded">
                          {availableTracks.length}
                        </span>
                      </div>
                      <button
                        onClick={() => setShowTracksPopover(false)}
                        className="p-1 rounded-md text-[#6B6B66] hover:text-[#111111] hover:bg-[#F1F1EE] transition-colors"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <div className="max-h-60 overflow-y-auto pr-1">
                      <table className="w-full text-left text-xs font-mono">
                        <thead>
                          <tr className="text-[10px] text-[#6B6B66] uppercase border-b border-[#E6E6E2]">
                            <th className="py-1 px-1.5">ID</th>
                            <th className="py-1 px-1.5">Class</th>
                            <th className="py-1 px-1.5">Time</th>
                            <th className="py-1 px-1.5 text-right">Frames</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#F1F1EE]">
                          {availableTracks.map((trk) => {
                            const isChosen = parsedTrackIds.includes(trk.track_id);
                            return (
                              <tr
                                key={trk.track_id}
                                onClick={() => handleSelectTrackFromTable(trk)}
                                className={`cursor-pointer transition-colors ${
                                  isChosen
                                    ? 'bg-[#ECFDF5] text-[#047857]'
                                    : 'hover:bg-[#F7F7F5] text-[#111111]'
                                }`}
                              >
                                <td className="py-1.5 px-1.5 font-bold">
                                  #{trk.track_id}
                                </td>
                                <td className="py-1.5 px-1.5 capitalize text-[#6B6B66]">
                                  {trk.category || trk.vehicle_class}
                                </td>
                                <td className="py-1.5 px-1.5 text-[11px] text-[#6B6B66]">
                                  {trk.start_sec}s–{trk.end_sec}s
                                </td>
                                <td className="py-1.5 px-1.5 text-right text-[11px] text-[#6B6B66]">
                                  {trk.frame_count}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    <div className="text-[10px] text-[#6B6B66] pt-1 border-t border-[#E6E6E2]">
                      Click any row to populate Track ID & seek to start frame
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* 2. Processing State Card */}
      {isProcessing && (
        <div className="flex-1 min-h-0 flex items-center justify-center">
          <div className="w-full max-w-lg bg-white rounded-2xl border border-[#E6E6E2] p-8 shadow-[0_2px_12px_rgba(0,0,0,0.03)] flex flex-col gap-6">
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-[#10B981] animate-ping" />
                <span className="font-mono text-[11px] text-[#047857] font-semibold tracking-wider uppercase">
                  Stage 1 Pipeline Active
                </span>
              </div>
              <h2 className="text-xl font-semibold text-[#111111] tracking-tight">
                Analyzing CCTV Footage
              </h2>
              <p className="text-xs text-[#6B6B66] leading-relaxed">
                Executing YOLO tracking and generating H.264 annotated video on{' '}
                <span className="font-mono font-medium text-[#111111]">{selectedClip.filename}</span>.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <div className="w-full h-2 bg-[#F1F1EE] rounded-full overflow-hidden border border-[#E6E6E2]">
                <div
                  className="h-full bg-[#047857] transition-all duration-500 ease-out rounded-full"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="flex justify-between items-center text-[11px] font-mono text-[#6B6B66] tabular-nums">
                <span className="truncate max-w-[320px]">{statusMessage}</span>
                <span className="font-semibold text-[#111111]">{progress}%</span>
              </div>
            </div>

            <div className="pt-2 border-t border-[#E6E6E2] flex items-center justify-between text-[11px] font-mono text-[#6B6B66]">
              <span>Source: {selectedClip.filename}</span>
              <span>Codec: H.264 (yuv420p)</span>
            </div>
          </div>
        </div>
      )}

      {/* 3. Error State Card */}
      {isError && (
        <div className="flex-1 min-h-0 flex items-center justify-center p-6">
          <div className="w-full max-w-lg bg-white rounded-2xl border border-[#E6E6E2] p-8 shadow-[0_2px_12px_rgba(0,0,0,0.03)] flex flex-col items-center text-center gap-5">
            <div className="w-12 h-12 rounded-full bg-red-50 border border-red-100 text-red-600 flex items-center justify-center">
              <AlertCircle className="w-6 h-6" />
            </div>

            <div className="flex flex-col gap-1.5">
              <h2 className="text-lg font-semibold text-[#111111] tracking-tight">Data Error</h2>
              <p className="text-xs text-[#6B6B66] leading-relaxed max-w-sm">
                {errorMessage || 'Unable to execute Stage 1 video tracking pipeline.'}
              </p>
            </div>

            <div className="w-full bg-[#F7F7F5] rounded-xl p-3 text-left font-mono text-xs border border-[#E6E6E2] flex flex-col gap-1 text-[#6B6B66]">
              <div>Endpoint: <code className="text-[#111111]">POST /api/videos/select-local</code></div>
              <div>Source File: <code className="text-[#111111]">{selectedClip.filename}</code></div>
            </div>

            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={onBackToIntake}
                className="px-4 py-2 rounded-xl text-xs font-medium border border-[#E6E6E2] bg-white text-[#111111] hover:bg-[#F1F1EE] transition-colors"
              >
                ← Back
              </button>
              <button
                onClick={runStage1Pipeline}
                className="inline-flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-medium bg-[#111111] hover:bg-[#262626] text-white transition-colors shadow-sm"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 4. Main Video Stage & Clean Bottom Dock */}
      {isReady && stage1Result && (
        <>
          {/* Main Cinema Container holding <video> with baked annotations */}
          <div className="flex-1 min-h-0 w-full relative bg-[#0E0E0D] rounded-2xl overflow-hidden shadow-[0_4px_24px_rgba(0,0,0,0.08)] border border-[#262626]/40 flex items-center justify-center">
            <video
              ref={videoRef}
              src={trackedVideoUrl}
              muted={isMuted}
              playsInline
              preload="auto"
              onLoadedMetadata={handleLoadedMetadata}
              onTimeUpdate={handleTimeUpdate}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
              className="w-full h-full object-contain"
            />

            {/* Quiet Top-Right HUD Badge */}
            <div className="absolute top-3 right-3 pointer-events-none">
              <div className="bg-black/60 backdrop-blur-md px-2.5 py-1 rounded-md border border-white/10 text-white font-mono text-[11px] tabular-nums flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#10B981] animate-pulse" />
                <span>STAGE 1 ANNOTATED · {stage1Result.fps || 30} FPS</span>
              </div>
            </div>
          </div>

          {/* 5. Clean Bottom Control & Track ID Dock (shrink-0) */}
          <div className="shrink-0 bg-white rounded-2xl border border-[#E6E6E2] px-4 py-2.5 shadow-[0_2px_12px_rgba(0,0,0,0.03)] flex flex-col gap-2">
            {/* Timeline Scrubber */}
            <div className="relative flex items-center w-full">
              <input
                type="range"
                min={0}
                max={duration || 10}
                step={0.033}
                value={currentTime}
                onChange={handleSeek}
                aria-label="Video scrubber"
                className="w-full h-1.5 bg-[#F1F1EE] rounded-lg appearance-none cursor-pointer accent-[#111111] focus:outline-none"
              />
            </div>

            {/* Transport controls (left) and Track ID input dock (right) */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              {/* Left Transport: Play/Pause, Reset, Mute, Timestamp, Speed buttons */}
              <div className="flex items-center gap-2.5">
                <button
                  onClick={togglePlay}
                  className="p-1.5 rounded-lg bg-[#111111] hover:bg-[#262626] text-white transition-colors flex items-center justify-center focus:outline-none"
                  title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
                >
                  {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                </button>

                <button
                  onClick={handleReset}
                  className="p-1.5 rounded-lg border border-[#E6E6E2] bg-white hover:bg-[#F1F1EE] text-[#111111] transition-colors focus:outline-none"
                  title="Reset to 00:00"
                >
                  <RotateCcw className="w-3 h-3" />
                </button>

                <button
                  onClick={() => setIsMuted(!isMuted)}
                  className="p-1.5 rounded-lg border border-[#E6E6E2] bg-white hover:bg-[#F1F1EE] text-[#111111] transition-colors focus:outline-none"
                  title={isMuted ? 'Unmute' : 'Mute'}
                >
                  {isMuted ? <VolumeX className="w-3 h-3" /> : <Volume2 className="w-3 h-3" />}
                </button>

                <div className="font-mono text-xs text-[#111111] tabular-nums px-1">
                  <span>{formatTime(currentTime)}</span>
                  <span className="text-[#6B6B66] mx-1">/</span>
                  <span className="text-[#6B6B66]">{formatTime(duration)}</span>
                </div>

                <div className="flex items-center gap-1 bg-[#F1F1EE] p-0.5 rounded-md border border-[#E6E6E2] text-[11px] font-mono">
                  {[0.5, 1.0].map((rate) => (
                    <button
                      key={rate}
                      onClick={() => handleSpeedChange(rate)}
                      className={`px-1.5 py-0.5 rounded font-medium transition-colors ${
                        playbackSpeed === rate
                          ? 'bg-white text-[#111111] shadow-sm'
                          : 'text-[#6B6B66] hover:text-[#111111]'
                      }`}
                    >
                      {rate}x
                    </button>
                  ))}
                </div>
              </div>

              {/* Right Dock: Track ID Label + Bordered Monospace Input + Badges + Run Verification */}
              <div className="flex items-center gap-2.5">
                {/* Instant subtle green badge for valid track or red error hint */}
                {trackBadgeInfo && (
                  <span className="text-[11px] font-mono text-[#047857] bg-[#ECFDF5] px-2 py-1 rounded-md border border-[#A7F3D0] flex items-center gap-1 shadow-sm">
                    <Check className="w-3 h-3" />
                    <span>{trackBadgeInfo}</span>
                  </span>
                )}
                {validationError && (
                  <span className="text-[11px] font-mono text-red-700 bg-red-50 px-2 py-1 rounded-md border border-red-200 shadow-sm">
                    {validationError}
                  </span>
                )}

                {/* Inline Track ID Label and Input */}
                <div className="flex items-center gap-2">
                  <label htmlFor="track-id-input" className="text-xs font-mono font-semibold text-[#111111] select-none">
                    Track ID:
                  </label>
                  <input
                    id="track-id-input"
                    type="text"
                    value={trackIdInput}
                    onChange={(e) => setTrackIdInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && isValid) {
                        e.preventDefault();
                        handleContinue();
                      }
                    }}
                    placeholder="e.g. 20 or 20, 33"
                    className="w-40 sm:w-48 px-3 py-1.5 rounded-xl border border-[#E6E6E2] bg-[#F7F7F5] focus:bg-white focus:border-[#111111] text-xs font-mono text-[#111111] placeholder:text-[#A1A1AA] focus:outline-none transition-colors"
                  />
                </div>

                {/* Solid Obsidian Run Verification Button */}
                <button
                  disabled={!isValid}
                  onClick={handleContinue}
                  className={`px-4 py-2 rounded-xl text-xs font-medium transition-all flex items-center gap-1.5 select-none ${
                    isValid
                      ? 'bg-[#111111] hover:bg-[#262626] text-white shadow-sm cursor-pointer'
                      : 'bg-[#F1F1EE] text-[#A1A1AA] cursor-not-allowed border border-[#E6E6E2]'
                  }`}
                >
                  <span>Run Verification</span>
                  <span className="font-mono text-xs leading-none">→</span>
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
