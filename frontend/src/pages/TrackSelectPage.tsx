import React, { useState, useEffect, useMemo } from 'react';
import { ArrowLeft, AlertCircle, RefreshCw } from 'lucide-react';
import { ClipItem, Stage1Result, BoxesData } from '../types';
import { InteractiveVideoCanvas } from '../components/InteractiveVideoCanvas';
import { selectLocalVideo, pollStage1Job, fetchBoxesData } from '../services/api';

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
  const [jobStatus, setJobStatus] = useState<'queued' | 'processing' | 'ready' | 'error'>('processing');
  const [statusMessage, setStatusMessage] = useState<string>('Initializing YOLO tracking pipeline...');
  const [progress, setProgress] = useState<number>(0.15);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [stage1Result, setStage1Result] = useState<Stage1Result | null>(null);
  const [boxesData, setBoxesData] = useState<BoxesData | null>(null);

  // Track ID input state
  const [trackIdInput, setTrackIdInput] = useState<string>('');

  // Stage 1 live pipeline execution
  const runStage1Pipeline = async () => {
    setJobStatus('processing');
    setErrorMessage(null);
    setProgress(0.15);
    setStatusMessage(`Requesting Stage 1 CV processing for ${selectedClip.filename}...`);

    try {
      const initResp = await selectLocalVideo(selectedClip.filename, false);

      let finalResult: Stage1Result;

      if (initResp.status === 'ready' && initResp.result) {
        finalResult = initResp.result;
      } else {
        finalResult = await pollStage1Job(initResp.job_id, (statusResp) => {
          setJobStatus(statusResp.status);
          if (statusResp.progress !== undefined) {
            setProgress(statusResp.progress);
          }
          if (statusResp.message) {
            setStatusMessage(statusResp.message);
          }
        });
      }

      setStage1Result(finalResult);

      // Fetch live bounding boxes from FastAPI backend
      const boxes = await fetchBoxesData(finalResult.boxes_url);
      setBoxesData(boxes);

      setJobStatus('ready');
      setProgress(1.0);
    } catch (err) {
      setJobStatus('error');
      setErrorMessage((err as Error).message || 'Stage 1 processing failed on server.');
    }
  };

  useEffect(() => {
    runStage1Pipeline();
  }, [selectedClip.filename]);

  // Parse and validate typed Track ID(s)
  const availableTrackIds = useMemo(() => {
    if (!stage1Result || !stage1Result.tracks) return new Set<number>();
    return new Set<number>(stage1Result.tracks.map((t) => t.track_id));
  }, [stage1Result]);

  const { parsedTrackIds, validationError } = useMemo(() => {
    const trimmed = trackIdInput.trim();
    if (!trimmed) {
      return { parsedTrackIds: [], validationError: null };
    }

    const rawParts = trimmed.split(',').map((p) => p.trim()).filter(Boolean);
    const ids: number[] = [];
    const missing: number[] = [];

    for (const part of rawParts) {
      const num = parseInt(part, 10);
      if (isNaN(num)) {
        return { parsedTrackIds: [], validationError: `Invalid number "${part}"` };
      }
      if (!ids.includes(num)) {
        ids.push(num);
      }
      if (availableTrackIds.size > 0 && !availableTrackIds.has(num)) {
        missing.push(num);
      }
    }

    if (missing.length > 0) {
      return {
        parsedTrackIds: ids,
        validationError: missing.length === 1 ? `ID #${missing[0]} not in clip` : `IDs ${missing.map(m => `#${m}`).join(', ')} not in clip`,
      };
    }

    return { parsedTrackIds: ids, validationError: null };
  }, [trackIdInput, availableTrackIds]);

  const isValid = parsedTrackIds.length > 0 && !validationError;

  const handleContinue = () => {
    if (isValid && stage1Result) {
      onContinueToVerify(parsedTrackIds, stage1Result);
    }
  };

  return (
    <div className="h-[calc(100vh-3.5rem)] flex flex-col p-4 bg-[#F7F7F5] overflow-hidden">
      {/* Top Status & Back Bar */}
      <div className="flex items-center justify-between pb-3 shrink-0">
        <button
          onClick={onBackToIntake}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-xl border border-[#E6E6E2] bg-white hover:bg-[#F1F1EE] text-xs font-medium text-[#111111] transition-colors focus:outline-none"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Footage Library</span>
        </button>

        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-[#6B6B66] bg-[#F1F1EE] px-2.5 py-1 rounded-md border border-[#E6E6E2]">
            Stage 2 • Target Selection
          </span>
          <span className="font-mono text-xs text-[#111111] bg-white px-2.5 py-1 rounded-md border border-[#E6E6E2]">
            {selectedClip.filename}
          </span>
        </div>
      </div>

      {/* Processing State */}
      {(jobStatus === 'queued' || jobStatus === 'processing') && (
        <div className="flex-1 flex items-center justify-center">
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
                Executing YOLO tracking and BoT-SORT trajectory analysis on{' '}
                <span className="font-mono font-medium text-[#111111]">{selectedClip.filename}</span>.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <div className="w-full h-2 bg-[#F1F1EE] rounded-full overflow-hidden border border-[#E6E6E2]">
                <div
                  className="h-full bg-[#047857] transition-all duration-500 ease-out rounded-full"
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>
              <div className="flex justify-between items-center text-[11px] font-mono text-[#6B6B66] tabular-nums">
                <span className="truncate max-w-[320px]">{statusMessage}</span>
                <span className="font-semibold text-[#111111]">{Math.round(progress * 100)}%</span>
              </div>
            </div>

            <div className="pt-2 border-t border-[#E6E6E2] flex items-center justify-between text-[11px] font-mono text-[#6B6B66]">
              <span>Source: {selectedClip.filename}</span>
              <span>Polling: 700ms</span>
            </div>
          </div>
        </div>
      )}

      {/* Honest Data Error State */}
      {jobStatus === 'error' && (
        <div className="flex-1 flex items-center justify-center p-6">
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

      {/* True Near-Fullscreen Video Canvas & Slim Bottom Dock */}
      {jobStatus === 'ready' && stage1Result && (
        <div className="flex-1 min-h-0 flex flex-col">
          <InteractiveVideoCanvas
            videoUrl={stage1Result.video_url || selectedClip.video_url}
            boxesData={boxesData}
            selectedTrackIds={parsedTrackIds}
            rightSlot={
              <div className="flex items-center gap-3">
                {/* Subtle Inline Validation Badge */}
                {validationError && (
                  <span className="text-[11px] font-mono text-amber-600 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                    {validationError}
                  </span>
                )}

                {/* Inline Track ID Label & Input */}
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

                {/* Solid Obsidian Continue Button */}
                <button
                  disabled={!isValid}
                  onClick={handleContinue}
                  className={`px-4 py-2 rounded-xl text-xs font-medium transition-all flex items-center gap-1.5 select-none ${
                    isValid
                      ? 'bg-[#111111] hover:bg-[#262626] text-white shadow-sm cursor-pointer'
                      : 'bg-[#F1F1EE] text-[#A1A1AA] cursor-not-allowed border border-[#E6E6E2]'
                  }`}
                >
                  <span>Continue</span>
                  <span className="font-mono text-xs leading-none">→</span>
                </button>
              </div>
            }
          />
        </div>
      )}
    </div>
  );
};
