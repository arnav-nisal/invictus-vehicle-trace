import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX } from 'lucide-react';
import { BoxesData, FrameBox } from '../types';

interface InteractiveVideoCanvasProps {
  videoUrl: string;
  boxesData: BoxesData | null;
  selectedTrackIds: number[];
  seekTime?: number | null;
  rightSlot?: React.ReactNode;
}

export const InteractiveVideoCanvas: React.FC<InteractiveVideoCanvasProps> = ({
  videoUrl,
  boxesData,
  selectedTrackIds,
  seekTime,
  rightSlot,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);

  // Handle external seek requests
  useEffect(() => {
    if (seekTime !== null && seekTime !== undefined && videoRef.current) {
      videoRef.current.currentTime = seekTime;
      setCurrentTime(seekTime);
    }
  }, [seekTime]);

  const drawOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = video.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!boxesData) return;

    const fps = boxesData.fps || 30;
    const nativeW = boxesData.width || 1920;
    const nativeH = boxesData.height || 1080;

    const currentFrameIdx = Math.floor(video.currentTime * fps);
    const activeBoxes: FrameBox[] = boxesData.frames[String(currentFrameIdx)] || [];

    const scaleX = canvas.width / nativeW;
    const scaleY = canvas.height / nativeH;

    const hasSelection = selectedTrackIds.length > 0;

    // Helper to draw clean numbered pill
    const drawIdPill = (id: number, x: number, y: number, isSelected: boolean) => {
      const text = `#${id}`;
      ctx.font = 'bold 10px "JetBrains Mono", monospace, monospace';
      const textMetrics = ctx.measureText(text);
      const textWidth = textMetrics.width;
      const pillW = textWidth + 8;
      const pillH = 15;
      const pillX = x;
      const pillY = y >= pillH ? y - pillH : y;

      ctx.save();
      ctx.fillStyle = isSelected ? '#10B981' : 'rgba(17, 17, 17, 0.88)';
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(pillX, pillY, pillW, pillH, 3);
      } else {
        ctx.rect(pillX, pillY, pillW, pillH);
      }
      ctx.fill();

      if (!isSelected) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 0.75;
        ctx.stroke();
      }

      ctx.fillStyle = '#FFFFFF';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, pillX + 4, pillY + pillH / 2 + 0.5);
      ctx.restore();
    };

    activeBoxes.forEach((box) => {
      const [x1, y1, x2, y2] = box.bbox;
      const bx = x1 * scaleX;
      const by = y1 * scaleY;
      const bw = (x2 - x1) * scaleX;
      const bh = (y2 - y1) * scaleY;

      const isSelected = selectedTrackIds.includes(box.track_id);

      ctx.save();

      if (isSelected) {
        // Vivid Deep Emerald #10B981 2.5px stroke + soft fill
        ctx.fillStyle = 'rgba(16, 185, 129, 0.16)';
        ctx.fillRect(bx, by, bw, bh);

        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        ctx.shadowBlur = 4;
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = '#10B981';
        ctx.strokeRect(bx, by, bw, bh);

        drawIdPill(box.track_id, bx, by, true);
      } else {
        // Clean subtle white outline with dark shadow halo
        ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
        ctx.shadowBlur = 2.5;
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = hasSelection
          ? 'rgba(255, 255, 255, 0.35)'
          : 'rgba(255, 255, 255, 0.85)';
        ctx.strokeRect(bx, by, bw, bh);

        drawIdPill(box.track_id, bx, by, false);
      }

      ctx.restore();
    });
  }, [boxesData, selectedTrackIds]);

  // Synchronized animation loop
  useEffect(() => {
    let animId: number;
    const renderLoop = () => {
      drawOverlay();
      animId = requestAnimationFrame(renderLoop);
    };

    animId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animId);
  }, [drawOverlay]);

  // Video metadata and playback events
  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    setDuration(video.duration);
    if (video.duration > 0.5 && currentTime === 0) {
      video.currentTime = 0.5;
      setCurrentTime(0.5);
    }
    drawOverlay();
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
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
    drawOverlay();
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const targetTime = parseFloat(e.target.value);
    const video = videoRef.current;
    if (video) {
      video.currentTime = targetTime;
      setCurrentTime(targetTime);
      drawOverlay();
    }
  };

  const handleSpeedChange = (speed: number) => {
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
      setPlaybackSpeed(speed);
    }
  };

  // Spacebar toggle shortcut (no input focus collision)
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
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms}`;
  };

  const currentFrame = boxesData ? Math.floor(currentTime * (boxesData.fps || 30)) : 0;
  const totalFrames = boxesData?.total_frames || (duration ? Math.floor(duration * 30) : 300);

  return (
    <div
      ref={containerRef}
      className="relative flex-1 min-h-0 flex flex-col bg-white rounded-2xl border border-[#E6E6E2] overflow-hidden shadow-[0_2px_12px_rgba(0,0,0,0.03)]"
    >
      {/* Fullscreen Video Area with Crisp Numbered Canvas (No hit testing / default cursor) */}
      <div className="relative flex-1 min-h-0 w-full bg-[#111111] overflow-hidden flex items-center justify-center select-none">
        <video
          ref={videoRef}
          src={videoUrl}
          muted={isMuted}
          playsInline
          preload="metadata"
          onLoadedMetadata={handleLoadedMetadata}
          onTimeUpdate={handleTimeUpdate}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          className="w-full h-full object-contain"
        />

        {/* Clean Monospace Canvas Overlay with default cursor */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full cursor-default pointer-events-none"
        />

        {/* Quiet Frame Counter Pill (Top-Right) */}
        <div className="absolute top-3 right-3 pointer-events-none">
          <div className="bg-black/70 backdrop-blur-md px-2.5 py-1 rounded-md border border-white/10 text-white font-mono text-[11px] tabular-nums flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#10B981] animate-pulse" />
            <span>Frame {currentFrame} / {totalFrames}</span>
          </div>
        </div>
      </div>

      {/* Slim Integrated Scrubber & Transport Bottom Dock */}
      <div className="px-5 py-2.5 bg-white border-t border-[#E6E6E2] flex flex-col gap-2 shrink-0">
        <div className="relative flex items-center">
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

        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Left: Play/Pause, Scrubber reset, Mute, Timestamp, Speed toggle */}
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
              title="Reset"
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

            <div className="font-mono text-xs text-[#111111] tabular-nums px-1.5">
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

          {/* Right Slot: Inline Track ID Input + Continue Button */}
          {rightSlot && (
            <div className="flex items-center gap-3">
              {rightSlot}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
