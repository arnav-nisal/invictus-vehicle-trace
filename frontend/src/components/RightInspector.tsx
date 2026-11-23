import React, { useEffect, useState } from 'react';
import { AnalysisResponse, MatchCandidate, SightingRecord } from '../types';
import { exportReportHtml, getSightings } from '../api';
import { CandidateCardSkeleton } from './Skeletons';

interface RightInspectorProps {
  analysis: AnalysisResponse | null;
  isAnalyzing: boolean;
  onOpenPreviewModal: (html: string) => void;
}

// CCTV OCR visual confusion pairs
const CONFUSION_PAIRS: Set<string> = new Set([
  '8-B', 'B-8',
  '0-O', 'O-0',
  '0-D', 'D-0',
  '1-I', 'I-1',
  '5-S', 'S-5',
  '2-Z', 'Z-2',
  '6-G', 'G-6',
  '4-A', 'A-4',
]);

interface CharDiff {
  char: string;
  obsChar: string;
  type: 'exact' | 'wildcard' | 'confusion' | 'mismatch';
  isFromBackend: boolean;
}

/** Parse backend character_diff string like "M H 1 2 [?->D] [?->E] 1 4 3 2" into chip data. */
function parseBackendCharDiff(diffStr: string): CharDiff[] {
  const tokens = diffStr.split(/\s+/);
  return tokens.map((tok) => {
    const wildcard = tok.match(/^\[(.+)->(.+)\]$/);
    if (wildcard) {
      const obs = wildcard[1];
      const cand = wildcard[2];
      const pairKey = `${obs}-${cand}`;
      return {
        char: cand,
        obsChar: obs,
        type: (obs === '?') ? 'wildcard' : CONFUSION_PAIRS.has(pairKey) ? 'confusion' : 'mismatch',
        isFromBackend: true,
      };
    }
    return { char: tok, obsChar: tok, type: 'exact' as const, isFromBackend: true };
  });
}

function computeCharDiffs(observed?: string | null, candidate?: string | null): CharDiff[] {
  const obsStr = observed || '';
  const candStr = candidate || '';
  const diffs: CharDiff[] = [];
  const maxLen = Math.max(obsStr.length, candStr.length);

  for (let i = 0; i < maxLen; i++) {
    const obs = obsStr[i] || '';
    const cand = candStr[i] || '';

    if (obs === '?') {
      diffs.push({ char: cand || '?', obsChar: obs, type: 'wildcard', isFromBackend: false });
    } else if (obs === cand) {
      diffs.push({ char: cand, obsChar: obs, type: 'exact', isFromBackend: false });
    } else if (CONFUSION_PAIRS.has(`${obs}-${cand}`)) {
      diffs.push({ char: cand, obsChar: obs, type: 'confusion', isFromBackend: false });
    } else {
      diffs.push({ char: cand || '-', obsChar: obs, type: 'mismatch', isFromBackend: false });
    }
  }

  return diffs;
}

// SVG Confidence Ring Component
interface ConfidenceRingProps {
  score: number;
  size?: number;
  strokeWidth?: number;
  isGatedOut?: boolean;
}

const ConfidenceRing: React.FC<ConfidenceRingProps> = ({
  score,
  size = 52,
  strokeWidth = 4,
  isGatedOut = false,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clampedScore = Math.max(0, Math.min(1, score));
  const strokeDashoffset = circumference - clampedScore * circumference;
  const pctText = (clampedScore * 100).toFixed(1);

  const strokeColor = isGatedOut
    ? '#EF4444'
    : clampedScore >= 0.75
    ? '#10B981'
    : clampedScore >= 0.65
    ? '#0284C7'
    : '#F59E0B';

  return (
    <div className="relative inline-flex items-center justify-center flex-shrink-0" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={radius} stroke="#E2E8F0" strokeWidth={strokeWidth} fill="transparent" />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          stroke={strokeColor} strokeWidth={strokeWidth} fill="transparent"
          strokeDasharray={circumference} strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-500 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="font-mono font-bold text-[11px] text-slate-900 leading-none">{pctText}%</span>
      </div>
    </div>
  );
};

// Score bar pill row
function ScoreBarRow({
  label, score, colorClass,
}: { label: string; score: number | null | undefined; colorClass: string }) {
  if (score == null) return null;
  const pct = (Math.max(0, Math.min(1, score)) * 100).toFixed(1);
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-slate-500 font-medium">{label}</span>
        <span className={`font-mono font-bold ${colorClass}`}>{pct}%</span>
      </div>
      <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-500 ${colorClass.replace('text-', 'bg-').replace('-600', '-500').replace('-800', '-500')}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// Sighting Timeline component
function SightingTimeline({ plateNumber }: { plateNumber: string }) {
  const [sightings, setSightings] = useState<SightingRecord[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!plateNumber) return;
    setLoading(true);
    setError(null);
    getSightings(plateNumber)
      .then((data) => {
        setSightings(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err?.message || 'Failed to load sightings');
        setLoading(false);
      });
  }, [plateNumber]);

  if (loading) {
    return (
      <div className="pt-2 border-t border-slate-100">
        <span className="text-[10px] font-sans font-bold uppercase tracking-wider text-slate-400 block mb-1">Multi-Camera Sighting Trail</span>
        <div className="flex gap-2 items-center text-xs text-slate-400 font-sans">
          <span className="w-3 h-3 rounded-full bg-slate-200 animate-pulse flex-shrink-0" />
          Loading sighting timeline…
        </div>
      </div>
    );
  }

  if (error || !sightings || sightings.length === 0) {
    return (
      <div className="pt-2 border-t border-slate-100">
        <span className="text-[10px] font-sans font-bold uppercase tracking-wider text-slate-400 block mb-1">Multi-Camera Sighting Trail</span>
        <p className="text-xs text-slate-400 font-sans italic">
          {error ? `Error: ${error}` : 'No prior sightings recorded for this plate.'}
        </p>
      </div>
    );
  }

  return (
    <div className="pt-2 border-t border-slate-100 space-y-2">
      <span className="text-[10px] font-sans font-bold uppercase tracking-wider text-slate-400 block">
        Multi-Camera Sighting Trail ({sightings.length})
      </span>
      <ol className="relative border-l border-slate-200 ml-2 space-y-3">
        {sightings.map((s, i) => (
          <li key={s.sighting_id} className="ml-3">
            <span className="absolute -left-[5px] mt-1 w-2.5 h-2.5 rounded-full bg-sky-400 border-2 border-white shadow-sm" />
            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
              <span className="font-mono text-[11px] font-bold text-sky-700">
                {s.camera_id}
              </span>
              <span className="text-[10px] text-slate-400 font-sans">
                {s.timestamp}
              </span>
              <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-md border ${
                s.confidence >= 0.85 ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                s.confidence >= 0.65 ? 'bg-amber-50 text-amber-700 border-amber-200' :
                'bg-slate-100 text-slate-500 border-slate-200'
              }`}>
                {(s.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

// Anomaly alert banner
function AnomalyBanner({ candidate }: { candidate: MatchCandidate }) {
  const hasCloned = candidate.anomaly_flags?.includes('SUSPECTED_CLONED_PLATE');
  const isStolen = candidate.watchlist_status === 'STOLEN';
  const isFlagged = candidate.watchlist_status === 'FLAGGED';

  if (!hasCloned && !isStolen && !isFlagged) return null;

  return (
    <div className="space-y-1.5">
      {(hasCloned || isStolen) && (
        <div className="flex items-start gap-2 rounded-xl border border-rose-300 bg-rose-50/90 px-3 py-2.5 shadow-sm">
          <span className="flex-shrink-0 w-5 h-5 rounded-full bg-rose-600 text-white flex items-center justify-center font-bold text-xs mt-0.5 animate-pulse">!</span>
          <div className="flex-1 min-w-0">
            {isStolen && (
              <p className="text-xs font-bold text-rose-900 uppercase tracking-wider">
                ⚠️ STOLEN VEHICLE ALERT
              </p>
            )}
            {hasCloned && (
              <p className={`text-xs font-bold text-rose-900 uppercase tracking-wider ${isStolen ? 'mt-0.5' : ''}`}>
                🔴 SUSPECTED CLONED PLATE
              </p>
            )}
            <p className="text-[11px] text-rose-700 font-sans leading-snug mt-0.5">
              {hasCloned
                ? 'Plate score ≥ 95% but VLM attributes (body type / colour) contradict registry. Physical inspection required.'
                : 'This vehicle is listed as stolen. Immediate law enforcement notification advised.'}
            </p>
          </div>
        </div>
      )}
      {isFlagged && !isStolen && (
        <div className="flex items-start gap-2 rounded-xl border border-amber-300 bg-amber-50/90 px-3 py-2.5 shadow-sm">
          <span className="flex-shrink-0 w-5 h-5 rounded-full bg-amber-500 text-white flex items-center justify-center font-bold text-xs mt-0.5">!</span>
          <div>
            <p className="text-xs font-bold text-amber-900 uppercase tracking-wider">🚩 FLAGGED VEHICLE</p>
            <p className="text-[11px] text-amber-700 font-sans leading-snug mt-0.5">
              This vehicle is on the watchlist. Exercise caution and cross-reference with issuing authority.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export const RightInspector: React.FC<RightInspectorProps> = ({
  analysis,
  isAnalyzing,
  onOpenPreviewModal,
}) => {
  const [isExporting, setIsExporting] = useState(false);
  // Track which candidate card is expanded for sighting timeline
  const [expandedPlate, setExpandedPlate] = useState<string | null>(null);

  // Auto-expand rank-1 for sightings whenever analysis changes
  useEffect(() => {
    if (analysis?.top_matches && analysis.top_matches.length > 0 && !analysis.no_confident_match) {
      const rank1 = analysis.top_matches.find((m) => m.rank === 1);
      if (rank1) setExpandedPlate(rank1.plate_number);
    } else {
      setExpandedPlate(null);
    }
  }, [analysis]);

  if (isAnalyzing && !analysis) {
    return (
      <div className="rounded-3xl bg-white border border-[rgba(15,23,42,0.06)] shadow-bento p-6 space-y-4">
        <div className="h-20 bg-slate-100 rounded-2xl animate-pulse" />
        <CandidateCardSkeleton />
        <CandidateCardSkeleton />
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="rounded-3xl bg-white border border-[rgba(15,23,42,0.06)] shadow-bento p-12 text-center text-slate-400 font-sans">
        <p className="text-sm font-medium">No candidate shortlist loaded.</p>
      </div>
    );
  }

  const topMatches = Array.isArray(analysis.top_matches) ? analysis.top_matches : [];
  const noConfidentMatch = analysis.no_confident_match ?? false;
  const explanation = analysis.explanation || '';
  const observedPlate = analysis.evidence?.consensus_plate || '';

  const handleDownloadHtml = async () => {
    try {
      setIsExporting(true);
      const htmlText = await exportReportHtml(analysis);
      const blob = new Blob([htmlText], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `invictus-forensic-report-track-${analysis.track_id}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to export HTML report:', err);
      alert('Failed to generate report export. Please check backend connectivity.');
    } finally {
      setIsExporting(false);
    }
  };

  const handlePrintPdf = async () => {
    try {
      setIsExporting(true);
      const htmlText = await exportReportHtml(analysis);
      const printWindow = window.open('', '_blank');
      if (printWindow) {
        printWindow.document.open();
        printWindow.document.write(htmlText);
        printWindow.document.close();
        setTimeout(() => { printWindow.print(); }, 300);
      } else {
        alert('Popup blocked. Please allow popups to open the printable report.');
      }
    } catch (err) {
      console.error('Failed to open printable report:', err);
      alert('Failed to generate printable report.');
    } finally {
      setIsExporting(false);
    }
  };

  const handlePreviewReport = async () => {
    try {
      setIsExporting(true);
      const htmlText = await exportReportHtml(analysis);
      onOpenPreviewModal(htmlText);
    } catch (err) {
      console.error('Failed to load report preview:', err);
      alert('Failed to load report preview.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="rounded-3xl bg-white border border-[rgba(15,23,42,0.06)] shadow-bento p-6 space-y-6">
      {/* 1. Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-900" />
            <h2 className="text-base font-bold font-sans text-slate-900 tracking-snugger">
              Registry Shortlist &amp; Probabilistic Match
            </h2>
          </div>
          <p className="text-xs text-slate-500 font-sans mt-0.5">
            Bayesian ranking against 1,020 state vehicle records
          </p>
        </div>
        <span className="px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-mono text-xs font-semibold border border-slate-200">
          Threshold: 65.0%
        </span>
      </div>

      {/* 2. Authoritative Security Verdict Banner */}
      {noConfidentMatch ? (
        <div className="rounded-2xl border border-rose-200 bg-rose-50/80 p-4 space-y-2 shadow-sm">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-full bg-rose-600 text-white flex items-center justify-center font-bold text-xs flex-shrink-0">!</div>
            <span className="font-sans text-xs font-bold uppercase tracking-wider text-rose-900">
              Security Gate Triggered • No Confident Match
            </span>
          </div>
          <p className="text-xs text-rose-800 font-sans leading-relaxed">
            {explanation || 'No vehicle in registry achieved the minimum 65.0% composite threshold. Physical attributes contradict database records.'}
          </p>
          <div className="text-[11px] font-sans text-rose-700 pt-2 border-t border-rose-200/60 font-medium">
            Audit Notice: Candidates below failed threshold and are displayed strictly for secondary forensic review.
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50/80 p-4 space-y-1.5 shadow-sm">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50" />
            <span className="font-sans text-xs font-bold uppercase tracking-wider text-emerald-900">
              Confirmed High-Confidence Match (Rank #1)
            </span>
          </div>
          <p className="text-xs text-emerald-800 font-sans leading-relaxed">
            {explanation || 'Suspect resolved via multi-frame consensus OCR, colorimetric verification, and confusion matrix tolerance.'}
          </p>
        </div>
      )}

      {/* 2a. VLM Outage / Fallback Warning */}
      {analysis.warning && (
        <div className="rounded-2xl border border-amber-300 bg-amber-50/90 p-3.5 space-y-1.5 shadow-sm">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-600 animate-pulse flex-shrink-0" />
            <span className="font-sans text-xs font-bold uppercase tracking-wider text-amber-900">
              VLM Outage • Local Fallback Active
            </span>
          </div>
          <p className="text-xs text-amber-800 font-sans leading-relaxed">{analysis.warning}</p>
        </div>
      )}

      {/* 2b. Target Vehicle Evidence Badges */}
      <div className="rounded-2xl bg-slate-50 border border-slate-200/80 p-3 flex flex-wrap items-center gap-2 text-xs font-sans">
        <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Target Evidence:</span>
        <span className="px-2 py-0.5 rounded-md bg-white border border-slate-200 font-mono font-bold text-slate-800">
          {observedPlate || '—'}
        </span>
        <span className="px-2 py-0.5 rounded-md bg-white border border-slate-200 text-slate-700 capitalize font-medium">
          {analysis.evidence?.colour || 'unknown'}
        </span>
        <span className="px-2 py-0.5 rounded-md bg-white border border-slate-200 text-slate-700 uppercase font-medium">
          {analysis.evidence?.body_type || 'vehicle'}
        </span>
        {analysis.evidence?.make && analysis.evidence.make.toLowerCase() !== 'unknown' && (
          <span className="px-2.5 py-0.5 rounded-md bg-indigo-50 border border-indigo-200 text-indigo-700 font-semibold flex items-center gap-1">
            <span className="text-[10px] text-indigo-400 font-normal">Make:</span>
            {analysis.evidence.make}
          </span>
        )}
        {analysis.evidence?.model && analysis.evidence.model.toLowerCase() !== 'unknown' && (
          <span className="px-2.5 py-0.5 rounded-md bg-amber-50 border border-amber-200 text-amber-700 font-semibold flex items-center gap-1">
            <span className="text-[10px] text-amber-500 font-normal">Model:</span>
            {analysis.evidence.model}
          </span>
        )}
        {analysis.evidence?.plate_detected === false && (
          <span className="px-2 py-0.5 rounded-md bg-slate-200 border border-slate-300 text-slate-600 font-semibold text-[10px] uppercase">
            No Plate (Visual Only)
          </span>
        )}
      </div>

      {/* 3. Diff Legend */}
      <div className="flex flex-wrap items-center gap-2 p-2.5 rounded-2xl bg-slate-50 border border-slate-200/80 text-[11px] font-sans">
        <span className="text-slate-500 font-semibold mr-1">Char Diff:</span>
        <span className="px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-800 font-medium border border-emerald-200">Exact</span>
        <span className="px-2 py-0.5 rounded-md bg-amber-100 text-amber-800 font-medium border border-amber-200">8↔B Tolerance</span>
        <span className="px-2 py-0.5 rounded-md bg-amber-50 text-amber-700 font-medium border border-amber-300">Wildcard ?</span>
        <span className="px-2 py-0.5 rounded-md bg-rose-100 text-rose-700 font-medium border border-rose-200">Mismatch</span>
      </div>

      {/* 4. Top-5 Ranked Match Cards */}
      <div className="space-y-4">
        {topMatches.map((cand, idx) => {
          const rankNum = cand.rank ?? idx + 1;
          const isRank1 = rankNum === 1 && !noConfidentMatch;
          const isExpanded = expandedPlate === cand.plate_number;
          const isCloned = cand.anomaly_flags?.includes('SUSPECTED_CLONED_PLATE');
          const isStolen = cand.watchlist_status === 'STOLEN';
          const isFlagged = cand.watchlist_status === 'FLAGGED';

          // Use backend character_diff if available, otherwise compute from observed/candidate plates
          const diffs = cand.character_diff
            ? parseBackendCharDiff(cand.character_diff)
            : computeCharDiffs(observedPlate, cand.plate_number);

          const composite = cand.total_confidence ?? cand.composite_score ?? 0;
          const reasons = Array.isArray(cand.match_reasons) ? cand.match_reasons : [];

          // Classic score bars from score_breakdown
          const plateScorePct = (((cand.score_breakdown?.plate_score) ?? 0) * 100).toFixed(1);
          const colourScorePct = (((cand.score_breakdown?.colour_score) ?? 0) * 100).toFixed(1);
          const bodyScorePct = (((cand.score_breakdown?.body_type_score) ?? 0) * 100).toFixed(1);

          // Forensic sub-scores (new)
          const topLevelPlate = cand.plate_score;
          const topLevelRto = cand.rto_prior_score;
          const topLevelVisual = cand.visual_score;
          const hasForensicScores = topLevelPlate != null || topLevelRto != null || topLevelVisual != null;

          // Border class: cloned plate = rose, stolen = rose, flagged = amber, rank1 = slate-900
          const borderClass = (isCloned || isStolen)
            ? 'border-rose-400 bg-rose-50/30 shadow-md ring-1 ring-rose-300'
            : isFlagged
            ? 'border-amber-400 bg-amber-50/30 shadow-md ring-1 ring-amber-300'
            : isRank1
            ? 'border-slate-900 bg-white shadow-bento ring-1 ring-slate-900/10'
            : 'border-slate-200 bg-white hover:border-slate-300 shadow-sm';

          return (
            <div
              key={cand.plate_number || idx}
              className={`rounded-2xl border p-4 space-y-3.5 transition-all ${borderClass}`}
            >
              {/* Card Header: Rank, Plate, Confidence Ring */}
              <div className="flex items-center justify-between pb-3 border-b border-slate-100 gap-3">
                <div className="flex flex-col gap-1.5 flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`text-[10px] font-sans font-bold px-2 py-0.5 rounded-full uppercase ${
                      isRank1 ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 border border-slate-200'
                    }`}>
                      Rank #{rankNum}
                    </span>

                    {/* Watchlist status badge */}
                    {isStolen && (
                      <span className="text-[9px] font-sans font-bold px-2 py-0.5 rounded-md uppercase tracking-wider bg-rose-600 text-white animate-pulse">
                        ⚠️ STOLEN
                      </span>
                    )}
                    {isFlagged && !isStolen && (
                      <span className="text-[9px] font-sans font-bold px-2 py-0.5 rounded-md uppercase tracking-wider bg-amber-100 text-amber-700 border border-amber-300">
                        🚩 Flagged
                      </span>
                    )}
                    {isCloned && (
                      <span className="text-[9px] font-sans font-bold px-2 py-0.5 rounded-md uppercase tracking-wider bg-rose-100 text-rose-700 border border-rose-400 animate-pulse">
                        🔴 Cloned?
                      </span>
                    )}

                    <span className="font-mono text-base font-extrabold text-slate-900 tracking-wider">
                      {cand.plate_number}
                    </span>
                  </div>

                  {/* Character Diff Chips */}
                  <div className="flex items-center gap-1 flex-wrap">
                    {diffs.map((d, i) => {
                      const chipCls =
                        d.type === 'exact'
                          ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                          : d.type === 'wildcard'
                          ? 'bg-amber-100 text-amber-800 border-amber-300'
                          : d.type === 'confusion'
                          ? 'bg-amber-100 text-amber-800 border-amber-300'
                          : 'bg-rose-100 text-rose-800 border-rose-300';

                      return (
                        <span
                          key={i}
                          className={`font-mono text-[11px] font-bold w-6 h-6 rounded-md border flex items-center justify-center tabular-nums shadow-2xs ${chipCls}`}
                          title={`${d.type}: candidate '${d.char}' vs observed '${d.obsChar}'`}
                        >
                          {d.char}
                        </span>
                      );
                    })}
                  </div>
                </div>

                {/* Confidence Ring */}
                <div className="flex flex-col items-center">
                  <ConfidenceRing score={composite} isGatedOut={noConfidentMatch} />
                  <span className="text-[9px] font-sans font-semibold text-slate-400 mt-0.5 uppercase">Score</span>
                </div>
              </div>

              {/* Anomaly & Watchlist Banner (prominent, inside card) */}
              <AnomalyBanner candidate={cand} />

              {/* Vehicle Specifications Grid */}
              <div className="grid grid-cols-2 gap-2 text-xs font-sans bg-slate-50 p-3 rounded-xl border border-slate-100">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Make &amp; Model</span>
                  <span className="text-slate-800 font-bold truncate block">{cand.make} {cand.model}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Color &amp; Body</span>
                  <span className="text-slate-800 font-medium capitalize truncate block">{cand.colour} {cand.body_type}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Registered RTO</span>
                  <span className="text-slate-700 font-mono text-[11px] block truncate" title={cand.registered_rto || cand.rto_city}>
                    {cand.registered_rto || cand.rto_city || 'N/A'}
                    {cand.registration_year ? ` (${cand.registration_year})` : ''}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Synthetic Owner</span>
                  <span className="text-slate-700 font-mono text-[11px] truncate block" title={cand.synthetic_owner_id}>
                    {cand.owner_name || cand.synthetic_owner_id}
                  </span>
                </div>
              </div>

              {/* Forensic Score Breakdown — Plate / RTO Prior / Visual pills */}
              {hasForensicScores && (
                <div className="space-y-1.5 rounded-xl bg-slate-50 border border-slate-100 p-3">
                  <span className="text-[10px] font-sans font-bold uppercase tracking-wider text-slate-400 block mb-2">
                    Forensic Score Fusion
                  </span>
                  <ScoreBarRow label="Plate Similarity (OCR-aware)" score={topLevelPlate} colorClass="text-slate-800" />
                  <ScoreBarRow label="RTO Spatial Prior" score={topLevelRto} colorClass="text-purple-600" />
                  <ScoreBarRow label="Visual VLM (Body/Colour/Make)" score={topLevelVisual} colorClass="text-sky-600" />
                </div>
              )}

              {/* Classic score bars from score_breakdown */}
              <div className="space-y-1.5 text-xs font-sans text-slate-600">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="text-slate-500 font-medium">Plate Similarity</span>
                  <span className="font-mono font-bold text-slate-800">{plateScorePct}%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div className="h-full bg-slate-900 rounded-full" style={{ width: `${plateScorePct}%` }} />
                </div>

                <div className="flex items-center justify-between text-[11px] pt-1">
                  <span className="text-slate-500 font-medium">Colour Match</span>
                  <span className="font-mono font-bold text-emerald-600">{colourScorePct}%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${colourScorePct}%` }} />
                </div>

                <div className="flex items-center justify-between text-[11px] pt-1">
                  <span className="text-slate-500 font-medium">Body Type Compatibility</span>
                  <span className="font-mono font-bold text-blue-600">{bodyScorePct}%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: `${bodyScorePct}%` }} />
                </div>
              </div>

              {/* Plain-English Match Reasoning */}
              {reasons.length > 0 && (
                <div className="pt-2 border-t border-slate-100 space-y-1">
                  <span className="text-[10px] font-sans font-bold uppercase tracking-wider text-slate-400 block">
                    Match Rationale:
                  </span>
                  <ul className="space-y-1">
                    {reasons.map((r, rIdx) => (
                      <li key={rIdx} className="text-xs text-slate-600 flex items-start gap-1.5 font-sans leading-snug">
                        <span className="text-slate-400 mt-0.5">•</span>
                        <span>{r}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Multi-Camera Sighting Timeline — toggled for rank-1 by default, manually for others */}
              <div>
                {!isExpanded ? (
                  <button
                    type="button"
                    onClick={() => setExpandedPlate(cand.plate_number)}
                    className="text-[11px] text-sky-600 hover:text-sky-800 font-semibold font-sans flex items-center gap-1 pt-1"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Show sighting timeline
                  </button>
                ) : (
                  <>
                    <SightingTimeline plateNumber={cand.plate_number} />
                    <button
                      type="button"
                      onClick={() => setExpandedPlate(null)}
                      className="text-[11px] text-slate-400 hover:text-slate-600 font-semibold font-sans mt-1"
                    >
                      ▲ Collapse timeline
                    </button>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* 5. Forensic Dossier Export Card */}
      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-200/80">
          <span className="font-sans text-xs font-bold uppercase tracking-wider text-slate-800">
            Forensic Dossier Export
          </span>
          <span className="text-[10px] font-mono text-slate-500">COURT DOSSIER</span>
        </div>

        <p className="text-xs text-slate-500 font-sans leading-relaxed">
          Export an audit-compliant, self-contained forensic dossier with embedded base64 crops,
          confidence matrix, watchlist badges, and anomaly flags.
        </p>

        <div className="flex flex-col gap-2 pt-1">
          <button
            type="button"
            onClick={handleDownloadHtml}
            disabled={isExporting}
            className="tactile-button w-full bg-slate-900 hover:bg-slate-800 text-white font-sans font-semibold text-xs py-2.5 px-3 rounded-xl transition duration-150 flex items-center justify-center gap-2 shadow-sm disabled:opacity-50"
          >
            <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Download Forensic HTML Dossier</span>
          </button>

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={handlePrintPdf}
              disabled={isExporting}
              className="tactile-button bg-white hover:bg-slate-100 text-slate-700 font-sans text-xs font-semibold py-2 px-3 rounded-xl border border-slate-200 transition duration-150 flex items-center justify-center gap-1.5 shadow-sm active:scale-95 disabled:opacity-50"
            >
              <svg className="w-3.5 h-3.5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
              </svg>
              <span>Print / PDF</span>
            </button>

            <button
              type="button"
              onClick={handlePreviewReport}
              disabled={isExporting}
              className="tactile-button bg-white hover:bg-slate-100 text-slate-700 font-sans text-xs font-semibold py-2 px-3 rounded-xl border border-slate-200 transition duration-150 flex items-center justify-center gap-1.5 shadow-sm active:scale-95 disabled:opacity-50"
            >
              <svg className="w-3.5 h-3.5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              <span>Preview</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
