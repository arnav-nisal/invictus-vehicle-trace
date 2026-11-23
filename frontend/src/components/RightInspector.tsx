import React, { useState } from 'react';
import { AnalysisResponse } from '../types';
import { exportReportHtml } from '../api';
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
]);

interface CharDiff {
  char: string;
  obsChar: string;
  type: 'exact' | 'wildcard' | 'confusion' | 'mismatch';
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
      diffs.push({ char: cand || '?', obsChar: obs, type: 'wildcard' });
    } else if (obs === cand) {
      diffs.push({ char: cand, obsChar: obs, type: 'exact' });
    } else if (CONFUSION_PAIRS.has(`${obs}-${cand}`)) {
      diffs.push({ char: cand, obsChar: obs, type: 'confusion' });
    } else {
      diffs.push({ char: cand || '-', obsChar: obs, type: 'mismatch' });
    }
  }

  return diffs;
}

// SVG Confidence Ring Component
interface ConfidenceRingProps {
  score: number; // 0 to 1
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
    ? '#EF4444' // Rose
    : clampedScore >= 0.75
    ? '#10B981' // Emerald
    : clampedScore >= 0.65
    ? '#0284C7' // Sky
    : '#F59E0B'; // Amber

  return (
    <div className="relative inline-flex items-center justify-center flex-shrink-0" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#E2E8F0"
          strokeWidth={strokeWidth}
          fill="transparent"
        />
        {/* Animated Progress ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-500 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="font-mono font-bold text-[11px] text-slate-900 leading-none">
          {pctText}%
        </span>
      </div>
    </div>
  );
};

export const RightInspector: React.FC<RightInspectorProps> = ({
  analysis,
  isAnalyzing,
  onOpenPreviewModal,
}) => {
  const [isExporting, setIsExporting] = useState(false);

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
        setTimeout(() => {
          printWindow.print();
        }, 300);
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
      {/* 1. Header with Confidence Gating Indicator */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-900" />
            <h2 className="text-base font-bold font-sans text-slate-900 tracking-snugger">
              Registry Shortlist & Probabilistic Match
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
            <div className="w-5 h-5 rounded-full bg-rose-600 text-white flex items-center justify-center font-bold text-xs flex-shrink-0">
              !
            </div>
            <span className="font-sans text-xs font-bold uppercase tracking-wider text-rose-900">
              Security Gate Triggered • No Confident Match
            </span>
          </div>

          <p className="text-xs text-rose-800 font-sans leading-relaxed">
            {explanation ||
              'No vehicle in registry achieved the minimum 65.0% composite threshold. Physical attributes contradict database records.'}
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
            {explanation ||
              'Suspect resolved via multi-frame consensus OCR, colorimetric verification, and confusion matrix tolerance.'}
          </p>
        </div>
      )}

      {/* 3. Character Alignment Diff Legend */}
      <div className="flex flex-wrap items-center gap-2 p-2.5 rounded-2xl bg-slate-50 border border-slate-200/80 text-[11px] font-sans">
        <span className="text-slate-500 font-semibold mr-1">Diff Alignment:</span>
        <span className="px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-800 font-medium border border-emerald-200">
          Exact Match
        </span>
        <span className="px-2 py-0.5 rounded-md bg-amber-100 text-amber-800 font-medium border border-amber-200">
          8↔B Tolerance
        </span>
        <span className="px-2 py-0.5 rounded-md bg-amber-50 text-amber-700 font-medium border border-amber-300">
          Wildcard ?
        </span>
        <span className="px-2 py-0.5 rounded-md bg-rose-100 text-rose-700 font-medium border border-rose-200">
          Mismatch
        </span>
      </div>

      {/* 4. Top-5 Ranked Match Cards with SVG Confidence Rings */}
      <div className="space-y-4">
        {topMatches.map((cand, idx) => {
          const rankNum = cand.rank ?? idx + 1;
          const isRank1 = rankNum === 1 && !noConfidentMatch;
          const diffs = computeCharDiffs(observedPlate, cand.plate_number);
          const composite = cand.composite_score ?? 0;
          const plateScorePct = (((cand.score_breakdown?.plate_score) ?? 0) * 100).toFixed(1);
          const colourScorePct = (((cand.score_breakdown?.colour_score) ?? 0) * 100).toFixed(1);
          const bodyScorePct = (((cand.score_breakdown?.body_type_score) ?? 0) * 100).toFixed(1);
          const reasons = Array.isArray(cand.match_reasons) ? cand.match_reasons : [];

          return (
            <div
              key={cand.plate_number || idx}
              className={`rounded-2xl border p-4 space-y-3.5 transition-all ${
                isRank1
                  ? 'border-slate-900 bg-white shadow-bento ring-1 ring-slate-900/10'
                  : 'border-slate-200 bg-white hover:border-slate-300 shadow-sm'
              }`}
            >
              {/* Card Header: Rank, Diff Chips, Plate, and SVG Confidence Ring */}
              <div className="flex items-center justify-between pb-3 border-b border-slate-100 gap-3">
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-[10px] font-sans font-bold px-2 py-0.5 rounded-full uppercase ${
                        isRank1
                          ? 'bg-slate-900 text-white'
                          : 'bg-slate-100 text-slate-600 border border-slate-200'
                      }`}
                    >
                      Rank #{rankNum}
                    </span>

                    <span className="font-mono text-base font-extrabold text-slate-900 tracking-wider">
                      {cand.plate_number}
                    </span>
                  </div>

                  {/* Character Diff Chips Alignment Row */}
                  <div className="flex items-center gap-1">
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

                {/* SVG Confidence Ring */}
                <div className="flex flex-col items-center">
                  <ConfidenceRing score={composite} isGatedOut={noConfidentMatch} />
                  <span className="text-[9px] font-sans font-semibold text-slate-400 mt-0.5 uppercase">
                    Score
                  </span>
                </div>
              </div>

              {/* Vehicle Specifications Grid */}
              <div className="grid grid-cols-2 gap-2 text-xs font-sans bg-slate-50 p-3 rounded-xl border border-slate-100">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Make & Model</span>
                  <span className="text-slate-800 font-bold truncate block">
                    {cand.make} {cand.model}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Color & Body</span>
                  <span className="text-slate-800 font-medium capitalize truncate block">
                    {cand.colour} {cand.body_type}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">RTO Jurisdiction</span>
                  <span className="text-slate-700 font-mono text-[11px] block">{cand.rto_city || 'N/A'}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Synthetic Owner ID</span>
                  <span className="text-slate-700 font-mono text-[11px] truncate block">
                    {cand.synthetic_owner_id}
                  </span>
                </div>
              </div>

              {/* Micro Attribute Breakdown Bars */}
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
            </div>
          );
        })}
      </div>

      {/* 5. Dossier Export Studio Card */}
      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-200/80">
          <span className="font-sans text-xs font-bold uppercase tracking-wider text-slate-800">
            Forensic Dossier Export
          </span>
          <span className="text-[10px] font-mono text-slate-500">COURT DOSSIER</span>
        </div>

        <p className="text-xs text-slate-500 font-sans leading-relaxed">
          Export an audit-compliant, self-contained forensic dossier with embedded base64 crops,
          confidence matrix, and candidate ranking.
        </p>

        <div className="flex flex-col gap-2 pt-1">
          {/* Download HTML Button */}
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

          {/* Print / Save PDF & Preview */}
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
