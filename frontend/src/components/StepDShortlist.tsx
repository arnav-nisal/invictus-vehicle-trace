import React from 'react';
import { AnalysisResponse, MatchCandidate } from '../types';

interface StepDShortlistProps {
  analysis: AnalysisResponse | null;
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

function computeCharDiffs(observed: string, candidate: string): CharDiff[] {
  const diffs: CharDiff[] = [];
  const maxLen = Math.max(observed.length, candidate.length);

  for (let i = 0; i < maxLen; i++) {
    const obs = observed[i] || '';
    const cand = candidate[i] || '';

    if (obs === '?') {
      diffs.push({ char: cand, obsChar: obs, type: 'wildcard' });
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

export const StepDShortlist: React.FC<StepDShortlistProps> = ({ analysis }) => {
  if (!analysis) {
    return null;
  }

  const { top_matches, no_confident_match, explanation, evidence } = analysis;
  const observedPlate = evidence.consensus_plate;

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 mb-6 gap-2">
        <div className="flex items-center gap-3">
          <span className="w-7 h-7 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/80 font-mono text-xs font-bold flex items-center justify-center">
            D
          </span>
          <div>
            <h2 className="font-mono text-sm font-bold tracking-wider uppercase text-white">
              Probabilistic Shortlist Ranking & Candidate Evaluation
            </h2>
            <p className="text-xs text-slate-400">
              Multi-signal Bayesian ranking against synthetic vehicle registry with domain OCR confusion matrix
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
            Decision Threshold: <strong className="text-cyan-400">65.0% Composite</strong>
          </span>
        </div>
      </div>

      {/* Outcome Banner */}
      {no_confident_match ? (
        <div className="bg-rose-950/40 border-2 border-rose-600/80 rounded-xl p-5 mb-6 shadow-lg shadow-rose-950/30">
          <div className="flex items-start gap-3.5">
            <div className="w-10 h-10 rounded-lg bg-rose-900/60 border border-rose-700 flex items-center justify-center flex-shrink-0 text-rose-300">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="px-2 py-0.5 rounded bg-rose-900 text-rose-200 font-mono text-[10px] font-extrabold uppercase tracking-wider">
                  SECURITY GATE TRIGGERED
                </span>
                <h3 className="font-mono text-sm font-black text-rose-200 uppercase tracking-wide">
                  NO CONFIDENT REGISTRY MATCH IDENTIFIED
                </h3>
              </div>
              <p className="text-xs text-rose-300/90 leading-relaxed font-sans mt-1">
                {explanation ||
                  'No vehicle in the database achieved the required minimum confidence threshold (65.0%). Observed attributes strongly contradict registered records.'}
              </p>
              <div className="mt-2 text-[11px] font-mono text-rose-400">
                Notice: Candidate vehicles below are displayed strictly for audit analysis. None meet the forensic confidence threshold.
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-emerald-950/40 border-2 border-emerald-600/80 rounded-xl p-5 mb-6 shadow-lg shadow-emerald-950/30">
          <div className="flex items-start gap-3.5">
            <div className="w-10 h-10 rounded-lg bg-emerald-900/60 border border-emerald-700 flex items-center justify-center flex-shrink-0 text-emerald-300">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="px-2 py-0.5 rounded bg-emerald-900 text-emerald-200 font-mono text-[10px] font-extrabold uppercase tracking-wider">
                  CONFIDENT MATCH IDENTIFIED
                </span>
                <h3 className="font-mono text-sm font-black text-emerald-200 uppercase tracking-wide">
                  PRIMARY SUSPECT CONFIRMED (RANK #1)
                </h3>
              </div>
              <p className="text-xs text-emerald-300/90 leading-relaxed font-sans mt-1">
                {explanation ||
                  'High-confidence match confirmed based on multi-frame plate similarity, colour verification, and body classification.'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Diff Legend */}
      <div className="flex flex-wrap items-center gap-3 p-3 rounded-lg bg-slate-950 border border-slate-800 mb-6 text-[11px] font-mono">
        <span className="text-slate-400 font-semibold uppercase">Character Diff Key:</span>
        <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-700">
          <span className="w-2 h-2 rounded-full bg-emerald-400"></span> Exact Match
        </span>
        <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-700">
          <span className="w-2 h-2 rounded-full bg-amber-400"></span> Wildcard ? Slot
        </span>
        <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-700">
          <span className="w-2 h-2 rounded-full bg-purple-400"></span> OCR Confusion (8↔B, etc.)
        </span>
        <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-700">
          <span className="w-2 h-2 rounded-full bg-rose-400"></span> Mismatch
        </span>
      </div>

      {/* Ranked Candidate Cards */}
      <div className="space-y-4">
        {top_matches.map((cand) => {
          const isRank1 = cand.rank === 1 && !no_confident_match;
          const diffs = computeCharDiffs(observedPlate, cand.plate_number);
          const compositePct = (cand.composite_score * 100).toFixed(2);
          const plateScorePct = (cand.score_breakdown.plate_score * 100).toFixed(1);
          const colourScorePct = (cand.score_breakdown.colour_score * 100).toFixed(1);
          const bodyScorePct = (cand.score_breakdown.body_type_score * 100).toFixed(1);

          return (
            <div
              key={cand.plate_number}
              className={`rounded-xl border p-5 transition-all ${
                isRank1
                  ? 'bg-gradient-to-br from-slate-900 to-slate-850 border-cyan-500 ring-1 ring-cyan-500/40 shadow-xl shadow-cyan-950/40'
                  : 'bg-slate-850/60 border-slate-800 hover:border-slate-700'
              }`}
            >
              {/* Card Top Row: Rank, Plate Diffs, Composite Score */}
              <div className="flex flex-col md:flex-row items-start md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
                <div className="flex flex-wrap items-center gap-3">
                  {/* Rank Badge */}
                  <span
                    className={`font-mono text-xs font-black px-2.5 py-1 rounded tracking-wider uppercase ${
                      isRank1
                        ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/30'
                        : 'bg-slate-800 text-slate-300 border border-slate-700'
                    }`}
                  >
                    RANK #{cand.rank}
                  </span>

                  {/* Character-by-character diff chips */}
                  <div className="flex items-center gap-1 bg-slate-950 p-1.5 rounded-lg border border-slate-800">
                    {diffs.map((d, i) => {
                      const chipCls =
                        d.type === 'exact'
                          ? 'bg-emerald-950 text-emerald-300 border-emerald-700'
                          : d.type === 'wildcard'
                          ? 'bg-amber-950 text-amber-300 border-amber-700'
                          : d.type === 'confusion'
                          ? 'bg-purple-950 text-purple-300 border-purple-700'
                          : 'bg-rose-950 text-rose-300 border-rose-700';

                      return (
                        <span
                          key={i}
                          className={`font-mono text-xs font-bold w-6 h-6 rounded border flex items-center justify-center ${chipCls}`}
                          title={`${d.type.toUpperCase()}: candidate '${d.char}' vs observed '${d.obsChar}'`}
                        >
                          {d.char}
                        </span>
                      );
                    })}
                  </div>

                  <span className="font-mono text-sm font-bold text-white tracking-wider">
                    {cand.plate_number}
                  </span>
                </div>

                {/* Composite Score Display */}
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <span className="font-mono text-[10px] text-slate-400 uppercase block">
                      COMPOSITE SCORE
                    </span>
                    <span
                      className={`font-mono text-lg font-black ${
                        isRank1
                          ? 'text-cyan-400'
                          : no_confident_match
                          ? 'text-rose-400'
                          : 'text-slate-300'
                      }`}
                    >
                      {compositePct}%
                    </span>
                  </div>
                  <div className="w-16 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        isRank1
                          ? 'bg-cyan-400'
                          : no_confident_match
                          ? 'bg-rose-500'
                          : 'bg-slate-500'
                      }`}
                      style={{ width: `${compositePct}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Card Middle Row: Vehicle Attributes & Score Bars */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-4 font-mono text-xs">
                {/* Vehicle Specs */}
                <div className="grid grid-cols-2 gap-2.5 bg-slate-900/60 p-3.5 rounded-lg border border-slate-800/80">
                  <div>
                    <span className="text-[10px] text-slate-500 block uppercase">MAKE & MODEL</span>
                    <span className="font-bold text-slate-200">
                      {cand.make} {cand.model}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block uppercase">BODY & COLOUR</span>
                    <span className="font-bold text-slate-200 capitalize">
                      {cand.colour} {cand.body_type}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block uppercase">RTO JURISDICTION</span>
                    <span className="text-slate-300">{cand.rto_city}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block uppercase">SYNTHETIC OWNER ID</span>
                    <span className="text-cyan-400 font-semibold">{cand.synthetic_owner_id}</span>
                  </div>
                </div>

                {/* Score Breakdown Bars */}
                <div className="bg-slate-900/60 p-3.5 rounded-lg border border-slate-800/80 space-y-2">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-400">Plate Similarity:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-cyan-400 rounded-full" style={{ width: `${plateScorePct}%` }} />
                      </div>
                      <span className="w-10 text-right font-bold text-slate-200">{plateScorePct}%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-400">Colour Match:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-emerald-400 rounded-full" style={{ width: `${colourScorePct}%` }} />
                      </div>
                      <span className="w-10 text-right font-bold text-slate-200">{colourScorePct}%</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-slate-400">Body Type Compatibility:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-indigo-400 rounded-full" style={{ width: `${bodyScorePct}%` }} />
                      </div>
                      <span className="w-10 text-right font-bold text-slate-200">{bodyScorePct}%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Match Reasons List */}
              <div className="border-t border-slate-800/80 pt-3">
                <span className="text-[10px] font-mono uppercase text-slate-500 font-semibold block mb-1.5">
                  Algorithmic Match Reasoning:
                </span>
                <ul className="space-y-1">
                  {cand.match_reasons.map((reason, idx) => (
                    <li key={idx} className="text-xs text-slate-300 flex items-start gap-2">
                      <span className="text-cyan-400 mt-0.5">•</span>
                      <span>{reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
