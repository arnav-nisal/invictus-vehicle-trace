import React, { useState } from 'react';
import {
  FileDown,
  Layers,
  MapPin,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Clock,
  Loader2,
} from 'lucide-react';
import { RegistryMatch, VerificationResult } from '../types';
import { downloadPdfReport } from '../services/api';

interface MatchesPanelProps {
  result: VerificationResult;
  onHoverPlate?: (plate: string | null) => void;
}

export const MatchesPanel: React.FC<MatchesPanelProps> = ({ result, onHoverPlate }) => {
  const matches = result.registry_matches || [];
  const [expandedRanks, setExpandedRanks] = useState<Record<number, boolean>>({ 1: true });
  const [isSightingsOpen, setIsSightingsOpen] = useState<boolean>(true);

  // PDF Download state
  const [isDownloadingPdf, setIsDownloadingPdf] = useState<boolean>(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const toggleRank = (rank: number) => {
    setExpandedRanks((prev) => ({
      ...prev,
      [rank]: !prev[rank],
    }));
  };

  const handleDownloadPdf = async () => {
    setIsDownloadingPdf(true);
    setPdfError(null);
    try {
      const filename = `Invictus_Report_${result.job_id}_${result.track_key}.pdf`;
      await downloadPdfReport(result.pdf_report_url, filename);
    } catch (err) {
      setPdfError((err as Error).message || 'Failed to download PDF report');
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  return (
    <div className="flex flex-col gap-5">
      {/* 1. Ranked Registry Matches Card */}
      <div className="bg-white rounded-2xl border border-[#E6E6E2] p-6 shadow-[0_1px_3px_rgba(0,0,0,0.02)] flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-[#E6E6E2] pb-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#111111]" />
            <span className="text-xs font-semibold uppercase tracking-wider text-[#111111]">
              Registry Matches
            </span>
          </div>

          <span className="font-mono text-[11px] text-[#6B6B66] bg-[#F1F1EE] px-2 py-0.5 rounded-full border border-[#E6E6E2] tabular-nums">
            {matches.length} Records
          </span>
        </div>

        {matches.length === 0 ? (
          <div className="py-8 text-center flex flex-col items-center gap-2 text-[#6B6B66]">
            <p className="text-xs font-mono">No matching record found in local registry</p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {matches.map((cand) => {
              const isRank1 = cand.rank === 1;
              const isExpanded = isRank1 || Boolean(expandedRanks[cand.rank]);

              return (
                <div
                  key={cand.rank}
                  onMouseEnter={() => onHoverPlate?.(cand.plate_number)}
                  onMouseLeave={() => onHoverPlate?.(null)}
                  className={`rounded-xl border transition-all ${
                    isRank1
                      ? 'border-[#047857] bg-[#F0FDF4]/30 shadow-sm'
                      : 'border-[#E6E6E2] bg-white hover:border-[#111111]/30'
                  }`}
                >
                  {/* Card Header Row */}
                  <div
                    onClick={() => !isRank1 && toggleRank(cand.rank)}
                    className={`p-3.5 flex items-center justify-between gap-3 ${
                      !isRank1 ? 'cursor-pointer select-none' : ''
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <span
                        className={`w-5 h-5 rounded-full flex items-center justify-center font-mono text-[10px] font-bold ${
                          isRank1 ? 'bg-[#047857] text-white' : 'bg-[#F1F1EE] text-[#6B6B66]'
                        }`}
                      >
                        #{cand.rank}
                      </span>
                      <span className="font-mono text-sm font-bold text-[#111111] tracking-tight">
                        {cand.plate_number}
                      </span>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className="font-mono text-xs font-semibold text-[#047857] tabular-nums">
                        {Math.round(cand.total_confidence * 100)}% Match
                      </span>
                      {!isRank1 && (
                        <span className="text-[#6B6B66]">
                          {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <div className="px-3.5 pb-3.5 pt-1 border-t border-[#E6E6E2]/70 flex flex-col gap-3 text-xs">
                      {/* Vehicle Summary */}
                      <div className="text-[#111111] font-medium leading-relaxed">
                        {cand.colour} {cand.make} {cand.model} · {cand.body_type}
                      </div>

                      {/* Owner, RTO, Year */}
                      <div className="flex flex-col gap-1 font-mono text-[11px] text-[#6B6B66] bg-[#F7F7F5] p-2.5 rounded-lg border border-[#E6E6E2]">
                        <div className="flex justify-between">
                          <span>Owner:</span>
                          <span className="text-[#111111] font-semibold">{cand.owner_name}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>RTO Location:</span>
                          <span className="text-[#111111]">{cand.registered_rto}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Registration Year:</span>
                          <span className="text-[#111111]">{cand.registration_year}</span>
                        </div>
                      </div>

                      {/* Score Fusion Bars */}
                      <div className="flex flex-col gap-2 font-mono text-[11px]">
                        <div className="flex flex-col gap-1">
                          <div className="flex justify-between text-[#6B6B66]">
                            <span>Plate Match</span>
                            <span className="font-semibold text-[#111111]">
                              {Math.round(cand.plate_score * 100)}%
                            </span>
                          </div>
                          <div className="w-full h-1.5 bg-[#F1F1EE] rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[#047857] rounded-full"
                              style={{ width: `${Math.round(cand.plate_score * 100)}%` }}
                            />
                          </div>
                        </div>

                        <div className="flex flex-col gap-1">
                          <div className="flex justify-between text-[#6B6B66]">
                            <span>Visual Match</span>
                            <span className="font-semibold text-[#111111]">
                              {Math.round(cand.visual_score * 100)}%
                            </span>
                          </div>
                          <div className="w-full h-1.5 bg-[#F1F1EE] rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[#111111] rounded-full"
                              style={{ width: `${Math.round(cand.visual_score * 100)}%` }}
                            />
                          </div>
                        </div>
                      </div>

                      {/* Sightings Timeline for Rank #1 */}
                      {isRank1 && cand.sightings && cand.sightings.length > 0 && (
                        <div className="pt-2 border-t border-[#E6E6E2] flex flex-col gap-1.5">
                          <button
                            onClick={() => setIsSightingsOpen(!isSightingsOpen)}
                            className="flex items-center justify-between text-[11px] font-mono font-semibold text-[#047857] uppercase text-left select-none"
                          >
                            <span>Surveillance Sightings ({cand.sightings.length})</span>
                            {isSightingsOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                          </button>

                          {isSightingsOpen && (
                            <div className="flex flex-col gap-1.5 pt-1">
                              {cand.sightings.map((s, si) => (
                                <div
                                  key={si}
                                  className="flex items-start gap-2 p-2 rounded bg-white border border-[#E6E6E2] font-mono text-[11px] leading-tight"
                                >
                                  <MapPin className="w-3.5 h-3.5 text-[#047857] shrink-0 mt-0.5" />
                                  <div className="flex flex-col min-w-0">
                                    <span className="font-semibold text-[#111111] truncate">{s.camera_id}</span>
                                    <span className="text-[#6B6B66] text-[10px]">{s.timestamp}</span>
                                    {s.location && <span className="text-[#6B6B66] text-[10px]">{s.location}</span>}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 2. Compact Case Metadata & Anchored PDF Export Button */}
      <div className="bg-white rounded-2xl border border-[#E6E6E2] p-6 shadow-[0_1px_3px_rgba(0,0,0,0.02)] flex flex-col gap-4 font-mono text-xs">
        <div className="flex items-center gap-2 text-[#111111] font-semibold uppercase pb-2 border-b border-[#E6E6E2]">
          <Clock className="w-4 h-4" />
          <span>Case Provenance</span>
        </div>

        <div className="flex justify-between text-[#6B6B66]">
          <span>Case ID:</span>
          <span className="text-[#111111] font-bold">{result.case_id}</span>
        </div>
        <div className="flex justify-between text-[#6B6B66]">
          <span>Track Key:</span>
          <span className="text-[#111111] truncate max-w-[180px]">{result.track_key}</span>
        </div>
        <div className="flex justify-between text-[#6B6B66]">
          <span>Processing Time:</span>
          <span className="text-[#111111]">{result.processing_time_sec}s</span>
        </div>

        {/* Inline PDF Error if failed */}
        {pdfError && (
          <div className="p-2.5 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span className="truncate">{pdfError}</span>
          </div>
        )}

        {/* Direct Blob PDF Downloader */}
        <div className="pt-2 border-t border-[#E6E6E2]">
          <button
            onClick={handleDownloadPdf}
            disabled={isDownloadingPdf}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-[#111111] hover:bg-[#262626] text-xs font-semibold text-white transition-colors disabled:opacity-50 shadow-sm"
          >
            {isDownloadingPdf ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Downloading PDF Report...</span>
              </>
            ) : (
              <>
                <FileDown className="w-3.5 h-3.5" />
                <span>Export Forensic PDF Report</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
