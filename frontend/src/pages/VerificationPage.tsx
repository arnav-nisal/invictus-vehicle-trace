import React, { useState, useEffect } from 'react';
import { ArrowLeft, AlertCircle, RefreshCw } from 'lucide-react';
import { ClipItem, Stage1Result, VerificationResult } from '../types';
import { verifyPipeline } from '../services/api';
import { PlateGhostStack } from '../components/PlateGhostStack';
import { VehicleDetailsCard } from '../components/VehicleDetailsCard';
import { MatchesPanel } from '../components/MatchesPanel';

interface VerificationPageProps {
  selectedClip: ClipItem;
  selectedTrackIds: number[];
  stage1Result: Stage1Result;
  onBackToTarget: () => void;
}

export const VerificationPage: React.FC<VerificationPageProps> = ({
  selectedClip,
  selectedTrackIds,
  stage1Result,
  onBackToTarget,
}) => {
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [hoveredCandidatePlate, setHoveredCandidatePlate] = useState<string | null>(null);

  const executeVerification = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await verifyPipeline(
        stage1Result.job_id,
        selectedTrackIds,
        false
      );
      setResult(data);
    } catch (err) {
      setError((err as Error).message || 'Failed to complete forensic verification');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    executeVerification();
  }, [stage1Result.job_id, selectedTrackIds.join(',')]);

  // Warm-grey shimmer skeleton while POST /api/pipeline/verify is running
  if (isLoading) {
    return (
      <div className="min-h-[calc(100vh-3.5rem)] bg-[#F7F7F5] pb-24 pt-6 px-6">
        <div className="max-w-7xl mx-auto flex flex-col gap-6">
          {/* Skeleton Header */}
          <div className="flex items-center justify-between pb-4 border-b border-[#E6E6E2]">
            <div className="flex items-center gap-3">
              <div className="w-32 h-8 rounded-xl bg-[#EEEEEC] animate-pulse" />
              <div className="w-48 h-6 rounded-md bg-[#EEEEEC] animate-pulse" />
            </div>
            <div className="w-36 h-9 rounded-xl bg-[#EEEEEC] animate-pulse" />
          </div>

          {/* Skeleton 12-Col Bento Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
            {/* Left 7 Cols */}
            <div className="col-span-12 lg:col-span-7 flex flex-col gap-5">
              {/* Hero Plate Skeleton */}
              <div className="bg-white rounded-2xl border border-[#E6E6E2] p-6 flex flex-col gap-4 animate-pulse">
                <div className="flex justify-between items-center pb-3 border-b border-[#E6E6E2]">
                  <div className="w-44 h-4 bg-[#EEEEEC] rounded" />
                  <div className="w-28 h-6 bg-[#EEEEEC] rounded-full" />
                </div>
                <div className="flex flex-col gap-2 py-4">
                  <div className="w-full h-8 bg-[#EEEEEC] rounded-lg" />
                  <div className="w-full h-8 bg-[#EEEEEC] rounded-lg" />
                  <div className="w-full h-16 bg-[#EEEEEC] rounded-xl" />
                </div>
                <div className="flex justify-between items-center pt-3 border-t border-[#E6E6E2]">
                  <div className="w-48 h-3 bg-[#EEEEEC] rounded" />
                  <div className="w-32 h-7 bg-[#EEEEEC] rounded-xl" />
                </div>
              </div>

              {/* Vehicle Card Skeleton */}
              <div className="bg-white rounded-2xl border border-[#E6E6E2] p-6 flex flex-col gap-4 animate-pulse">
                <div className="w-40 h-4 bg-[#EEEEEC] rounded" />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                  <div className="aspect-video w-full rounded-xl bg-[#EEEEEC]" />
                  <div className="grid grid-cols-2 gap-3">
                    <div className="h-16 rounded-xl bg-[#EEEEEC]" />
                    <div className="h-16 rounded-xl bg-[#EEEEEC]" />
                    <div className="h-16 rounded-xl bg-[#EEEEEC]" />
                    <div className="h-16 rounded-xl bg-[#EEEEEC]" />
                  </div>
                </div>
              </div>
            </div>

            {/* Right 5 Cols */}
            <div className="col-span-12 lg:col-span-5 flex flex-col gap-5">
              <div className="bg-white rounded-2xl border border-[#E6E6E2] p-6 flex flex-col gap-4 animate-pulse">
                <div className="flex justify-between items-center pb-3 border-b border-[#E6E6E2]">
                  <div className="w-36 h-4 bg-[#EEEEEC] rounded" />
                  <div className="w-16 h-5 bg-[#EEEEEC] rounded-full" />
                </div>
                <div className="flex flex-col gap-3">
                  <div className="w-full h-32 bg-[#EEEEEC] rounded-xl" />
                  <div className="w-full h-14 bg-[#EEEEEC] rounded-xl" />
                  <div className="w-full h-14 bg-[#EEEEEC] rounded-xl" />
                </div>
              </div>

              <div className="bg-white rounded-2xl border border-[#E6E6E2] p-6 flex flex-col gap-3 animate-pulse">
                <div className="w-32 h-4 bg-[#EEEEEC] rounded" />
                <div className="w-full h-4 bg-[#EEEEEC] rounded" />
                <div className="w-full h-10 bg-[#EEEEEC] rounded-xl mt-2" />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Honest Data Error State
  if (error || !result) {
    return (
      <div className="min-h-[calc(100vh-3.5rem)] bg-[#F7F7F5] flex items-center justify-center p-6">
        <div className="w-full max-w-lg bg-white rounded-2xl border border-[#E6E6E2] p-8 shadow-[0_2px_12px_rgba(0,0,0,0.03)] flex flex-col items-center text-center gap-5">
          <div className="w-12 h-12 rounded-full bg-red-50 border border-red-100 text-red-600 flex items-center justify-center">
            <AlertCircle className="w-6 h-6" />
          </div>

          <div className="flex flex-col gap-1.5">
            <h2 className="text-lg font-semibold text-[#111111] tracking-tight">Data Error</h2>
            <p className="text-xs text-[#6B6B66] leading-relaxed max-w-sm">
              {error || 'Unable to retrieve live verification result from FastAPI backend.'}
            </p>
          </div>

          <div className="w-full bg-[#F7F7F5] rounded-xl p-3 text-left font-mono text-xs border border-[#E6E6E2] flex flex-col gap-1 text-[#6B6B66]">
            <div>Endpoint: <code className="text-[#111111]">POST /api/pipeline/verify</code></div>
            <div>Job ID: <code className="text-[#111111]">{stage1Result.job_id}</code></div>
            <div>Track IDs: <code className="text-[#111111]">[{selectedTrackIds.join(', ')}]</code></div>
          </div>

          <div className="flex items-center gap-3 pt-2">
            <button
              onClick={onBackToTarget}
              className="px-4 py-2 rounded-xl text-xs font-medium border border-[#E6E6E2] bg-white text-[#111111] hover:bg-[#F1F1EE] transition-colors"
            >
              ← Back
            </button>
            <button
              onClick={executeVerification}
              className="inline-flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-medium bg-[#111111] hover:bg-[#262626] text-white transition-colors shadow-sm"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-3.5rem)] bg-[#F7F7F5] pb-24 pt-6 px-6">
      <main className="max-w-7xl mx-auto flex flex-col gap-6">
        {/* Top Header & Breadcrumb */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E6E6E2] pb-4">
          <div className="flex items-center gap-3">
            <button
              onClick={onBackToTarget}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-xl border border-[#E6E6E2] bg-white hover:bg-[#F1F1EE] text-xs font-medium text-[#111111] transition-colors focus:outline-none"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Target Selection</span>
            </button>

            <span className="font-mono text-xs text-[#6B6B66] bg-[#F1F1EE] px-2.5 py-1 rounded-md border border-[#E6E6E2]">
              Stage 3 • Forensic Verification
            </span>
          </div>

          <div className="font-mono text-xs text-[#6B6B66] flex items-center gap-2">
            <span>Case:</span>
            <span className="text-[#111111] font-semibold">{result.case_id}</span>
            <span className="text-[#E6E6E2]">·</span>
            <span className="text-[#111111]">{selectedClip.filename}</span>
          </div>
        </div>

        {/* Asymmetric 12-Column Bento Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          {/* Left Column (7 Cols) */}
          <div className="col-span-12 lg:col-span-7 flex flex-col gap-5">
            {/* A. Reconstructed License Plate Hero Card (Centerpiece) */}
            <PlateGhostStack
              plate={result.plate}
              evidenceUrls={result.evidence_urls}
              isMergedTrack={result.is_merged_track}
              mergedTrackIds={result.merged_track_ids}
              hoveredCandidatePlate={hoveredCandidatePlate}
            />

            {/* B. Vehicle Classification & Evidence Card */}
            <VehicleDetailsCard
              vehicle={result.vehicle}
              evidenceUrls={result.evidence_urls}
              onAnalyzeAnother={onBackToTarget}
            />
          </div>

          {/* Right Column (5 Cols) */}
          <div className="col-span-12 lg:col-span-5 flex flex-col gap-5">
            <MatchesPanel
              result={result}
              onHoverPlate={setHoveredCandidatePlate}
            />
          </div>
        </div>
      </main>
    </div>
  );
};
