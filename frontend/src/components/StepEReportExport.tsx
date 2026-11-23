import React, { useState } from 'react';
import { AnalysisResponse } from '../types';
import { exportReportHtml } from '../api';

interface StepEReportExportProps {
  analysis: AnalysisResponse | null;
}

export const StepEReportExport: React.FC<StepEReportExportProps> = ({ analysis }) => {
  const [isExporting, setIsExporting] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [previewHtml, setPreviewHtml] = useState<string | null>(null);

  if (!analysis) {
    return null;
  }

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
        // Give time for base64 images to render, then print
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

  const handleTogglePreview = async () => {
    if (showPreview) {
      setShowPreview(false);
      return;
    }

    try {
      setIsExporting(true);
      const htmlText = await exportReportHtml(analysis);
      setPreviewHtml(htmlText);
      setShowPreview(true);
    } catch (err) {
      console.error('Failed to load report preview:', err);
      alert('Failed to load report preview.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 mb-6 gap-2">
        <div className="flex items-center gap-3">
          <span className="w-7 h-7 rounded bg-amber-950 text-amber-400 border border-amber-800/80 font-mono text-xs font-bold flex items-center justify-center">
            E
          </span>
          <div>
            <h2 className="font-mono text-sm font-bold tracking-wider uppercase text-white">
              One-Click Forensic Dossier Export (AIML-01 Case Report)
            </h2>
            <p className="text-xs text-slate-400">
              Generate self-contained, audit-compliant HTML & print-ready PDF case files with embedded keyframes
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
            Export Format: <strong className="text-amber-400">Standalone HTML & PDF</strong>
          </span>
        </div>
      </div>

      {/* Export Action Card */}
      <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-6 flex flex-col md:flex-row items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
            <h3 className="font-mono text-sm font-bold text-white tracking-wide uppercase">
              Forensic Case Report Package Ready
            </h3>
          </div>
          <p className="text-xs text-slate-400 max-w-xl">
            Fully self-contained package including case timestamps, target vehicle track ID #{analysis.track_id},
            embedded base64 keyframe crops, OCR character confidence table, consensus partial plate, and ranked
            probabilistic candidates with plain-English match rationale.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          {/* Download HTML Button */}
          <button
            type="button"
            onClick={handleDownloadHtml}
            disabled={isExporting}
            className="flex-1 md:flex-initial px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white border border-slate-700 font-mono text-xs font-bold uppercase tracking-wider transition flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Download HTML Report</span>
          </button>

          {/* Print / Save PDF Button */}
          <button
            type="button"
            onClick={handlePrintPdf}
            disabled={isExporting}
            className="flex-1 md:flex-initial px-5 py-2.5 rounded-lg bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-slate-950 font-mono text-xs font-black uppercase tracking-wider transition shadow-lg shadow-amber-950/40 flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <svg className="w-4 h-4 text-slate-950" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
            </svg>
            <span>Print / Save as PDF</span>
          </button>

          {/* Toggle In-Workbench Preview */}
          <button
            type="button"
            onClick={handleTogglePreview}
            disabled={isExporting}
            className="px-3.5 py-2.5 rounded-lg bg-slate-850 hover:bg-slate-800 text-slate-300 border border-slate-700 font-mono text-xs font-semibold uppercase tracking-wider transition flex items-center justify-center gap-1.5"
            title="Inspect report layout inline"
          >
            <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            <span>{showPreview ? 'Hide Preview' : 'Preview'}</span>
          </button>
        </div>
      </div>

      {/* Inline Preview Iframe */}
      {showPreview && previewHtml && (
        <div className="mt-6 border border-slate-700 rounded-xl overflow-hidden shadow-2xl bg-slate-950">
          <div className="bg-slate-850 px-4 py-2.5 border-b border-slate-700 flex items-center justify-between">
            <span className="font-mono text-xs font-bold text-slate-300 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              Interactive Report Preview (Stand-alone Document Output)
            </span>
            <button
              onClick={() => setShowPreview(false)}
              className="text-slate-400 hover:text-white font-mono text-xs"
            >
              Close [✕]
            </button>
          </div>
          <iframe
            title="Report Preview"
            srcDoc={previewHtml}
            className="w-full h-[650px] border-none bg-white"
          />
        </div>
      )}
    </div>
  );
};
