import React, { useEffect } from 'react';

interface ReportPreviewModalProps {
  html: string | null;
  onClose: () => void;
}

export const ReportPreviewModal: React.FC<ReportPreviewModalProps> = ({ html, onClose }) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!html) return null;

  const handlePrint = () => {
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.open();
      printWindow.document.write(html);
      printWindow.document.close();
      setTimeout(() => {
        printWindow.print();
      }, 300);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-obsidian/80 backdrop-blur-xl p-4 sm:p-6"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Forensic Report Dossier Preview"
    >
      <div
        className="w-full max-w-5xl h-[88vh] bg-white border border-slate-200 rounded-3xl shadow-obsidian flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-4 bg-obsidian border-b border-white/10 flex items-center justify-between text-white">
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse shadow-sm shadow-emerald-400" />
            <div>
              <h3 className="font-sans font-bold text-sm tracking-snugger text-white">
                Forensic Case Dossier Preview (Standalone HTML Document)
              </h3>
              <p className="text-xs text-slate-400 font-sans">
                Self-contained judicial report with embedded visual evidence and cryptographic audit metadata
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={handlePrint}
              className="tactile-button px-4 py-1.5 rounded-full bg-white/10 hover:bg-white/20 text-white font-sans text-xs font-semibold border border-white/10 transition active:scale-95"
            >
              Print / Save PDF
            </button>
            <button
              type="button"
              onClick={onClose}
              className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-slate-300 hover:text-white flex items-center justify-center transition border border-white/10 active:scale-95"
              title="Close [Esc]"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Modal Body: Document Iframe */}
        <div className="flex-1 bg-white overflow-hidden">
          <iframe
            title="Dossier Document"
            srcDoc={html}
            className="w-full h-full border-none"
          />
        </div>
      </div>
    </div>
  );
};
