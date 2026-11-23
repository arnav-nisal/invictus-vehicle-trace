import React, { useRef } from 'react';
import { DemoScenario } from '../types';

interface StepAFootageInputProps {
  scenarios: DemoScenario[];
  selectedScenarioId: string | null;
  onSelectScenario: (scenario: DemoScenario) => void;
  uploadedFile: File | null;
  onFileUpload: (file: File | null) => void;
  knownHint: string;
  onKnownHintChange: (hint: string) => void;
  onScanVideo: () => void;
  isScanning: boolean;
}

export const StepAFootageInput: React.FC<StepAFootageInputProps> = ({
  scenarios = [],
  selectedScenarioId,
  onSelectScenario,
  uploadedFile,
  onFileUpload,
  knownHint = '',
  onKnownHintChange,
  onScanVideo,
  isScanning,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scenarioList = Array.isArray(scenarios) ? scenarios : [];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
      {/* Step Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800 mb-6">
        <div className="flex items-center gap-3">
          <span className="w-7 h-7 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/80 font-mono text-xs font-bold flex items-center justify-center">
            A
          </span>
          <div>
            <h2 className="font-mono text-sm font-bold tracking-wider uppercase text-white">
              CCTV Footage Intake & Verification Scenarios
            </h2>
            <p className="text-xs text-slate-400">
              Upload local surveillance media or run one of our calibrated multi-signal test presets
            </p>
          </div>
        </div>
        <span className="text-[11px] font-mono text-slate-500 uppercase tracking-widest hidden sm:inline-block">
          SOURCE NODE: SURV-01
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (7 cols): 4 Pre-loaded Demo Scenarios */}
        <div className="lg:col-span-7 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                Pre-Loaded Test Scenarios (AIML-01 Presets)
              </label>
              <span className="text-[11px] text-slate-400 font-mono">
                {scenarioList.length} calibrated suites
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {scenarioList.map((scenario, idx) => {
                if (!scenario) return null;
                const isSelected = selectedScenarioId === scenario.id;
                const isBlur = scenario.id === 'scenario-1';
                const isConfusion = scenario.id === 'scenario-2';
                const isPartial = scenario.id === 'scenario-3';
                const isOutOfReg = scenario.id === 'scenario-4';

                const titleText = (scenario.title || `Scenario ${idx + 1}`).replace(/Scenario \d+: /, '');
                const plateText = scenario.evidence?.consensus_plate || 'UNKNOWN';
                const colourText = scenario.evidence?.colour || '';
                const bodyText = scenario.evidence?.body_type || '';

                return (
                  <button
                    key={scenario.id || idx}
                    type="button"
                    onClick={() => onSelectScenario(scenario)}
                    className={`text-left p-3.5 rounded-lg border transition duration-150 flex flex-col justify-between ${
                      isSelected
                        ? 'bg-cyan-950/40 border-cyan-500 shadow-md shadow-cyan-950/40'
                        : 'bg-slate-850/60 border-slate-700/80 hover:bg-slate-800 hover:border-slate-600'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-mono text-[11px] font-bold text-cyan-400">
                          SCENARIO {idx + 1}
                        </span>
                        <span
                          className={`text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border ${
                            isOutOfReg
                              ? 'bg-rose-950/80 text-rose-300 border-rose-800/80'
                              : isConfusion
                              ? 'bg-purple-950/80 text-purple-300 border-purple-800/80'
                              : isBlur
                              ? 'bg-amber-950/80 text-amber-300 border-amber-800/80'
                              : 'bg-emerald-950/80 text-emerald-300 border-emerald-800/80'
                          }`}
                        >
                          {isOutOfReg ? 'Out-of-Reg' : isConfusion ? '8↔B Confusion' : isBlur ? 'Blurry Plate' : 'Multi-Attr'}
                        </span>
                      </div>
                      <div className="font-semibold text-xs text-slate-200 line-clamp-1 mb-1">
                        {titleText}
                      </div>
                      <div className="flex items-center gap-2 mb-2 font-mono text-[11px]">
                        <span className="px-1.5 py-0.5 rounded bg-slate-900 text-slate-300 font-bold border border-slate-700">
                          {plateText}
                        </span>
                        {(colourText || bodyText) && (
                          <span className="text-slate-400 capitalize">
                            {colourText} {bodyText}
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="text-[11px] text-slate-400 line-clamp-2 mt-1 italic border-t border-slate-800/80 pt-2">
                      {scenario.expected_outcome || 'Benchmark validation test.'}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column (5 cols): File Upload Dropzone & Known Hint */}
        <div className="lg:col-span-5 flex flex-col justify-between space-y-4">
          {/* Dropzone */}
          <div>
            <label className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2 mb-3">
              <span className="w-2 h-2 rounded-full bg-blue-400"></span>
              Custom CCTV Video Footage
            </label>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition ${
                uploadedFile
                  ? 'border-cyan-500 bg-cyan-950/20'
                  : 'border-slate-700 hover:border-slate-500 bg-slate-850/40 hover:bg-slate-800/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/avi,.mp4,.avi"
                onChange={handleFileChange}
                className="hidden"
              />
              <div className="flex flex-col items-center justify-center py-2">
                <svg className="w-8 h-8 text-slate-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                {uploadedFile ? (
                  <div>
                    <p className="text-xs font-semibold text-cyan-300 font-mono line-clamp-1">
                      {uploadedFile.name}
                    </p>
                    <p className="text-[11px] text-slate-400">
                      {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for scan
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="text-xs text-slate-300 font-medium">
                      Drag & drop CCTV video (.mp4, .avi)
                    </p>
                    <p className="text-[11px] text-slate-500">
                      or click to browse local surveillance footage
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Known Partial Plate Witness Hint */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label htmlFor="known-hint" className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-300">
                Known Partial Plate Hint <span className="text-slate-500">(Optional)</span>
              </label>
              <span className="text-[10px] text-cyan-400 font-mono">WITNESS / TOLL CLUE</span>
            </div>
            <input
              id="known-hint"
              type="text"
              placeholder="e.g. MH12DE (blank by default)"
              value={knownHint}
              onChange={(e) => onKnownHintChange(e.target.value.toUpperCase())}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3.5 py-2 font-mono text-sm text-cyan-300 placeholder-slate-600 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 tracking-wider"
            />
            <p className="text-[11px] text-slate-400 mt-1">
              Provides investigator witness characters to resolve blurry/obscured slots during consensus.
            </p>
          </div>

          {/* Action Trigger Button */}
          <button
            type="button"
            onClick={onScanVideo}
            disabled={isScanning}
            className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-mono font-bold text-xs uppercase tracking-wider py-3 px-4 rounded-lg shadow-lg shadow-cyan-900/30 transition flex items-center justify-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isScanning ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Scanning Video Stream...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>Scan Video & Extract Vehicle Tracks</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
