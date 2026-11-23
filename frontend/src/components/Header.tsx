import React from 'react';
import { DemoScenario, SystemHealth } from '../types';

interface HeaderProps {
  health: SystemHealth | null;
  onReset: () => void;
  scenarios?: DemoScenario[];
  selectedScenarioId?: string | null;
  onSelectScenario?: (scenario: DemoScenario) => void;
  onExportDossier?: () => void;
  isExporting?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  onReset,
  scenarios = [],
  selectedScenarioId,
  onSelectScenario,
  onExportDossier,
  isExporting = false,
}) => {
  const isOnline = health?.database_connected ?? true;
  const vehicleCount = health?.registry_vehicles_count ?? 1020;

  return (
    <header className="sticky top-4 z-40 px-4 sm:px-6 lg:px-8 max-w-[1440px] mx-auto w-full transition-all">
      <div className="bg-white/85 backdrop-blur-2xl border border-[rgba(15,23,42,0.08)] rounded-full px-4 sm:px-6 py-2.5 shadow-pill flex items-center justify-between gap-4">
        {/* Left: Emblem & Studio Brand */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-obsidian text-white flex items-center justify-center font-sans font-extrabold text-sm shadow-sm ring-2 ring-slate-900/10">
            <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-sans font-bold text-sm tracking-display text-slate-900">
                INVICTUS
              </h1>
              <span className="text-[10px] font-sans font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                Studio
              </span>
            </div>
            <p className="text-[11px] text-slate-500 font-sans hidden sm:block">
              Forensic Vehicle Intelligence & Bayesian Match
            </p>
          </div>
        </div>

        {/* Center: Quick Clip Switcher Tabs */}
        {scenarios.length > 0 && onSelectScenario && (
          <nav
            aria-label="CCTV Clip Presets"
            className="hidden xl:flex items-center bg-slate-100/80 p-1 rounded-full border border-slate-200/60"
          >
            {scenarios.map((scenario, idx) => {
              const isSelected = selectedScenarioId === scenario.id;
              const shortLabels = ['Clip 1: Blurry', 'Clip 2: Confusion', 'Clip 3: Partial', 'Clip 4: Out-Reg'];
              const label = shortLabels[idx] || `Clip ${idx + 1}`;

              return (
                <button
                  key={scenario.id}
                  type="button"
                  onClick={() => onSelectScenario(scenario)}
                  className={`px-3 py-1 rounded-full text-xs font-sans font-medium transition-all ${
                    isSelected
                      ? 'bg-white text-slate-900 font-semibold shadow-sm border border-slate-200/80'
                      : 'text-slate-500 hover:text-slate-900'
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </nav>
        )}

        {/* Right: Live Registry Status Pill & Obsidian Export Button */}
        <div className="flex items-center gap-2.5 sm:gap-3">
          {/* Live SQLite registry status pill */}
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-50 border border-slate-200/80 text-xs font-sans shadow-sm">
            <span
              className={`w-2 h-2 rounded-full ${
                isOnline ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50' : 'bg-rose-500'
              } animate-pulse`}
            />
            <span className="text-[11px] font-medium text-slate-600 hidden md:inline">
              Registry:
            </span>
            <span className="text-xs font-semibold text-slate-900 font-mono">
              {vehicleCount.toLocaleString()} Vehicles
            </span>
          </div>

          {/* High-Contrast Obsidian "Export Forensic Dossier" Button */}
          {onExportDossier && (
            <button
              type="button"
              onClick={onExportDossier}
              disabled={isExporting}
              className="tactile-button px-4 py-1.5 rounded-full bg-obsidian hover:bg-slate-800 text-white font-sans font-semibold text-xs transition duration-150 flex items-center gap-2 shadow-sm border border-white/10 active:scale-95 disabled:opacity-50"
              title="Generate and export comprehensive court-ready forensic report"
            >
              {isExporting ? (
                <>
                  <svg className="animate-spin h-3.5 w-3.5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Exporting...</span>
                </>
              ) : (
                <>
                  <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span className="hidden sm:inline">Export Forensic Dossier</span>
                  <span className="sm:hidden">Export</span>
                </>
              )}
            </button>
          )}

          {/* Reset button */}
          <button
            type="button"
            onClick={onReset}
            className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-slate-900 flex items-center justify-center transition border border-slate-200/80 active:scale-95"
            title="Reset workbench to default calibration state"
          >
            <svg
              className="w-3.5 h-3.5 text-slate-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.8"
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
};
