import React from 'react';
import { SystemHealth } from '../types';

interface HeaderProps {
  health: SystemHealth | null;
  onReset: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, onReset }) => {
  const isOnline = health?.database_connected ?? false;
  const vehicleCount = health?.registry_vehicles_count ?? 0;

  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        {/* Left: Branding & Case Info */}
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center font-mono font-black text-slate-950 text-xl shadow-lg shadow-cyan-500/20">
            IV
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h1 className="font-mono font-bold tracking-tight text-lg text-white">
                INVICTUS <span className="text-cyan-400">//</span> FORENSIC SURVEILLANCE WORKBENCH
              </h1>
              <span className="text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
                AIML-01
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Automated Multi-Signal CCTV Vehicle Re-Identification & Probabilistic Matching Engine
            </p>
          </div>
        </div>

        {/* Right: Live Diagnostics & Actions */}
        <div className="flex items-center gap-3 font-mono text-xs">
          {/* Status badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-800/80 border border-slate-700/60">
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isOnline ? 'bg-emerald-400' : 'bg-rose-400'}`} />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isOnline ? 'bg-emerald-500' : 'bg-rose-500'}`} />
            </span>
            <span className="text-slate-300">
              {isOnline ? 'SYSTEM ONLINE' : 'DISCONNECTED'}
            </span>
            <span className="text-slate-500">|</span>
            <span className="text-cyan-400 font-semibold">
              {vehicleCount > 0 ? `${vehicleCount.toLocaleString()} REGISTRY VEHICLES` : 'SYNCING...'}
            </span>
          </div>

          {/* Reset button */}
          <button
            onClick={onReset}
            className="px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition flex items-center gap-1.5 text-xs font-sans font-medium"
            title="Reset workbench to initial state"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Reset
          </button>
        </div>
      </div>
    </header>
  );
};
