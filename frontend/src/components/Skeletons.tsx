import React from 'react';

export const TrackSkeleton: React.FC = () => (
  <div className="rounded-2xl border border-slate-200/80 bg-slate-50/50 p-3 space-y-2.5 animate-pulse">
    <div className="w-full h-24 rounded-xl bg-slate-200/60" />
    <div className="flex items-center justify-between">
      <div className="h-3.5 w-16 bg-slate-200 rounded" />
      <div className="h-3 w-12 bg-slate-200 rounded" />
    </div>
    <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-100">
      <div className="h-3 w-14 bg-slate-200/70 rounded" />
      <div className="h-3 w-14 bg-slate-200/70 rounded" />
    </div>
  </div>
);

export const PlateConsensusSkeleton: React.FC = () => (
  <div className="rounded-3xl border border-white/10 bg-obsidian p-6 space-y-4 animate-pulse">
    <div className="flex items-center justify-between">
      <div className="h-4 w-40 bg-white/10 rounded-full" />
      <div className="h-3 w-28 bg-white/10 rounded-full" />
    </div>
    <div className="flex flex-wrap gap-2 py-4 justify-center">
      {[...Array(10)].map((_, i) => (
        <div
          key={i}
          className="w-12 sm:w-14 h-20 rounded-xl bg-white/10 flex flex-col items-center justify-between p-2"
        >
          <div className="w-4 h-2 bg-white/20 rounded" />
          <div className="w-6 h-6 bg-white/20 rounded" />
          <div className="w-8 h-1.5 bg-white/20 rounded-full" />
        </div>
      ))}
    </div>
  </div>
);

export const FilmstripSkeleton: React.FC = () => (
  <div className="rounded-3xl border border-white/10 bg-obsidian p-6 space-y-3 animate-pulse">
    <div className="flex items-center justify-between">
      <div className="h-4 w-44 bg-white/10 rounded-full" />
      <div className="h-3 w-28 bg-white/10 rounded-full" />
    </div>
    <div className="flex gap-4 overflow-x-hidden pb-2">
      {[...Array(3)].map((_, i) => (
        <div
          key={i}
          className="flex-shrink-0 w-64 rounded-2xl bg-white/5 border border-white/10 p-3 space-y-2.5"
        >
          <div className="flex items-center justify-between">
            <div className="h-3 w-16 bg-white/10 rounded" />
            <div className="h-3 w-12 bg-white/10 rounded" />
          </div>
          <div className="w-full h-24 bg-white/10 rounded-xl" />
          <div className="w-full h-10 bg-white/10 rounded-xl" />
          <div className="flex items-center justify-between pt-1 border-t border-white/10">
            <div className="h-3 w-16 bg-white/10 rounded" />
            <div className="h-3 w-10 bg-white/10 rounded" />
          </div>
        </div>
      ))}
    </div>
  </div>
);

export const CandidateCardSkeleton: React.FC = () => (
  <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-3 animate-pulse shadow-sm">
    <div className="flex items-center justify-between pb-3 border-b border-slate-100">
      <div className="flex items-center gap-2">
        <div className="h-4 w-14 bg-slate-200 rounded-full" />
        <div className="h-5 w-24 bg-slate-200 rounded" />
      </div>
      <div className="w-10 h-10 rounded-full bg-slate-200" />
    </div>
    <div className="grid grid-cols-2 gap-2">
      <div className="h-3 w-20 bg-slate-100 rounded" />
      <div className="h-3 w-20 bg-slate-100 rounded" />
      <div className="h-3 w-24 bg-slate-100 rounded" />
      <div className="h-3 w-24 bg-slate-100 rounded" />
    </div>
    <div className="pt-2 border-t border-slate-100 space-y-1.5">
      <div className="h-2 w-full bg-slate-100 rounded" />
      <div className="h-2 w-4/5 bg-slate-100 rounded" />
    </div>
  </div>
);
