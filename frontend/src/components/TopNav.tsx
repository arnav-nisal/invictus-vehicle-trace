import React from 'react';
import { StepType } from '../types';

interface TopNavProps {
  currentStep: StepType;
  clipCount: number | null;
  isLoading: boolean;
  onStepClick?: (step: StepType) => void;
}

const STEPS: { id: StepType; label: string; number: string }[] = [
  { id: 'intake', label: 'Select Footage', number: '1' },
  { id: 'target', label: 'Select Target', number: '2' },
  { id: 'verify', label: 'Verification', number: '3' },
];

export const TopNav: React.FC<TopNavProps> = ({
  currentStep,
  clipCount,
  isLoading,
  onStepClick,
}) => {
  return (
    <header className="sticky top-0 z-50 h-14 w-full bg-white border-b border-[#E6E6E2] px-6">
      <div className="max-w-6xl mx-auto h-full flex items-center justify-between">
        {/* Left: INVICTUS Wordmark */}
        <div className="flex items-center gap-3">
          <span className="font-semibold text-xs tracking-[0.18em] text-[#111111] uppercase select-none">
            INVICTUS
          </span>
          <span className="hidden sm:inline-block w-px h-3.5 bg-[#E6E6E2]" />
          <span className="hidden sm:inline-block text-[11px] text-[#6B6B66] font-mono tracking-tight">
            CCTV Trace
          </span>
        </div>

        {/* Center: Quiet 3-step horizontal stepper */}
        <nav aria-label="Progress Stepper" className="hidden md:flex items-center gap-2 text-xs">
          {STEPS.map((step, index) => {
            const isActive = currentStep === step.id;
            const isPast =
              (currentStep === 'target' && step.id === 'intake') ||
              (currentStep === 'verify' && (step.id === 'intake' || step.id === 'target'));

            return (
              <React.Fragment key={step.id}>
                <div
                  onClick={() => {
                    if (isPast && onStepClick) {
                      onStepClick(step.id);
                    }
                  }}
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md transition-colors duration-150 ${
                    isActive
                      ? 'text-[#111111] font-medium bg-[#F1F1EE]'
                      : isPast
                      ? 'text-[#6B6B66] hover:text-[#111111] cursor-pointer'
                      : 'text-[#6B6B66]/60 cursor-default'
                  }`}
                >
                  <span
                    className={`font-mono text-[11px] tabular-nums ${
                      isActive
                        ? 'text-[#047857] font-semibold'
                        : isPast
                        ? 'text-[#6B6B66]'
                        : 'text-[#6B6B66]/60'
                    }`}
                  >
                    {step.number}.
                  </span>
                  <span>{step.label}</span>
                </div>

                {index < STEPS.length - 1 && (
                  <span className="text-[#E6E6E2] select-none text-xs font-mono">→</span>
                )}
              </React.Fragment>
            );
          })}
        </nav>

        {/* Right: Understated monospace pill */}
        <div className="flex items-center">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#F1F1EE] border border-[#E6E6E2] text-xs font-mono text-[#111111] tabular-nums shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isLoading
                  ? 'bg-amber-500 animate-pulse'
                  : clipCount && clipCount > 0
                  ? 'bg-[#047857]'
                  : 'bg-[#6B6B66]'
              }`}
            />
            <span>
              {isLoading
                ? 'Scanning...'
                : clipCount !== null
                ? `${clipCount} ${clipCount === 1 ? 'Clip' : 'Clips'} Ready`
                : 'Library Idle'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
