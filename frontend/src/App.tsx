import React, { useState } from 'react';
import { TopNav } from './components/TopNav';
import { IntakePage } from './pages/IntakePage';
import { TrackSelectPage } from './pages/TrackSelectPage';
import { VerificationPage } from './pages/VerificationPage';
import { ErrorBoundary } from './components/ErrorBoundary';
import { ClipItem, StepType, Stage1Result } from './types';

export const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<StepType>('intake');
  const [selectedClip, setSelectedClip] = useState<ClipItem | null>(null);
  const [selectedTrackIds, setSelectedTrackIds] = useState<number[]>([]);
  const [stage1Result, setStage1Result] = useState<Stage1Result | null>(null);
  const [clipCount, setClipCount] = useState<number | null>(null);
  const [isLoadingLibrary, setIsLoadingLibrary] = useState<boolean>(true);

  const handleClipsLoaded = (count: number) => {
    setClipCount(count);
    setIsLoadingLibrary(false);
  };

  const handleContinueToTarget = () => {
    if (selectedClip) {
      setCurrentStep('target');
    }
  };

  const handleBackToIntake = () => {
    setCurrentStep('intake');
  };

  const handleContinueToVerify = (trackIds: number[], result: Stage1Result) => {
    setSelectedTrackIds(trackIds);
    setStage1Result(result);
    setCurrentStep('verify');
  };

  const handleBackToTarget = () => {
    setCurrentStep('target');
  };

  return (
    <ErrorBoundary onReset={handleBackToIntake}>
      <div className="min-h-screen bg-[#F7F7F5] flex flex-col font-sans antialiased">
        {/* 56px Persistent Header */}
        <TopNav
          currentStep={currentStep}
          clipCount={clipCount}
          isLoading={isLoadingLibrary}
          onStepClick={(step) => {
            if (step === 'intake') {
              setCurrentStep('intake');
            } else if (step === 'target' && selectedClip) {
              setCurrentStep('target');
            }
          }}
        />

        {/* Screen 1: Footage Library Selection */}
        {currentStep === 'intake' && (
          <IntakePage
            selectedClip={selectedClip}
            onSelectClip={(clip) => setSelectedClip(clip)}
            onContinue={handleContinueToTarget}
            onClipsLoaded={handleClipsLoaded}
          />
        )}

        {/* Screen 2: Near-Fullscreen Video Canvas & Multi-ID Target Selection */}
        {currentStep === 'target' && selectedClip && (
          <TrackSelectPage
            selectedClip={selectedClip}
            onBackToIntake={handleBackToIntake}
            onContinueToVerify={handleContinueToVerify}
          />
        )}

        {/* Screen 3: 12-Column Asymmetric Bento Forensic Verification Workspace */}
        {currentStep === 'verify' && selectedClip && stage1Result && (
          <VerificationPage
            selectedClip={selectedClip}
            selectedTrackIds={selectedTrackIds}
            stage1Result={stage1Result}
            onBackToTarget={handleBackToTarget}
          />
        )}
      </div>
    </ErrorBoundary>
  );
};

export default App;
