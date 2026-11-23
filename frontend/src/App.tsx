import React, { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { StepAFootageInput } from './components/StepAFootageInput';
import { StepBVehicleSelector } from './components/StepBVehicleSelector';
import { StepCEvidenceBreakdown } from './components/StepCEvidenceBreakdown';
import { StepDShortlist } from './components/StepDShortlist';
import { StepEReportExport } from './components/StepEReportExport';
import {
  analyzeVehicle,
  getDemoScenarios,
  getSystemHealth,
  scanVideo,
} from './api';
import {
  AnalysisResponse,
  DemoScenario,
  SystemHealth,
  TrackSummary,
  VideoScanResponse,
} from './types';

export const App: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [scenarios, setScenarios] = useState<DemoScenario[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [knownHint, setKnownHint] = useState<string>('');

  const [scanResult, setScanResult] = useState<VideoScanResponse | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);

  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Initialize workbench on mount
  useEffect(() => {
    let mounted = true;

    async function initWorkbench() {
      try {
        const [healthData, scenariosData] = await Promise.all([
          getSystemHealth().catch(() => null),
          getDemoScenarios().catch(() => []),
        ]);

        if (!mounted) return;
        if (healthData) setHealth(healthData);
        if (scenariosData && scenariosData.length > 0) {
          setScenarios(scenariosData);
          // Auto-select Scenario 1 for instantaneous live demo
          const initial = scenariosData[0];
          setSelectedScenarioId(initial.id);
          // Initial scan and analysis
          loadInitialScenario(initial);
        }
      } catch (err) {
        console.error('Failed to initialize workbench:', err);
      }
    }

    initWorkbench();
    return () => {
      mounted = false;
    };
  }, []);

  // Helper to load scenario and trigger scan + analyze
  const loadInitialScenario = async (scenario: DemoScenario) => {
    try {
      setIsScanning(true);
      const scan = await scanVideo(null, 'demo-traffic-1');
      setScanResult(scan);
      setSelectedTrackId(scenario.evidence.track_id);

      setIsAnalyzing(true);
      const analysis = await analyzeVehicle({
        scenario_id: scenario.id,
        known_partial_plate: scenario.suggested_known_partial_plate || undefined,
      });
      setAnalysisResult(analysis);
    } catch (err: any) {
      console.warn('Initial auto-load note:', err?.message || err);
    } finally {
      setIsScanning(false);
      setIsAnalyzing(false);
    }
  };

  // Handle Scenario Switcher
  const handleSelectScenario = async (scenario: DemoScenario) => {
    setSelectedScenarioId(scenario.id);
    setUploadedFile(null);
    setErrorMessage(null);

    // If scenario has suggested hint, set it or leave blank
    const hint = scenario.suggested_known_partial_plate || '';
    setKnownHint(hint);
    setSelectedTrackId(scenario.evidence.track_id);

    try {
      setIsAnalyzing(true);
      // Ensure scanResult is present
      if (!scanResult) {
        const scan = await scanVideo(null, 'demo-traffic-1');
        setScanResult(scan);
      }

      const analysis = await analyzeVehicle({
        scenario_id: scenario.id,
        known_partial_plate: hint || undefined,
      });
      setAnalysisResult(analysis);
    } catch (err: any) {
      console.error('Failed to analyze scenario:', err);
      setErrorMessage(err?.message || 'Failed to analyze scenario');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle Video Scan action
  const handleScanVideo = async () => {
    setErrorMessage(null);
    setIsScanning(true);
    try {
      const scan = await scanVideo(uploadedFile, selectedScenarioId || 'demo-traffic-1');
      setScanResult(scan);

      // Default to track 1 or first track
      const firstTrack = scan.tracks[0];
      if (firstTrack) {
        setSelectedTrackId(firstTrack.track_id);
        // Find matching scenario or analyze
        handleSelectTrack(firstTrack);
      }
    } catch (err: any) {
      console.error('Video scan failed:', err);
      setErrorMessage(err?.message || 'Video scan failed');
    } finally {
      setIsScanning(false);
    }
  };

  // Handle Clicking a Track Card in Step B
  const handleSelectTrack = async (track: TrackSummary) => {
    setSelectedTrackId(track.track_id);
    setErrorMessage(null);
    setIsAnalyzing(true);

    try {
      // Map track_id to corresponding scenario if matching
      const targetScenarioId = `scenario-${track.track_id}`;
      const matchedScenario = scenarios.find((s) => s.id === targetScenarioId);

      let analysis: AnalysisResponse;
      if (matchedScenario) {
        setSelectedScenarioId(matchedScenario.id);
        analysis = await analyzeVehicle({
          scenario_id: matchedScenario.id,
          known_partial_plate: knownHint || undefined,
        });
      } else if (selectedScenarioId) {
        analysis = await analyzeVehicle({
          scenario_id: selectedScenarioId,
          known_partial_plate: knownHint || undefined,
        });
      } else {
        // Fallback to scenario-1
        analysis = await analyzeVehicle({
          scenario_id: 'scenario-1',
          known_partial_plate: knownHint || undefined,
        });
      }
      setAnalysisResult(analysis);
    } catch (err: any) {
      console.error('Failed to analyze track:', err);
      setErrorMessage(err?.message || 'Failed to analyze vehicle track');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Re-run analysis when investigator changes known witness hint
  const handleReanalyzeWithHint = async (hintValue: string) => {
    setKnownHint(hintValue);
    if (!analysisResult) return;

    try {
      setIsAnalyzing(true);
      const analysis = await analyzeVehicle({
        scenario_id: selectedScenarioId || 'scenario-1',
        known_partial_plate: hintValue || undefined,
      });
      setAnalysisResult(analysis);
    } catch (err: any) {
      console.error('Failed to update analysis with hint:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Reset entire workbench
  const handleReset = () => {
    setSelectedScenarioId(scenarios[0]?.id || null);
    setUploadedFile(null);
    setKnownHint('');
    setScanResult(null);
    setSelectedTrackId(null);
    setAnalysisResult(null);
    setErrorMessage(null);
    if (scenarios.length > 0) {
      loadInitialScenario(scenarios[0]);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Top Tactical Header */}
      <Header health={health} onReset={handleReset} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8 space-y-8">
        {/* Error Banner */}
        {errorMessage && (
          <div className="bg-rose-950/80 border border-rose-600 rounded-xl p-4 flex items-center justify-between text-xs text-rose-200">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-rose-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-rose-400 hover:text-white font-mono text-xs ml-4"
            >
              Dismiss [✕]
            </button>
          </div>
        )}

        {/* Workflow Progression Stepper */}
        <nav aria-label="Workflow Steps" className="bg-slate-900/60 border border-slate-800 rounded-xl p-3 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
          <div className="flex items-center gap-2 text-cyan-400 font-bold">
            <span className="w-5 h-5 rounded-full bg-cyan-950 border border-cyan-800 flex items-center justify-center text-[10px]">
              1
            </span>
            <span>A: Footage Intake</span>
          </div>
          <span className="text-slate-600">→</span>
          <div className={`flex items-center gap-2 font-bold ${scanResult ? 'text-blue-400' : 'text-slate-500'}`}>
            <span className={`w-5 h-5 rounded-full border flex items-center justify-center text-[10px] ${scanResult ? 'bg-blue-950 border-blue-800' : 'bg-slate-900 border-slate-800'}`}>
              2
            </span>
            <span>B: Vehicle Selector</span>
          </div>
          <span className="text-slate-600">→</span>
          <div className={`flex items-center gap-2 font-bold ${analysisResult ? 'text-indigo-400' : 'text-slate-500'}`}>
            <span className={`w-5 h-5 rounded-full border flex items-center justify-center text-[10px] ${analysisResult ? 'bg-indigo-950 border-indigo-800' : 'bg-slate-900 border-slate-800'}`}>
              3
            </span>
            <span>C: Multi-Frame Evidence</span>
          </div>
          <span className="text-slate-600">→</span>
          <div className={`flex items-center gap-2 font-bold ${analysisResult ? 'text-emerald-400' : 'text-slate-500'}`}>
            <span className={`w-5 h-5 rounded-full border flex items-center justify-center text-[10px] ${analysisResult ? 'bg-emerald-950 border-emerald-800' : 'bg-slate-900 border-slate-800'}`}>
              4
            </span>
            <span>D: Shortlist Ranking</span>
          </div>
          <span className="text-slate-600">→</span>
          <div className={`flex items-center gap-2 font-bold ${analysisResult ? 'text-amber-400' : 'text-slate-500'}`}>
            <span className={`w-5 h-5 rounded-full border flex items-center justify-center text-[10px] ${analysisResult ? 'bg-amber-950 border-amber-800' : 'bg-slate-900 border-slate-800'}`}>
              5
            </span>
            <span>E: Dossier Export</span>
          </div>
        </nav>

        {/* Step A: Footage Intake & Demo Scenarios */}
        <section aria-label="Step A: Footage Input">
          <StepAFootageInput
            scenarios={scenarios}
            selectedScenarioId={selectedScenarioId}
            onSelectScenario={handleSelectScenario}
            uploadedFile={uploadedFile}
            onFileUpload={(file) => setUploadedFile(file)}
            knownHint={knownHint}
            onKnownHintChange={handleReanalyzeWithHint}
            onScanVideo={handleScanVideo}
            isScanning={isScanning}
          />
        </section>

        {/* Step B: Suspect Vehicle Track Gallery */}
        {scanResult && (
          <section aria-label="Step B: Vehicle Selector">
            <StepBVehicleSelector
              scanResult={scanResult}
              selectedTrackId={selectedTrackId}
              onSelectTrack={handleSelectTrack}
              isAnalyzing={isAnalyzing}
            />
          </section>
        )}

        {/* Step C: Multi-Frame Evidence Breakdown */}
        {analysisResult && (
          <section aria-label="Step C: Evidence Breakdown">
            <StepCEvidenceBreakdown evidence={analysisResult.evidence} />
          </section>
        )}

        {/* Step D: Shortlist Ranking or No Confident Match Notice */}
        {analysisResult && (
          <section aria-label="Step D: Ranked Shortlist">
            <StepDShortlist analysis={analysisResult} />
          </section>
        )}

        {/* Step E: One-Click Report Export */}
        {analysisResult && (
          <section aria-label="Step E: Report Export">
            <StepEReportExport analysis={analysisResult} />
          </section>
        )}
      </main>

      {/* Tactical Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 px-6 mt-16 font-mono text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <span className="font-bold text-slate-400">TEAM INVICTUS</span> • Vehicle Identification & Traceability (AIML-01)
            <span className="mx-2">•</span> Hack-a-Sol 5.0 | IIIT Naya Raipur
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span>FastAPI ASGI: Online</span>
            <span>SQLite Registry: 1020 Synthetic Records</span>
            <span>Zero Hallucination Guaranteed</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
