import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertCircle, RefreshCw, ArrowLeft } from 'lucide-react';

interface ErrorBoundaryProps {
  children: ReactNode;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught an unhandled runtime error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReset = (): void => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    if (this.state.hasError) {
      const errorMessage = this.state.error?.message || 'Unknown runtime error occurred';
      const errorStack = this.state.error?.stack || this.state.errorInfo?.componentStack || '';

      return (
        <div className="min-h-screen bg-[#F7F7F5] flex items-center justify-center p-6 antialiased selection:bg-[#111111] selection:text-white">
          <div className="w-full max-w-xl bg-white rounded-2xl border border-[#E6E6E2] p-8 shadow-[0_4px_24px_rgba(0,0,0,0.04)] flex flex-col gap-6">
            {/* Header with status badge */}
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-xl bg-amber-50 border border-amber-200 text-amber-700 flex items-center justify-center shrink-0 shadow-sm">
                <AlertCircle className="w-6 h-6 stroke-[2]" />
              </div>

              <div className="flex flex-col gap-1.5 flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded bg-amber-100 text-amber-900 border border-amber-200">
                    Application Shield
                  </span>
                  <span className="font-mono text-xs text-[#6B6B66]">Stage 1 / Stage 2 Guard</span>
                </div>
                <h1 className="text-lg font-semibold text-[#111111] tracking-tight">
                  Runtime Exception Recovered
                </h1>
                <p className="text-xs text-[#6B6B66] leading-relaxed">
                  An unexpected render exception was safely caught. The application prevented a blank screen and preserved system state.
                </p>
              </div>
            </div>

            {/* Error Message Box */}
            <div className="flex flex-col gap-2">
              <div className="text-[11px] font-mono font-semibold text-[#111111] uppercase tracking-wider">
                Error Details
              </div>
              <div className="p-3.5 rounded-xl bg-[#F7F7F5] border border-[#E6E6E2] text-[#111111] font-mono text-xs break-all leading-relaxed max-h-36 overflow-y-auto">
                {errorMessage}
              </div>

              {errorStack && (
                <details className="text-[11px] text-[#6B6B66] font-mono mt-1">
                  <summary className="cursor-pointer hover:text-[#111111] transition-colors py-1 select-none">
                    View Component Stack Trace
                  </summary>
                  <pre className="p-3 mt-1.5 rounded-xl bg-[#111111] text-[#A1A1AA] text-[10px] overflow-x-auto whitespace-pre-wrap max-h-48 leading-normal border border-neutral-800">
                    {errorStack}
                  </pre>
                </details>
              )}
            </div>

            {/* Action Bar */}
            <div className="pt-2 border-t border-[#E6E6E2] flex items-center justify-between gap-3">
              <button
                onClick={this.handleReload}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-medium text-[#6B6B66] hover:text-[#111111] hover:bg-[#F1F1EE] border border-transparent hover:border-[#E6E6E2] transition-colors focus:outline-none"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Reload Page</span>
              </button>

              <button
                onClick={this.handleReset}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium bg-[#111111] hover:bg-[#262626] text-white shadow-sm transition-all focus:outline-none"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                <span>Return to Footage Library</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
