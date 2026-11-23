import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error caught by ErrorBoundary:', error, errorInfo);
    this.setState({ error, errorInfo });
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleClearAndReload = () => {
    localStorage.clear();
    sessionStorage.clear();
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-6 font-mono">
          <div className="max-w-2xl w-full bg-slate-900 border-2 border-rose-600 rounded-2xl p-8 shadow-2xl relative overflow-hidden">
            {/* Top Tactical Alert Bar */}
            <div className="flex items-center gap-3 pb-4 border-b border-slate-800 mb-6">
              <div className="w-10 h-10 rounded-lg bg-rose-950 border border-rose-700 flex items-center justify-center text-rose-400">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div>
                <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-300 text-[10px] font-bold uppercase tracking-wider border border-rose-800">
                  CRITICAL RUNTIME EXCEPTION
                </span>
                <h1 className="text-base font-bold text-white mt-1">
                  Workbench UI Execution Interrupted
                </h1>
              </div>
            </div>

            {/* Error Message */}
            <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 mb-6">
              <span className="text-[11px] text-slate-500 uppercase block mb-1">
                ERROR REASON:
              </span>
              <p className="text-sm text-rose-300 font-semibold break-words">
                {this.state.error?.message || 'Unknown runtime error occurred'}
              </p>
            </div>

            {/* Collapsible Stack Trace */}
            {this.state.errorInfo && (
              <details className="mb-6 bg-slate-950/60 rounded-xl p-4 border border-slate-800/80 text-xs text-slate-400">
                <summary className="cursor-pointer text-cyan-400 font-semibold mb-2 select-none hover:underline">
                  View Component Stack Details
                </summary>
                <pre className="text-[11px] text-slate-500 overflow-x-auto whitespace-pre-wrap max-h-48 pt-2 border-t border-slate-800">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <button
                onClick={this.handleReload}
                className="px-5 py-2.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs uppercase tracking-wider transition"
              >
                Reload Workbench
              </button>
              <button
                onClick={this.handleClearAndReload}
                className="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs uppercase tracking-wider border border-slate-700 transition"
              >
                Clear Cache & Reload
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
