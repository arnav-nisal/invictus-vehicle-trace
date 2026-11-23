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
        <div className="min-h-screen bg-[#F4F5F7] text-slate-900 flex items-center justify-center p-6 font-sans">
          <div className="max-w-2xl w-full bg-white border border-rose-200 rounded-3xl p-8 shadow-bento relative overflow-hidden">
            {/* Top Alert Bar */}
            <div className="flex items-center gap-3 pb-4 border-b border-slate-100 mb-5">
              <div className="w-10 h-10 rounded-2xl bg-rose-100 text-rose-600 flex items-center justify-center font-bold">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div>
                <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 text-[10px] font-bold uppercase tracking-wider font-mono">
                  Runtime Interruption
                </span>
                <h1 className="text-base font-bold text-slate-900 mt-1">
                  Workbench UI Execution Interrupted
                </h1>
              </div>
            </div>

            {/* Error Message */}
            <div className="bg-slate-50 rounded-2xl p-4 border border-slate-200 mb-5">
              <span className="text-[10px] text-slate-500 font-semibold uppercase block mb-1">
                Error Reason:
              </span>
              <p className="text-xs text-rose-700 font-mono font-semibold break-words">
                {this.state.error?.message || 'Unknown runtime error occurred'}
              </p>
            </div>

            {/* Collapsible Stack Trace */}
            {this.state.errorInfo && (
              <details className="mb-5 bg-slate-50 rounded-2xl p-4 border border-slate-200 text-xs text-slate-600">
                <summary className="cursor-pointer text-slate-700 font-semibold mb-2 select-none hover:underline">
                  View Component Stack Details
                </summary>
                <pre className="text-[10px] font-mono text-slate-500 overflow-x-auto whitespace-pre-wrap max-h-48 pt-2 border-t border-slate-200">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <button
                onClick={this.handleReload}
                className="tactile-button px-5 py-2.5 rounded-full bg-slate-900 hover:bg-slate-800 text-white font-semibold text-xs transition active:scale-95 shadow-sm"
              >
                Reload Workbench
              </button>
              <button
                onClick={this.handleClearAndReload}
                className="tactile-button px-5 py-2.5 rounded-full bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold text-xs border border-slate-200 transition active:scale-95"
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
