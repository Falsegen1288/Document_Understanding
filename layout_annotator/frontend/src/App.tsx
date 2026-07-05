import { useState, useEffect } from 'react';
import { useAnnotationStore } from './store/annotationStore';
import { Dashboard } from './pages/Dashboard';
import { Annotate } from './pages/Annotate';
import { Compare } from './pages/Compare';
import { Export } from './pages/Export';
import { AnnotationProvider } from './context/AnnotationContext';
import { 
  FolderOpen, 
  Settings, 
  HelpCircle, 
  User, 
  Activity, 
  FileText,
  Workflow,
  Clock,
  History,
  Terminal,
  Save,
  Database
} from 'lucide-react';

type AppTab = 'dashboard' | 'annotate' | 'compare' | 'export';

function MainAppShell() {
  const store = useAnnotationStore();
  const [activeTab, setActiveTab] = useState<AppTab>('dashboard');
  const [isCommitting, setIsCommitting] = useState(false);

  // Automatically reset tab to dashboard if session is reset
  useEffect(() => {
    if (store.status === 'idle') {
      setActiveTab('dashboard');
    }
  }, [store.status]);

  const handleCommitSession = async () => {
    if (store.status !== 'results' || !store.sessionId) {
      alert('No active session to commit.');
      return;
    }
    setIsCommitting(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/session/${store.sessionId}/commit`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to generate PDF.');
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `ground_truth_${store.sessionId.slice(0, 8)}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`Commit failed: ${err.message}`);
    } finally {
      setIsCommitting(false);
    }
  };

  const hasActiveSession = store.status === 'results' && store.pages.length > 0;

  return (
    <div className="bg-background text-on-surface font-sans selection:bg-primary selection:text-on-primary min-h-screen flex flex-col overflow-hidden">
      
      {/* 1. Global Navigation Top Header */}
      <header className="bg-surface-container text-primary border-b border-border flex justify-between items-center w-full px-4 h-12 shrink-0 z-50">
        <div className="flex items-center gap-4">
          <span className="font-mono text-lg font-extrabold uppercase tracking-tighter text-primary flex items-center gap-1.5">
            <Database className="w-5 h-5 text-primary" />
            MED_ANNOTATOR_V1.2
          </span>
          <div className="hidden sm:block h-6 w-px bg-border mx-2"></div>
          <div className="hidden sm:flex flex-col">
            <span className="font-mono text-[9px] text-text-muted uppercase tracking-wider leading-none">Input Source</span>
            <span className="font-mono text-xs text-on-surface font-bold mt-1 truncate max-w-[160px]">
              {hasActiveSession ? store.uploadedPdfName : 'No Active Session'}
            </span>
          </div>
        </div>

        {/* Header center tabs list */}
        <nav className="flex items-center h-full">
          {[
            { id: 'dashboard', label: 'Dashboard', enabled: true },
            { id: 'annotate', label: 'Dataset (Active Workspace)', enabled: hasActiveSession },
            { id: 'compare', label: 'Validation (Comparator)', enabled: hasActiveSession },
            { id: 'export', label: 'Metrics (Summary & Export)', enabled: hasActiveSession }
          ].map((lnk) => (
            <button
              key={lnk.id}
              disabled={!lnk.enabled}
              onClick={() => setActiveTab(lnk.id as AppTab)}
              className={`h-full px-4 font-mono text-xs uppercase tracking-wider transition-all border-b-2 flex items-center disabled:opacity-40 disabled:hover:bg-transparent disabled:cursor-not-allowed ${
                activeTab === lnk.id
                  ? 'text-primary font-bold border-primary bg-primary/5'
                  : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-bright border-transparent'
              }`}
            >
              {lnk.label}
            </button>
          ))}
        </nav>

        {/* Header Right Actions */}
        <div className="flex items-center gap-3">
          <button 
            onClick={handleCommitSession}
            disabled={!hasActiveSession || isCommitting}
            className="bg-primary-container text-on-primary-container disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-110 active:scale-95 px-4 h-8 text-[11px] font-bold font-mono tracking-wider uppercase transition-all rounded-xs cursor-pointer flex items-center gap-1"
          >
            {isCommitting ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-on-primary-container border-t-transparent rounded-full animate-spin" />
                Generating PDF...
              </>
            ) : (
              <>
                <Save className="w-3.5 h-3.5" />
                Commit Session
              </>
            )}
          </button>
          <div className="h-5 w-px bg-border"></div>
          <div className="flex items-center gap-1.5 text-on-surface-variant">
            <span title="Settings" className="flex"><Settings className="w-4 h-4 hover:text-primary cursor-pointer transition-colors" /></span>
            <span title="Help Guide" className="flex"><HelpCircle className="w-4 h-4 hover:text-primary cursor-pointer transition-colors" /></span>
          </div>
        </div>
      </header>

      {/* 2. Main content arena body */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Side Rail Sidebar Panel (only shown when results exist and NOT on Compare view) */}
        {hasActiveSession && activeTab !== 'compare' && (
          <aside className="hidden md:flex flex-col w-[240px] bg-surface-container-low border-r border-border shrink-0 font-mono text-xs select-none">
            
            {/* Researcher profiling card */}
            <div className="p-4 border-b border-border flex items-center gap-3 bg-surface-container/30">
              <div className="w-8 h-8 rounded bg-surface-container-highest flex items-center justify-center border border-border">
                <User className="w-4 h-4 text-primary" />
              </div>
              <div className="truncate">
                <p className="text-on-surface font-semibold text-xs leading-none">Project Alpha</p>
                <p className="text-text-muted text-[10px] mt-1 uppercase tracking-wider leading-none">Radiology Suite</p>
              </div>
            </div>

            {/* Sidebar selection list */}
            <nav className="flex-1 py-4 space-y-1">
              <div className="flex items-center gap-2.5 px-4 py-2.5 bg-primary/5 text-primary border-l-4 border-primary font-semibold select-none">
                <FolderOpen className="w-4 h-4 shrink-0" />
                <span className="uppercase tracking-wider">Document Tree</span>
              </div>
              <div className="flex items-center gap-2.5 px-4 py-2.5 text-on-surface-variant hover:text-on-surface hover:bg-surface-container/60 cursor-not-allowed select-none transition-colors">
                <Workflow className="w-4 h-4 text-text-muted shrink-0" />
                <span className="uppercase tracking-wider">Automation Pipeline</span>
              </div>
              <div className="flex items-center gap-2.5 px-4 py-2.5 text-on-surface-variant hover:text-on-surface hover:bg-surface-container/60 cursor-not-allowed select-none transition-colors">
                <History className="w-4 h-4 text-text-muted shrink-0" />
                <span className="uppercase tracking-wider">Annotation History</span>
              </div>
              <div className="flex items-center gap-2.5 px-4 py-2.5 text-on-surface-variant hover:text-on-surface hover:bg-surface-container/60 cursor-not-allowed select-none transition-colors">
                <Terminal className="w-4 h-4 text-text-muted shrink-0" />
                <span className="uppercase tracking-wider">Diagnostic Console</span>
              </div>
            </nav>

            {/* Active document footer selector shortcut */}
            <div className="p-4 border-t border-border mt-auto bg-surface-container-lowest/50 text-[11px]">
              <span className="text-text-muted block font-bold uppercase tracking-wider mb-2 font-bold">QUICK FLIP PAGE</span>
              <div className="space-y-1 max-h-36 overflow-y-auto pr-1">
                {store.pages.map((p, idx) => (
                  <button
                    key={p.page}
                    onClick={() => store.changePage(idx)}
                    className={`w-full text-left font-mono truncate py-1.5 px-2 rounded-xs transition-all flex items-center justify-between text-[11px] ${
                      store.currentPageIndex === idx
                        ? 'bg-primary/10 text-primary border border-primary/20 font-bold'
                        : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container border border-transparent'
                    }`}
                  >
                    <span>Page {p.page}</span>
                    {p.ground_truth !== null ? (
                      <span className="text-[9px] bg-primary/20 text-primary px-1 rounded-sm font-bold">✓</span>
                    ) : (
                      <span className="text-[9px] bg-warning/20 text-warning px-1 rounded-sm font-bold">?</span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          </aside>
        )}

        {/* Active Content Panel */}
        <main className="flex-1 flex flex-col overflow-hidden bg-background">
          {activeTab === 'dashboard' && <Dashboard onNavigateToTab={setActiveTab} />}
          {activeTab === 'annotate' && <Annotate onNavigateToTab={setActiveTab} />}
          {activeTab === 'compare' && <Compare onNavigateToTab={setActiveTab} />}
          {activeTab === 'export' && <Export onNavigateToTab={setActiveTab} />}
        </main>
      </div>

      {/* 3. Synchronizing footer strip */}
      <footer className="h-8 bg-surface-container-lowest border-t border-border flex justify-between items-center px-4 shrink-0 font-mono text-[10px] select-none z-50">
        <div className="flex items-center gap-6">
          <span className="text-primary font-bold">© 2026 Clinical Systems Research. Experimental Ground Truth Build.</span>
          <div className="hidden sm:flex gap-4">
            <a href="#" className="text-text-muted hover:text-primary transition-colors">Privacy Protocol</a>
            <a href="#" className="text-text-muted hover:text-primary transition-colors">System Diagnostics</a>
            <a href="#" className="text-text-muted hover:text-primary transition-colors">Swagger API docs</a>
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-primary">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse inline-block shadow-[0_0_8px_#46f1c5]"></span>
          <span className="text-on-surface text-[10px] uppercase font-bold tracking-wider">CONNECTED NODE: US-EAST-ANNOTATE-NODE-01</span>
        </div>
      </footer>

    </div>
  );
}

export default function App() {
  return (
    <AnnotationProvider>
      <MainAppShell />
    </AnnotationProvider>
  );
}
