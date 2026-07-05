import React, { useState, useEffect, useRef } from 'react';
import { useAnnotationStore } from '../store/annotationStore';
import { 
  Upload, 
  Code, 
  FileText, 
  Settings, 
  Terminal, 
  Play, 
  CheckCircle2, 
  AlertTriangle, 
  Trash2, 
  ChevronRight,
  TrendingUp,
  Layers,
  Cpu,
  Thermometer
} from 'lucide-react';

const BACKEND_URL = 'http://127.0.0.1:8000';

interface DashboardProps {
  onNavigateToTab: (tab: 'dashboard' | 'annotate' | 'compare' | 'export') => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigateToTab }) => {
  const store = useAnnotationStore();
  const terminalEndRef = useRef<HTMLDivElement>(null);
  
  const [pageRangeInput, setPageRangeInput] = useState(store.selectedPageRange);
  const [rangeError, setRangeError] = useState<string | null>(null);
  const [activeStep, setActiveStep] = useState<1 | 2>(1);

  // Auto-scroll terminal logs
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [store.pipelineLogs]);

  // Automatically navigate to active workspace when pipeline succeeds
  useEffect(() => {
    if (store.status === 'results' && store.pages.length > 0) {
      onNavigateToTab('annotate');
    }
  }, [store.status, store.pages.length, onNavigateToTab]);

  // Sync state range input with store
  useEffect(() => {
    setPageRangeInput(store.selectedPageRange);
  }, [store.selectedPageRange]);

  // Page range validation logic
  const handlePageRangeChange = (value: string) => {
    setPageRangeInput(value);
    setRangeError(null);
    
    if (!value.trim()) {
      store.setSelectedPageRange('');
      return;
    }
    
    // Validate character format
    if (!/^[0-9\s,\-]+$/.test(value)) {
      setRangeError("Invalid characters. Use numbers, commas, and hyphens (e.g. 1-5,7).");
      return;
    }
    
    // Validate range expansion boundaries
    const cleanRange = value.replace(/\s+/g, '');
    const parts = cleanRange.split(',');
    const expanded: number[] = [];
    
    for (const part of parts) {
      if (!part) continue;
      if (part.includes('-')) {
        const rangeParts = part.split('-');
        if (rangeParts.length !== 2) {
          setRangeError("Invalid range format. Use 'start-end' (e.g., 7-12).");
          return;
        }
        const start = parseInt(rangeParts[0]);
        const end = parseInt(rangeParts[1]);
        if (isNaN(start) || isNaN(end) || start > end) {
          setRangeError("Invalid range. Start page must be less than or equal to end page.");
          return;
        }
        for (let i = start; i <= end; i++) expanded.push(i);
      } else {
        const val = parseInt(part);
        if (isNaN(val)) {
          setRangeError("Invalid page number.");
          return;
        }
        expanded.push(val);
      }
    }
    
    const pageLimit = store.pdfPageCount || 30;
    const outOfBounds = expanded.filter(p => p < 1 || p > pageLimit);
    if (outOfBounds.length > 0) {
      setRangeError(`Page numbers must be between 1 and the total PDF pages (${pageLimit}).`);
      return;
    }
    
    if (expanded.length > 30) {
      setRangeError("Maximum 30 pages can be selected per annotation session.");
      return;
    }
    
    store.setSelectedPageRange(value);
  };

  const getExpandedPages = () => {
    if (!store.selectedPageRange) return [];
    const cleanRange = store.selectedPageRange.replace(/\s+/g, '');
    const parts = cleanRange.split(',');
    const pages = new Set<number>();
    
    for (const part of parts) {
      if (!part) continue;
      if (part.includes('-')) {
        const [start, end] = part.split('-').map(Number);
        for (let i = start; i <= end; i++) pages.add(i);
      } else {
        pages.add(Number(part));
      }
    }
    return Array.from(pages).sort((a, b) => a - b);
  };

  const expandedPages = getExpandedPages();

  // Drag & Drop handlers for step 1 & 2
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent, type: 'pdf') => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (type === 'pdf' && file.name.endsWith('.pdf')) {
        store.setPdfFile(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'pdf') => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (type === 'pdf') {
        store.setPdfFile(file);
      }
    }
  };

  // Computations for Results
  const totalPages = store.pages.length;
  const confirmedPages = store.pages.filter(p => p.ground_truth !== null).length;
  const progressPercent = totalPages > 0 ? (confirmedPages / totalPages) * 100 : 0;
  
  const totalObjectsDetected = store.pages.reduce((sum, p) => {
    const detections = p.ground_truth || p.model_c.detections;
    return sum + detections.length;
  }, 0);

  return (
    <div className="flex-1 overflow-y-auto">
      
      {/* ================= STATE A: EMPTY (INTAKE SETUP) ================= */}
      {store.status === 'idle' && !store.isUploading && (
        <section className="p-6 flex items-center justify-center min-h-[calc(100vh-5rem)]">
          <div className="w-full max-w-[640px] bg-surface border border-border p-6 rounded-xs shadow-2xl flex flex-col gap-6">
            
            <header className="border-b border-border pb-3">
              <h1 className="font-mono text-lg font-bold text-primary uppercase tracking-wider">New Annotation Session</h1>
              <p className="text-xs text-text-muted mt-1 leading-normal">
                Follow the 2-step setup protocol to run layout predictions on your document.
              </p>
            </header>

            {/* Steps Progress Header */}
            <div className="grid grid-cols-2 gap-2 font-mono text-[10px] uppercase font-semibold">
              <button 
                onClick={() => setActiveStep(1)}
                className={`py-2 text-center border-b-2 transition-all ${
                  activeStep === 1 ? 'border-primary text-primary' : store.pdfFile ? 'border-primary/40 text-on-surface' : 'border-border text-text-muted'
                }`}
              >
                1. PDF File
              </button>
              <button 
                onClick={() => store.pdfFile && setActiveStep(2)}
                disabled={!store.pdfFile}
                className={`py-2 text-center border-b-2 transition-all ${
                  activeStep === 2 ? 'border-primary text-primary' : 'border-border text-text-muted disabled:opacity-50'
                }`}
              >
                2. Page Range
              </button>
            </div>

            {/* Step content */}
            <div className="min-h-56 flex flex-col justify-center">
              
              {/* STEP 1: PDF FILE UPLOAD */}
              {activeStep === 1 && (
                <div className="space-y-4">
                  <div 
                    onDragOver={handleDragOver}
                    onDrop={(e) => handleDrop(e, 'pdf')}
                    className="border-2 border-dashed border-border hover:border-primary/60 bg-surface-container-low/40 rounded-xs py-8 text-center relative group transition-colors"
                  >
                    <input 
                      type="file" 
                      accept=".pdf" 
                      className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                      onChange={(e) => handleFileChange(e, 'pdf')}
                    />
                    <div className="flex flex-col items-center gap-2">
                      <Upload className="w-10 h-10 text-primary group-hover:scale-105 transition-transform" />
                      <span className="font-mono text-xs text-on-surface uppercase tracking-wider font-bold">
                        {store.uploadedPdfName ? 'Replace PDF Document' : 'Upload Medical PDF'}
                      </span>
                      <span className="text-[10px] text-text-muted max-w-[280px] leading-normal">
                        Drag and drop your PDF here or click to browse.
                      </span>
                    </div>
                  </div>
                  
                  {store.pdfFile && (
                    <div className="bg-surface-container border border-border p-3 flex justify-between items-center font-mono text-xs">
                      <div className="flex items-center gap-2 text-primary">
                        <FileText className="w-4 h-4" />
                        <span className="truncate max-w-[320px] font-bold">{store.uploadedPdfName}</span>
                      </div>
                      <span className="text-[10px] text-text-muted">
                        ~{(store.pdfFile.size / (1024 * 1024)).toFixed(2)} MB
                      </span>
                    </div>
                  )}

                  {store.pdfFile && (
                    <button 
                      onClick={() => setActiveStep(2)}
                      className="w-full bg-primary-container text-on-primary-container py-2.5 font-mono text-xs font-bold uppercase tracking-wider hover:brightness-110 transition-all flex items-center justify-center gap-1.5 cursor-pointer"
                    >
                      Step 2: Define Page Range <ChevronRight className="w-4 h-4" />
                    </button>
                  )}
                </div>
              )}

              {/* STEP 2: PAGE RANGE SELECTOR */}
              {activeStep === 2 && (
                <div className="space-y-4">
                  <div className="flex flex-col gap-1.5">
                    <label className="font-mono text-[10px] text-text-muted uppercase tracking-wider font-bold">
                      Enter Page Numbers to Evaluate
                    </label>
                    <input 
                      type="text"
                      value={pageRangeInput}
                      onChange={(e) => handlePageRangeChange(e.target.value)}
                      placeholder="e.g. 7, 15, 17, 19, 25  or  1-5, 10, 20-25"
                      className="bg-surface-container-low border border-border text-xs font-mono p-2.5 rounded text-on-surface focus:outline-none focus:border-primary placeholder-text-muted w-full"
                    />
                    {rangeError && (
                      <span className="text-[10px] text-error font-mono font-semibold">{rangeError}</span>
                    )}
                  </div>

                  {expandedPages.length > 0 && (
                    <div className="space-y-2">
                      <span className="font-mono text-[9px] text-text-muted uppercase font-bold block">
                        Parsed Page Workspace ({expandedPages.length} sheets):
                      </span>
                      <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto pr-1">
                        {expandedPages.map(p => (
                          <div 
                            key={p} 
                            className="bg-primary/10 border border-primary/20 text-primary font-mono text-[10px] px-2 py-0.5 rounded flex items-center gap-1"
                          >
                            Page {p}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <button 
                    onClick={store.startPipeline}
                    disabled={!store.selectedPageRange || !!rangeError}
                    className="w-full bg-primary text-on-primary py-3 font-mono text-xs font-bold uppercase tracking-wider hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-1.5 shadow-lg shadow-primary/20 cursor-pointer"
                  >
                    <Play className="w-3.5 h-3.5" /> Run Layout Models on {expandedPages.length} Pages
                  </button>
                </div>
              )}

            </div>
          </div>
        </section>
      )}

      {/* ================= STATE B: PROCESSING (LIVE PROGRESS) ================= */}
      {(store.status === 'processing' || store.isUploading) && (
        <section className="p-6 max-w-4xl mx-auto flex items-center justify-center min-h-[calc(100vh-5rem)] animate-fade-in">
          <div className="w-full bg-surface border border-border rounded-xs overflow-hidden shadow-2xl flex flex-col">
            
            {/* Header */}
            <div className="bg-surface-container px-4 py-3 border-b border-border flex items-center justify-between">
              <div className="flex items-center gap-2 font-mono text-xs font-bold text-on-surface">
                <Terminal className="w-4 h-4 text-primary animate-pulse" />
                <span className="uppercase tracking-wider">PIPELINE ANALYSIS EXECUTION PROTOCOL</span>
              </div>
              <button 
                onClick={store.resetAllData}
                className="text-text-muted hover:text-error transition-colors font-mono text-[10px] uppercase font-bold cursor-pointer"
              >
                ✕ Cancel Session
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4">
              
              {/* Progress checklist panel */}
              <div className="p-5 bg-surface-container-low/40 border-r border-border space-y-4">
                <span className="text-[10px] text-text-muted font-bold uppercase tracking-wider block">Execution Stages</span>
                
                <div className="space-y-3 font-mono text-[11px]">
                  
                  <div className="flex items-center gap-2">
                    <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] ${
                      store.uploadPercent === 100 ? 'bg-primary text-on-primary' : 'bg-surface-container-high text-text-muted'
                    }`}>
                      {store.uploadPercent === 100 ? '✓' : '1'}
                    </span>
                    <span className={store.uploadPercent === 100 ? 'text-primary font-bold' : 'text-text-muted'}>
                      Upload Stream ({store.uploadPercent}%)
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] ${
                      store.pipelineStep >= 2 ? 'bg-primary text-on-primary' : 'bg-surface-container-high text-text-muted'
                    }`}>
                      {store.pipelineStep > 2 ? '✓' : '2'}
                    </span>
                    <span className={store.pipelineStep >= 2 ? 'text-on-surface' : 'text-text-muted'}>
                      Extract config variables
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] ${
                      store.pipelineStep >= 3 ? 'bg-primary text-on-primary' : 'bg-surface-container-high text-text-muted'
                    }`}>
                      {store.pipelineStep > 3 ? '✓' : '3'}
                    </span>
                    <span className={store.pipelineStep >= 3 ? 'text-on-surface' : 'text-text-muted'}>
                      Run DocLayoutYOLO
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] ${
                      store.pipelineStep >= 4 ? 'bg-primary text-on-primary' : 'bg-surface-container-high text-text-muted'
                    }`}>
                      {store.pipelineStep > 4 ? '✓' : '4'}
                    </span>
                    <span className={store.pipelineStep >= 4 ? 'text-on-surface' : 'text-text-muted'}>
                      Run Nemotron Parse
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] ${
                      store.pipelineStep >= 5 ? 'bg-primary text-on-primary' : 'bg-surface-container-high text-text-muted'
                    }`}>
                      {store.pipelineStep > 5 ? '✓' : '5'}
                    </span>
                    <span className={store.pipelineStep >= 5 ? 'text-on-surface' : 'text-text-muted'}>
                      Align ADE-DPT2
                    </span>
                  </div>
                </div>

                {/* Progress bar */}
                <div className="pt-4 border-t border-border">
                  <div className="h-1 w-full bg-surface-container rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary transition-all duration-300"
                      style={{ 
                        width: `${
                          store.isUploading 
                            ? store.uploadPercent * 0.2 
                            : 20 + ((store.pipelineStep - 1) / 4) * 80
                        }%` 
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Monospace Log console */}
              <div className="p-4 md:col-span-3 bg-black/90 text-primary-container max-h-[320px] min-h-[240px] overflow-y-auto font-mono text-[11px] leading-relaxed space-y-1 select-text">
                {store.pipelineLogs.map((log, index) => (
                  <div key={index} className="flex gap-2">
                    <span className="text-text-muted select-none">[{index + 1}]</span>
                    <span className={
                      log.includes('[SYSTEM]') ? 'text-primary' :
                      log.includes('[ERROR]') ? 'text-error font-bold' :
                      log.includes('[COMPLETE]') ? 'text-primary font-bold animate-pulse' : 'text-on-primary'
                    }>{log}</span>
                  </div>
                ))}
                <div ref={terminalEndRef} />
              </div>

            </div>
          </div>
        </section>
      )}

      {/* ================= STATE C: RESULTS GRID ================= */}
      {store.status === 'results' && store.pages.length > 0 && (
        <div className="animate-fade-in">
          
          {/* Top Bar session indicator */}
          <div className="bg-surface-container border-b border-border py-2 px-6 flex justify-between items-center">
            <div className="flex items-center gap-4 font-mono text-xs">
              <span className="text-primary font-bold">MED_ANNOTATOR_V1.2</span>
              <div className="h-4 w-px bg-border"></div>
              <span className="text-text-muted">SESSION: <strong className="text-on-surface">{store.uploadedPdfName || 'document.pdf'}</strong></span>
              <div className="h-4 w-px bg-border"></div>
              <span className="text-text-muted">PAGES: <strong className="text-on-surface">[{store.selectedPageRange}]</strong></span>
            </div>
            
            <button 
              onClick={store.resetAllData}
              className="bg-surface-container-high hover:bg-surface-bright border border-border text-on-surface font-mono text-[11px] font-bold uppercase tracking-wider px-3.5 py-1.5 rounded transition-all cursor-pointer flex items-center gap-1.5"
            >
              New Session
            </button>
          </div>

          {/* Aggregate Stats Section */}
          <section className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center border-b border-border bg-surface-container-low/40">
            <div className="lg:col-span-4 flex items-center gap-6">
              {/* Progress ring */}
              <div className="relative w-28 h-28 shrink-0">
                <svg className="w-full h-full transform -rotate-90">
                  <circle className="text-surface-container-highest" cx="56" cy="56" fill="transparent" r="48" stroke="currentColor" strokeWidth="6"/>
                  <circle className="text-primary transition-all duration-1000 ease-out" cx="56" cy="56" fill="transparent" r="48" stroke="currentColor" strokeDasharray="301.6" strokeDashoffset={301.6 - (301.6 * progressPercent) / 100} strokeWidth="6"/>
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="font-mono text-2xl font-bold text-on-surface leading-none">{confirmedPages}/{totalPages}</span>
                  <span className="font-mono text-[8px] text-text-muted uppercase tracking-wider mt-1">Confirmed</span>
                </div>
              </div>
              <div>
                <h2 className="font-mono text-sm font-semibold text-primary uppercase tracking-wider">Curation Benchmarking</h2>
                <p className="text-xs text-text-muted mt-2 max-w-xs leading-relaxed">
                  Verify and edit bounding boxes output from DocLayoutYOLO, Nemotron, and ADE-DPT2.
                </p>
              </div>
            </div>

            {/* Bento stats grid */}
            <div className="lg:col-span-8 grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-surface-container border border-border p-4 flex flex-col justify-between">
                <span className="font-mono text-[9px] text-text-muted uppercase tracking-wider">Objects Loaded</span>
                <span className="font-mono text-xl font-bold text-primary mt-2">{totalObjectsDetected}</span>
              </div>
              <div className="bg-surface-container border border-border p-4 flex flex-col justify-between">
                <span className="font-mono text-[9px] text-text-muted uppercase tracking-wider">Pipeline models</span>
                <span className="font-mono text-xl font-bold text-on-surface mt-2">3 Engines</span>
              </div>
              <div className="bg-surface-container border border-border p-4 flex flex-col justify-between">
                <span className="font-mono text-[9px] text-text-muted uppercase tracking-wider">Canvas Res</span>
                <span className="font-mono text-xl font-bold text-on-surface mt-2">150 DPI</span>
              </div>
              <div className="bg-surface-container border border-border p-4 flex flex-col justify-between">
                <span className="font-mono text-[9px] text-text-muted uppercase tracking-wider">Target Pages</span>
                <span className="font-mono text-xl font-bold text-warning mt-2">{totalPages} Total</span>
              </div>
            </div>
          </section>

          {/* Grid Viewer */}
          <section className="p-6 max-w-7xl mx-auto space-y-6">
            <h2 className="font-mono text-xs font-bold text-on-surface uppercase tracking-wider border-l-2 border-primary pl-2">
              Annotated Workspace Sheets
            </h2>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-6">
              {store.pages.map((p, idx) => {
                const isConfirmed = p.ground_truth !== null;
                const dlCount = p.model_a.detections.length;
                const nmCount = p.model_b.detections.length;
                const adeCount = p.model_c.detections.length;

                return (
                  <div 
                    key={p.page}
                    onClick={() => {
                      store.changePage(idx);
                      onNavigateToTab('annotate');
                    }}
                    className="group bg-surface border border-border hover:border-primary/50 transition-all cursor-pointer flex flex-col overflow-hidden"
                  >
                    {/* Thumbnail */}
                    <div className="aspect-[3/4] relative bg-black/50 overflow-hidden flex items-center justify-center border-b border-border/40">
                      {/* Check if image path starts with api to hit backend */}
                      <img 
                        alt={p.displayName}
                        src={p.image_path.startsWith('/') ? `${BACKEND_URL}${p.image_path}` : p.image_path}
                        className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity pointer-events-none"
                        loading="lazy"
                      />
                      
                      {/* Floating status badge */}
                      <div className="absolute top-2 right-2 bg-surface/90 border border-border px-1.5 py-0.5 rounded-xs shadow-md">
                        {isConfirmed ? (
                          <span className="text-[8px] font-mono text-primary font-bold uppercase">CONFIRMED</span>
                        ) : (
                          <span className="text-[8px] font-mono text-warning font-bold uppercase">PENDING</span>
                        )}
                      </div>
                      
                      {/* Bottom Text Overlay */}
                      <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/95 via-black/50 to-transparent">
                        <span className="font-mono text-[9px] text-primary tracking-wider block uppercase font-bold">
                          PAGE {p.page}
                        </span>
                        <span className="font-mono text-[10px] font-semibold text-on-surface block truncate mt-0.5">
                          {p.displayName}
                        </span>
                      </div>
                    </div>

                    {/* Detections summary breakdown */}
                    <div className="p-3 bg-surface-container-low/40 font-mono text-[9px] space-y-1 select-none">
                      <div className="flex justify-between items-center">
                        <span className="text-text-muted">DocLayoutYOLO</span>
                        <span className="text-on-surface font-bold">{dlCount}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-text-muted">Nemotron-Parse</span>
                        <span className="text-on-surface font-bold">{nmCount}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-text-muted">ADE-DPT2</span>
                        <span className="text-on-surface font-bold">{adeCount}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Export bar pinned to bottom */}
          <div className="sticky bottom-0 bg-surface-container border-t border-border py-4 px-6 flex justify-between items-center z-40">
            <span className="font-mono text-xs text-text-muted">
              Finalize ground truth labels to construct benchmark training set.
            </span>
            
            <button 
              onClick={store.exportGroundTruth}
              className="bg-primary text-on-primary font-mono text-xs font-bold uppercase tracking-wider px-5 py-2.5 shadow-lg shadow-primary/20 hover:brightness-110 active:scale-95 transition-all cursor-pointer"
            >
              ⬇ Export ground_truth.json
            </button>
          </div>

        </div>
      )}

    </div>
  );
};
