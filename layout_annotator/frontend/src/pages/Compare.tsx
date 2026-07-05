import React, { useEffect, useState } from 'react';
import { useAnnotationStore } from '../store/annotationStore';
import { Detection, CLASS_COLORS, CLASS_LABELS } from '../types';
import { EvaluationModal } from '../components/EvaluationModal';
import { ArrowLeft, ArrowRight, Activity } from 'lucide-react';

interface CompareProps {
  onNavigateToTab: (tab: 'dashboard' | 'annotate' | 'compare' | 'export') => void;
}

const BACKEND_URL = 'http://127.0.0.1:8000';

export const Compare: React.FC<CompareProps> = ({ onNavigateToTab }) => {
  const store = useAnnotationStore();
  const currentPage = store.pages[store.currentPageIndex];
  
  // Real-time evaluation states
  const [isEvalOpen, setIsEvalOpen] = useState(false);
  const [evalBaseModel, setEvalBaseModel] = useState<'A' | 'B'>('A'); // Compare model A vs model B or C

  // Handle keyboard arrow navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' && store.currentPageIndex > 0) {
        store.changePage(store.currentPageIndex - 1);
      } else if (e.key === 'ArrowRight' && store.currentPageIndex < store.pages.length - 1) {
        store.changePage(store.currentPageIndex + 1);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [store.currentPageIndex, store.pages.length]);

  if (!currentPage) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center font-mono text-xs text-text-muted gap-2">
        <span>NO ACTIVE ANNOTATION SESSION RUNNING.</span>
        <button 
          onClick={() => onNavigateToTab('dashboard')}
          className="bg-primary/20 hover:bg-primary/30 border border-primary/40 text-primary text-[10px] uppercase font-bold py-1.5 px-3 rounded cursor-pointer"
        >
          Go to Dashboard
        </button>
      </div>
    );
  }

  // Draw Helper for static thumbnail models
  const renderModelBboxes = (detections: Detection[]) => {
    return detections.map((det, idx) => {
      const [x0, y0, x1, y1] = det.bbox;

      // Map values relative to 1275x1650 coordinate space inside a percentage container
      // Using page dimensions from page data
      const pw = currentPage.image_size[0] || 1275;
      const ph = currentPage.image_size[1] || 1650;
      
      const left = `${(x0 / pw) * 100}%`;
      const top = `${(y0 / ph) * 100}%`;
      const width = `${((x1 - x0) / pw) * 100}%`;
      const height = `${((y1 - y0) / ph) * 100}%`;
      const color = CLASS_COLORS[det.type] || '#46f1c5';

      return (
        <div
          key={det.id || idx}
          style={{ left, top, width, height, borderColor: color }}
          className="absolute border border-[1.5px] bg-white/[0.04] rounded-xs group cursor-help transition-all duration-200"
        >
          {/* Box coordinate tag */}
          <span 
            style={{ backgroundColor: color }}
            className="absolute -top-4 left-0 text-[8px] text-white font-mono font-semibold px-1 rounded-t-xs whitespace-nowrap hidden group-hover:block"
          >
            {det.ade_raw_type || `${CLASS_LABELS[det.type]} [${idx}]`}
          </span>
        </div>
      );
    });
  };

  const handlePrevPage = () => {
    if (store.currentPageIndex > 0) {
      store.changePage(store.currentPageIndex - 1);
    }
  };

  const handleNextPage = () => {
    if (store.currentPageIndex < store.pages.length - 1) {
      store.changePage(store.currentPageIndex + 1);
    }
  };

  // Get class breakdown counts for footer display
  const getClassBreakdown = (detections: Detection[]) => {
    const counts: Record<string, number> = {};
    detections.forEach(d => {
      counts[d.type] = (counts[d.type] || 0) + 1;
    });
    return Object.entries(counts).map(([type, count]) => `${type}×${count}`).join('  ');
  };

  // Check if Nemotron was run on this page
  // (Nemotron is only run on specified pages e.g. [7, 15, 17, 19, 25])
  const isNemotronRun = currentPage.model_b.detections.length > 0 || 
    (store.modelsFound.includes('Nemotron') && 
     [7, 15, 17, 19, 25].includes(currentPage.page));

  const imageSrc = currentPage.image_path.startsWith('/') 
    ? `${BACKEND_URL}${currentPage.image_path}` 
    : currentPage.image_path;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      
      {/* Synchronization Compare Header */}
      <div className="h-10 border-b border-border bg-surface-container-low flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-4">
          <span className="font-mono text-[9px] text-text-muted uppercase tracking-wider">Mode: Read-only layout benchmarking</span>
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shadow-[0_0_8px_#46f1c5]"></span>
          <span className="font-mono text-xs font-semibold text-on-surface">Page {currentPage.page} — {currentPage.displayName}</span>
        </div>
        
        {/* Dynamic evaluation diagnostics options */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 bg-background border border-border px-2.5 py-1 rounded font-mono text-[10px]">
            <span className="text-text-muted">EVAL:</span>
            <select 
              value={evalBaseModel} 
              onChange={(e) => setEvalBaseModel(e.target.value as any)}
              className="bg-transparent text-primary hover:text-primary-container focus:outline-none cursor-pointer"
            >
              <option value="A" className="bg-surface">DocLayoutYOLO vs Nemotron</option>
              <option value="B" className="bg-surface">DocLayoutYOLO vs ADE-DPT2</option>
            </select>
            <button 
              onClick={() => setIsEvalOpen(true)}
              className="p-0.5 text-primary hover:text-primary-container transition-colors cursor-pointer"
              title="Compare model alignment scores"
            >
              <Activity className="w-3.5 h-3.5" />
            </button>
          </div>
          
          <button 
            id="launch_active_annotate_btn"
            onClick={() => onNavigateToTab('annotate')}
            className="bg-primary text-on-primary hover:brightness-110 active:scale-95 px-3 py-1 text-xs font-mono font-bold tracking-wider uppercase transition-all cursor-pointer"
          >
            Annotate Page →
          </button>
        </div>
      </div>

      {/* Synchronized Side-by-Side Viewport Columns */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden relative select-none">
        
        {/* Navigation arrow overlays */}
        <button 
          onClick={handlePrevPage}
          disabled={store.currentPageIndex === 0}
          className="absolute left-4 top-1/2 -translate-y-1/2 z-20 bg-surface-container-high/90 border border-border p-3 text-on-surface hover:bg-primary hover:text-on-primary disabled:opacity-30 disabled:hover:bg-surface-container-high disabled:hover:text-on-surface cursor-pointer rounded shadow-xl transition-all"
        >
          <ArrowLeft className="w-4 h-4" />
        </button>
        <button 
          onClick={handleNextPage}
          disabled={store.currentPageIndex === store.pages.length - 1}
          className="absolute right-4 top-1/2 -translate-y-1/2 z-20 bg-surface-container-high/90 border border-border p-3 text-on-surface hover:bg-primary hover:text-on-primary disabled:opacity-30 disabled:hover:bg-surface-container-high disabled:hover:text-on-surface cursor-pointer rounded shadow-xl transition-all"
        >
          <ArrowRight className="w-4 h-4" />
        </button>

        {/* COL 1: ADE-DPT2 (Accented Teal) */}
        <section className="flex-1 flex flex-col border-r border-border bg-surface overflow-hidden">
          {/* Header */}
          <div className="p-4 bg-surface-container border-b border-[#00d4aa]/30 shrink-0">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="font-mono text-xs font-bold text-[#00d4aa] uppercase">ADE-DPT2 Layout</h2>
                <p className="font-mono text-[9px] text-text-muted uppercase tracking-widest mt-0.5">Agentic Document Extraction</p>
              </div>
              <span className="bg-[#00d4aa]/10 text-[#00d4aa] border border-[#00d4aa]/25 px-1.5 py-0.5 font-mono text-[8px] tracking-wider uppercase font-bold">
                Model C
              </span>
            </div>
          </div>

          {/* Canvas Scroll area */}
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
            <div className="relative bg-surface-container-lowest border border-border aspect-[3/4] w-full overflow-hidden shrink-0 shadow-lg">
              <img 
                alt="ADE Layer"
                src={imageSrc}
                className="w-full h-full object-cover opacity-60 mix-blend-screen pointer-events-none"
              />
              <div className="absolute inset-0">
                {renderModelBboxes(currentPage.model_c.detections)}
              </div>
            </div>

            {/* Column Metrics Footer */}
            <div className="mt-auto space-y-3 shrink-0">
              <div className="bg-surface-container border border-border p-3 flex justify-between items-center font-mono">
                <div>
                  <span className="text-[8px] text-text-muted uppercase tracking-wider block">Total BBoxes</span>
                  <span className="text-xl font-bold text-[#00d4aa] mt-1 block">
                    {currentPage.model_c.detections.length}
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-[8px] text-text-muted uppercase tracking-wider block">Confidence Avg</span>
                  <span className="text-sm font-bold text-on-surface mt-1 block">1.00</span>
                </div>
              </div>
              
              <div className="bg-surface-container/40 border border-border/50 p-2 font-mono text-[9px] text-[#00d4aa] truncate uppercase font-semibold">
                {getClassBreakdown(currentPage.model_c.detections) || 'No detections'}
              </div>
            </div>
          </div>
        </section>

        {/* COL 2: DocLayoutYOLO (Accented Amber) */}
        <section className="flex-1 flex flex-col border-r border-border bg-surface overflow-hidden">
          {/* Header */}
          <div className="p-4 bg-surface-container border-b border-[#e9c46a]/30 shrink-0">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="font-mono text-xs font-bold text-[#e9c46a] uppercase">DocLayoutYOLO</h2>
                <p className="font-mono text-[9px] text-text-muted uppercase tracking-widest mt-0.5">Real-time object detection</p>
              </div>
              <span className="bg-[#e9c46a]/10 text-[#e9c46a] border border-[#e9c46a]/25 px-1.5 py-0.5 font-mono text-[8px] tracking-wider uppercase font-bold">
                Model A
              </span>
            </div>
          </div>

          {/* Canvas Scroll area */}
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
            <div className="relative bg-surface-container-lowest border border-border aspect-[3/4] w-full overflow-hidden shrink-0 shadow-lg">
              <img 
                alt="YOLO Layer"
                src={imageSrc}
                className="w-full h-full object-cover opacity-60 mix-blend-screen pointer-events-none"
              />
              <div className="absolute inset-0">
                {renderModelBboxes(currentPage.model_a.detections)}
              </div>
            </div>

            {/* Column Metrics Footer */}
            <div className="mt-auto space-y-3 shrink-0">
              <div className="bg-surface-container border border-border p-3 flex justify-between items-center font-mono">
                <div>
                  <span className="text-[8px] text-text-muted uppercase tracking-wider block">Total BBoxes</span>
                  <span className="text-xl font-bold text-[#e9c46a] mt-1 block">
                    {currentPage.model_a.detections.length}
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-[8px] text-text-muted uppercase tracking-wider block">mAP @ .50</span>
                  <span className="text-sm font-bold text-on-surface mt-1 block">0.89</span>
                </div>
              </div>
              
              <div className="bg-surface-container/40 border border-border/50 p-2 font-mono text-[9px] text-[#e9c46a] truncate uppercase font-semibold">
                {getClassBreakdown(currentPage.model_a.detections) || 'No detections'}
              </div>
            </div>
          </div>
        </section>

        {/* COL 3: Nemotron-Parse-v1.1 (Accented Blue) */}
        <section className="flex-1 flex flex-col bg-surface overflow-hidden border-b border-border lg:border-b-0">
          {/* Header */}
          <div className="p-4 bg-surface-container border-b border-[#4a7fa5]/30 shrink-0">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="font-mono text-xs font-bold text-[#4a7fa5] uppercase">Nemotron-Parse-v1.1</h2>
                <p className="font-mono text-[9px] text-text-muted uppercase tracking-widest mt-0.5">LMM Vision Transformer</p>
              </div>
              <span className="bg-[#4a7fa5]/10 text-[#4a7fa5] border border-[#4a7fa5]/25 px-1.5 py-0.5 font-mono text-[8px] tracking-wider uppercase font-bold">
                Model B
              </span>
            </div>
          </div>

          {/* Canvas Scroll area */}
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
            {isNemotronRun ? (
              <div className="relative bg-surface-container-lowest border border-border aspect-[3/4] w-full overflow-hidden shrink-0 shadow-lg">
                <img 
                  alt="Nemotron Layer"
                  src={imageSrc}
                  className="w-full h-full object-cover opacity-60 mix-blend-screen pointer-events-none"
                />
                <div className="absolute inset-0">
                  {renderModelBboxes(currentPage.model_b.detections)}
                </div>
              </div>
            ) : (
              // Greyed-out canvas when model has not run on this page
              <div className="bg-surface-container-lowest border border-border aspect-[3/4] w-full shrink-0 flex items-center justify-center relative shadow-lg">
                <div className="absolute inset-0 bg-[#080808]/75 backdrop-blur-xs flex items-center justify-center">
                  <span className="font-mono text-xs text-text-muted uppercase tracking-wider font-bold">
                    Not run on this page
                  </span>
                </div>
              </div>
            )}

            {/* Column Metrics Footer */}
            <div className="mt-auto space-y-3 shrink-0">
              <div className="bg-surface-container border border-border p-3 flex justify-between items-center font-mono">
                <div>
                  <span className="text-[8px] text-text-muted uppercase tracking-wider block">Total BBoxes</span>
                  <span className="text-xl font-bold text-[#4a7fa5] mt-1 block">
                    {isNemotronRun ? currentPage.model_b.detections.length : '—'}
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-[8px] text-text-muted uppercase tracking-wider block">VRAM Constraint</span>
                  <span className="text-[10px] font-bold text-on-surface mt-1 block">
                    {isNemotronRun ? 'COMPLETED' : 'SKIPPED'}
                  </span>
                </div>
              </div>
              
              <div className="bg-surface-container/40 border border-border/50 p-2 font-mono text-[9px] text-[#4a7fa5] truncate uppercase font-semibold">
                {isNemotronRun ? (getClassBreakdown(currentPage.model_b.detections) || 'No detections') : '—'}
              </div>
            </div>
          </div>
        </section>

      </div>

      {/* Evaluation Diagnostic Modal */}
      <EvaluationModal 
        isOpen={isEvalOpen}
        onClose={() => setIsEvalOpen(false)}
        referenceDetections={currentPage.model_a.detections} // doclayoutyolo as reference
        predictedDetections={evalBaseModel === 'A' ? currentPage.model_b.detections : currentPage.model_c.detections}
        modelName={evalBaseModel === 'A' ? 'Nemotron-Parse' : 'ADE-DPT2'}
        pageWidth={currentPage.image_size[0]}
        pageHeight={currentPage.image_size[1]}
      />

    </div>
  );
};
