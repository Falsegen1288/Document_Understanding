import React, { useState, useEffect, useRef } from 'react';
import { useAnnotationStore } from '../store/annotationStore';
import { DocLayClass, CLASS_COLORS, CLASS_LABELS } from '../types';
import { AnnotationCanvas } from '../components/AnnotationCanvas';
import { EvaluationModal } from '../components/EvaluationModal';
import { 
  Trash2, 
  Eye, 
  EyeOff, 
  Plus, 
  Sparkles, 
  HelpCircle, 
  AlertTriangle, 
  Check, 
  ChevronLeft,
  ChevronRight,
  Activity,
  Save,
  XCircle
} from 'lucide-react';

interface AnnotateProps {
  onNavigateToTab: (tab: 'dashboard' | 'annotate' | 'compare' | 'export') => void;
}

const BACKEND_URL = 'http://127.0.0.1:8000';

export const Annotate: React.FC<AnnotateProps> = ({ onNavigateToTab }) => {
  const store = useAnnotationStore();
  const currentPage = store.pages[store.currentPageIndex];
  
  // Row references for scroll sync
  const rowRefs = useRef<(HTMLDivElement | null)[]>([]);
  
  // Evaluation Modal visibility
  const [isEvalOpen, setIsEvalOpen] = useState(false);
  const [evalTargetModel, setEvalTargetModel] = useState<'DL' | 'NM'>('DL');

  // Base model swapper guard modal local visibility
  const [showResetConfirm, setShowResetConfirm] = useState<boolean>(false);
  const [selectedBaseModel, setSelectedBaseModel] = useState<'DL' | 'NM' | 'blank' | null>(null);

  // Local visibility states of individual detections (by index)
  const [hiddenDetections, setHiddenDetections] = useState<Record<number, boolean>>({});

  // Issue 4: Unsaved changes modal
  const [showUnsavedModal, setShowUnsavedModal] = useState<boolean>(false);
  const [pendingPageIndex, setPendingPageIndex] = useState<number | null>(null);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  // Reset hidden state on page change
  useEffect(() => {
    setHiddenDetections({});
  }, [store.currentPageIndex]);

  // Scroll synced row into view when canvas highlights a bbox
  useEffect(() => {
    if (store.hoveredDetectionIndex !== null && rowRefs.current[store.hoveredDetectionIndex]) {
      rowRefs.current[store.hoveredDetectionIndex]?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
      });
    }
  }, [store.hoveredDetectionIndex]);

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

  // Determine detections current context for canvas drawing
  const getRenderDetections = () => {
    if (store.activeModelTab === 'DL') return currentPage.model_a.detections;
    if (store.activeModelTab === 'NM') return currentPage.model_b.detections;
    return store.workingDetections.filter((_, idx) => !hiddenDetections[idx]);
  };

  const renderDetections = getRenderDetections();

  // Geometric Check for Overlapping Detections
  const computeOverlaps = () => {
    let overlapCount = 0;
    const boxes = store.workingDetections;
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const [ax0, ay0, ax1, ay1] = boxes[i].bbox;
        const [bx0, by0, bx1, by1] = boxes[j].bbox;

        // Intersection Check
        const noOverlap = ax1 < bx0 || bx1 < ax0 || ay1 < by0 || by1 < ay0;
        if (!noOverlap) {
          overlapCount++;
        }
      }
    }
    return overlapCount;
  };

  const overlaps = computeOverlaps();

  // Base model swapper guard
  const handleBaseModelClick = (base: 'DL' | 'NM' | 'blank') => {
    setSelectedBaseModel(base);
    // If working detections have changes or edits, show confirm dialog
    setShowResetConfirm(true);
  };

  const confirmBaseModelSwap = () => {
    if (selectedBaseModel) {
      store.initWorkingDetections(selectedBaseModel);
    }
    setShowResetConfirm(false);
    setSelectedBaseModel(null);
  };

  // Issue 4: Navigation guard — intercept all page changes
  const requestPageChange = (newIndex: number) => {
    if (newIndex === store.currentPageIndex) return;
    
    if (store.isPageDirty(store.currentPageIndex)) {
      setPendingPageIndex(newIndex);
      setShowUnsavedModal(true);
    } else {
      store.forceChangePage(newIndex);
    }
  };

  // Issue 4: Modal actions
  const handleSaveAndContinue = async () => {
    if (pendingPageIndex === null) return;
    setIsSaving(true);
    await store.savePageEdits(store.currentPageIndex);
    setIsSaving(false);
    setShowUnsavedModal(false);
    store.forceChangePage(pendingPageIndex);
    setPendingPageIndex(null);
  };

  const handleDiscardAndContinue = () => {
    if (pendingPageIndex === null) return;
    store.discardPageEdits(store.currentPageIndex);
    setShowUnsavedModal(false);
    store.forceChangePage(pendingPageIndex);
    setPendingPageIndex(null);
  };

  const handleCancelNavigation = () => {
    setShowUnsavedModal(false);
    setPendingPageIndex(null);
  };

  // Page index navigators using guard
  const handlePrevPage = () => {
    if (store.currentPageIndex > 0) {
      requestPageChange(store.currentPageIndex - 1);
    }
  };

  const handleNextPage = () => {
    if (store.currentPageIndex < store.pages.length - 1) {
      requestPageChange(store.currentPageIndex + 1);
    }
  };

  const getPredictedDetsForEval = () => {
    if (evalTargetModel === 'DL') return currentPage.model_a.detections;
    return currentPage.model_b.detections;
  };

  // Issue 3: Get current page's base model selection
  const currentBaseModel = store.baseModelPerPage[store.currentPageIndex] || null;

  // Helper for base model button styling
  const baseModelBtnClass = (model: 'DL' | 'NM' | 'blank') => {
    const isActive = currentBaseModel === model;
    return isActive
      ? "bg-primary/15 hover:bg-primary/25 border-2 border-primary text-primary text-[10px] font-mono px-2 py-1 rounded transition-colors cursor-pointer font-bold shadow-sm"
      : "bg-surface-container-high hover:bg-surface-bright border border-border text-on-surface text-[10px] font-mono px-2 py-1 rounded transition-colors cursor-pointer";
  };

  return (
    <div className="flex-1 flex flex-col md:flex-row overflow-hidden select-none">
      
      {/* PANEL 1: Left Navigation and Class Legend (240px fixed) */}
      <aside className="w-full md:w-[240px] bg-surface-container-low border-b md:border-b-0 md:border-r border-border flex flex-col shrink-0">
        
        {/* Document Selector Selector */}
        <div className="p-4 border-b border-border">
          <label className="font-mono text-[10px] text-text-muted uppercase tracking-wider mb-1 px-1.5 block">Active Document</label>
          <div className="relative">
            <select 
              id="active_document_navigator"
              value={store.currentPageIndex} 
              onChange={(e) => requestPageChange(Number(e.target.value))}
              className="w-full bg-surface-container-high border border-border text-on-surface text-xs font-mono p-2 rounded focus:outline-none focus:ring-1 focus:ring-primary cursor-pointer appearance-none"
            >
              {store.pages.map((p, idx) => (
                <option key={p.page} value={idx}>
                  Page {p.page} {p.ground_truth !== null ? '✓' : '⚠'}
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-text-muted">
              <ChevronRight className="w-4 h-4 transform rotate-90" />
            </div>
          </div>
        </div>

        {/* Scrollable Class Legend Chips */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div>
            <h3 className="font-mono text-[10px] font-bold text-primary uppercase tracking-wider mb-3 px-1.5 border-l border-primary">Class Legend</h3>
            <div className="space-y-1">
              {Object.entries(CLASS_LABELS).map(([cls, label]) => {
                const color = CLASS_COLORS[cls as DocLayClass];
                return (
                  <div 
                    key={cls}
                    className="flex items-center gap-2 text-xs font-mono hover:bg-surface-container px-2 py-1.5 rounded transition-all cursor-crosshair group"
                  >
                    <div className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ backgroundColor: color }} />
                    <span className="text-on-surface-variant group-hover:text-on-surface flex-1 truncate">{label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Lower trigger utilities */}
        <div className="p-4 bg-surface-container-lowest border-t border-border mt-auto flex flex-col gap-2">
          <button 
            onClick={() => onNavigateToTab('compare')}
            className="w-full bg-secondary-container/20 text-on-secondary-container border border-border py-2 text-xs font-mono font-medium tracking-wider uppercase hover:bg-secondary-container/40 transition-all cursor-pointer rounded-xs"
          >
            Launch Comparator
          </button>
          <button 
            onClick={store.confirmPage}
            className="w-full bg-primary text-on-primary py-2 text-xs font-mono font-bold tracking-wider uppercase hover:brightness-110 active:scale-95 transition-all cursor-pointer rounded-xs"
          >
            Save Page Edits
          </button>
        </div>
      </aside>

      {/* PANEL 2: Interactive Active Canvas Workspace */}
      <section className="flex-1 flex flex-col bg-surface overflow-hidden relative">
        
        {/* Navigation Toolbar & Model Tabs overlay */}
        <div className="h-10 border-b border-border bg-surface-container-low px-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button 
              onClick={handlePrevPage}
              disabled={store.currentPageIndex === 0}
              className="p-1 hover:bg-surface-container rounded text-on-surface disabled:opacity-40 disabled:hover:bg-transparent cursor-pointer"
            >
              <ChevronLeft className="w-5 h-5 text-primary" />
            </button>
            <span className="font-mono text-xs text-text-muted">
              LAYER: <span className="text-on-surface">{store.activeModelTab === 'GT' ? 'EDITS CURRENT_GT' : `COMPARE_${store.activeModelTab}_MODEL`}</span>
            </span>
            <button 
              onClick={handleNextPage}
              disabled={store.currentPageIndex === store.pages.length - 1}
              className="p-1 hover:bg-surface-container rounded text-on-surface disabled:opacity-40 disabled:hover:bg-transparent cursor-pointer"
            >
              <ChevronRight className="w-5 h-5 text-primary" />
            </button>
          </div>

          <div className="flex gap-1.5">
            {[
              { id: 'DL', label: 'DocLayoutYOLO' },
              { id: 'NM', label: 'Nemotron' },
              { id: 'GT', label: 'Current GT (Working)' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => store.setActiveModelTab(tab.id as any)}
                className={`px-3 py-1 text-[11px] font-mono tracking-wider transition-all cursor-pointer rounded ${
                  store.activeModelTab === tab.id
                    ? 'bg-primary-container text-on-primary-container font-medium border border-primary/50'
                    : 'text-text-muted hover:text-on-surface bg-surface-container hover:bg-surface-bright border border-transparent'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Base Model Swapper Select overlay banner */}
        <div className="bg-surface-container-high/80 backdrop-blur border-b border-border py-2 px-6 flex items-center justify-between z-10">
          <span className="text-[10px] font-mono text-text-muted flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-primary animate-pulse" />
            SET INITIAL BASE MODEL TO COMPOSE GROUND TRUTH:
          </span>
          <div className="flex gap-2">
            <button 
              onClick={() => handleBaseModelClick('DL')}
              className={baseModelBtnClass('DL')}
            >
              YOLO
            </button>
            <button 
              onClick={() => handleBaseModelClick('NM')}
              className={baseModelBtnClass('NM')}
            >
              Nemotron
            </button>
            <button 
              onClick={() => handleBaseModelClick('blank')}
              className={baseModelBtnClass('blank')}
            >
              Blank Slate
            </button>
          </div>
        </div>

        {/* Core Canvas element */}
        <AnnotationCanvas 
          detections={renderDetections}
          imagePath={currentPage.image_path.startsWith('/') ? `${BACKEND_URL}${currentPage.image_path}` : currentPage.image_path}
        />
      </section>

      {/* PANEL 3: Right Inspector (granular data listing and editing) */}
      <aside className="w-full md:w-[35%] bg-surface-container border-t md:border-t-0 md:border-l border-border flex flex-col shrink-0 overflow-hidden">
        
        {/* Panel Header */}
        <header className="p-4 border-b border-border bg-surface-container-high flex justify-between items-center shrink-0">
          <div>
            <h2 className="font-mono text-xs font-bold text-primary uppercase tracking-wider">Detections Editor</h2>
            <p className="font-mono text-[9px] text-text-muted uppercase tracking-widest mt-1">
              Page {currentPage.page} ({store.workingDetections.length} elements)
              {store.isPageDirty(store.currentPageIndex) && (
                <span className="ml-2 text-warning font-bold">● UNSAVED</span>
              )}
            </p>
          </div>
          
          {/* Diagnostic Icon */}
          <div className="flex gap-2 items-center">
            {store.activeModelTab !== 'GT' && (
              <span className="bg-warning/20 border border-warning/40 text-[9px] text-warning font-mono py-0.5 px-2 uppercase rounded tracking-wider animate-pulse font-bold">
                Read Only
              </span>
            )}
            
            {/* Real-time evaluate page trigger */}
            <div className="flex items-center gap-1 bg-background border border-border px-2 py-1 rounded">
              <select 
                value={evalTargetModel}
                onChange={(e) => setEvalTargetModel(e.target.value as any)}
                className="bg-transparent font-mono text-[9px] text-text-muted uppercase border-none focus:outline-none cursor-pointer"
              >
                <option value="DL" className="bg-surface">vs YOLO</option>
                <option value="NM" className="bg-surface">vs Nemotron</option>
              </select>
              <button 
                onClick={() => setIsEvalOpen(true)}
                title="Run real-time diagnostics check"
                className="p-1 text-primary hover:text-primary-container transition-colors cursor-pointer"
              >
                <Activity className="w-4 h-4" />
              </button>
            </div>
          </div>
        </header>

        {/* Scrollable list inspector cards */}
        <div className="flex-1 overflow-y-auto divide-y divide-border/60">
          {store.workingDetections.map((det, idx) => {
            const isHighlighted = store.hoveredDetectionIndex === idx;
            const isHidden = !!hiddenDetections[idx];

            return (
              <div 
                key={det.id || idx}
                ref={el => { rowRefs.current[idx] = el; }}
                onMouseEnter={() => store.setHoveredDetectionIndex(idx)}
                onMouseLeave={() => store.setHoveredDetectionIndex(null)}
                className={`p-4 transition-colors duration-200 ${
                  isHighlighted 
                    ? 'bg-surface-container-highest border-l-2 border-primary' 
                    : isHidden 
                    ? 'opacity-40 bg-surface-container-lowest/20' 
                    : 'bg-surface-container/20 hover:bg-surface-container-highest/20'
                }`}
              >
                {/* Detection Label Identifier */}
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-xs text-primary font-semibold flex items-center gap-1.5">
                    <span 
                      className="w-2.5 h-2.5 rounded-xs inline-block" 
                      style={{ backgroundColor: CLASS_COLORS[det.type] }}
                    />
                    #{String(idx + 1).padStart(2, '0')} — {det.type.toUpperCase()}
                  </span>
                  
                  {/* Actions buttons */}
                  {store.activeModelTab === 'GT' && (
                    <div className="flex items-center gap-1.5 text-text-muted">
                      {/* Hide Bounding box toggle */}
                      <button 
                        onClick={() => {
                          setHiddenDetections(prev => ({
                            ...prev,
                            [idx]: !prev[idx]
                          }));
                        }}
                        className="p-1 hover:text-on-surface hover:bg-surface border border-transparent rounded cursor-pointer transition-all"
                        title={isHidden ? 'Show layout box' : 'Hide layout box'}
                      >
                        {isHidden ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                      </button>
                      
                      {/* Delete BBox */}
                      <button 
                        onClick={() => store.deleteDetection(idx)}
                        className="p-1 hover:text-error hover:bg-error/10 border border-transparent hover:border-error/20 rounded cursor-pointer transition-all"
                        title="Delete detection"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                </div>

                {/* Coordinators Grid Inputs */}
                <div className="grid grid-cols-2 gap-3 mt-2 font-mono text-[11px]">
                  {/* Class dropdown indicator */}
                  <div className="relative">
                    <select
                      value={det.type}
                      disabled={store.activeModelTab !== 'GT'}
                      onChange={(e) => store.updateDetection(idx, { type: e.target.value as DocLayClass })}
                      className="w-full bg-background hover:bg-surface border border-border text-on-surface text-[11px] p-1.5 rounded cursor-pointer select-none appearance-none focus:outline-none focus:ring-1 focus:ring-primary h-full disabled:cursor-not-allowed"
                    >
                      {Object.keys(CLASS_LABELS).map((cls) => (
                        <option key={cls} value={cls}>
                          {CLASS_LABELS[cls as DocLayClass]}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* 2x2 coordinate details list */}
                  <div className="grid grid-cols-2 gap-1.5">
                    {[
                      { l: 'x0', val: det.bbox[0], fieldIndex: 0 },
                      { l: 'y0', val: det.bbox[1], fieldIndex: 1 },
                      { l: 'x1', val: det.bbox[2], fieldIndex: 2 },
                      { l: 'y1', val: det.bbox[3], fieldIndex: 3 }
                    ].map((coord) => (
                      <div key={coord.l} className="bg-background border border-border flex items-center justify-between px-1.5 py-0.5 rounded">
                        <span className="text-text-muted text-[9px] select-none uppercase">{coord.l}:</span>
                        <input
                          type="number"
                          value={Math.round(coord.val)}
                          disabled={store.activeModelTab !== 'GT'}
                          onChange={(e) => {
                            const nextVal = Number(e.target.value);
                            const nextBbox = [...det.bbox] as [number, number, number, number];
                            nextBbox[coord.fieldIndex] = nextVal;
                            store.updateDetection(idx, { bbox: nextBbox });
                          }}
                          className="w-10 bg-transparent text-right font-semibold text-on-surface hover:text-primary transition-all focus:outline-none font-mono text-[10px] disabled:opacity-85"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}

          {store.workingDetections.length === 0 && (
            <div className="p-8 text-center text-text-muted font-mono text-xs italic">
              NO DETECTIONS CURRENTLY IN WORKSPACE. SELECT BASE MODEL OR DRAW BOXES.
            </div>
          )}

          {/* Add New Detection Action Box */}
          {store.activeModelTab === 'GT' && (
            <div className="p-4 shrink-0 bg-surface-container-lowest/30">
              <button 
                onClick={() => store.addDetection('text', [200, 200, 500, 400])}
                className="w-full py-2.5 border border-dashed border-border text-on-surface-variant flex items-center justify-center gap-2 hover:bg-surface-variant/40 hover:border-primary/50 transition-all font-mono text-xs rounded uppercase tracking-wider cursor-pointer font-bold"
              >
                <Plus className="w-4 h-4 text-primary" /> Add Bounding Box
              </button>
            </div>
          )}
        </div>

        {/* Footer actions block */}
        <div className="p-4 bg-surface-container-low border-t border-border shadow-2xl shrink-0">
          {overlaps > 0 ? (
            <div className="flex items-center gap-1.5 mb-3 text-warning font-mono text-[10px] uppercase font-bold">
              <AlertTriangle className="w-4 h-4 text-warning" strokeWidth={2.5} />
              <span>{overlaps} overlapping layout blocks found</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 mb-3 text-primary font-mono text-[10px] uppercase font-bold">
              <Check className="w-4 h-4 text-primary animate-pulse" />
              <span>Perfect Layout Balance Achieved</span>
            </div>
          )}

          {store.activeModelTab !== 'GT' ? (
            <div className="bg-surface-container-high/60 border border-border p-3 text-center text-xs text-text-muted rounded-xs italic">
              SWITCH TO "CURRENT GT" TAB TO MAKE EDITS
            </div>
          ) : (
            <button 
              id="confirm_page_as_gt_btn"
              onClick={store.confirmPage}
              className="w-full bg-primary text-on-primary py-3 font-mono text-xs font-bold tracking-wider uppercase hover:brightness-110 active:scale-[0.98] transition-all rounded shadow-md cursor-pointer"
            >
              Confirm Page {currentPage.page} as Ground Truth
            </button>
          )}
        </div>
      </aside>

      {/* 4. BASE MODEL INITIALIZATION REQUIRE SURE DIALOG MODAL */}
      {showResetConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="bg-surface border border-border p-6 rounded shadow-2xl max-w-sm w-full font-sans">
            <h3 className="font-mono text-sm font-semibold text-warning flex items-center gap-2 uppercase tracking-wide">
              <AlertTriangle className="w-5 h-5 text-warning" /> Destructive Operation Guard
            </h3>
            <p className="text-sm text-text-muted mt-3 leading-relaxed">
              This will overwrite all active adjustments and restore the page's workspace to the model outputs of {selectedBaseModel === 'blank' ? 'a blank slate' : selectedBaseModel === 'DL' ? 'DocLayoutYOLO' : 'Nemotron'}.
            </p>
            <div className="flex gap-3 mt-6">
              <button 
                onClick={() => { setShowResetConfirm(false); setSelectedBaseModel(null); }}
                className="flex-1 bg-surface-container-high hover:bg-surface-bright border border-border text-on-surface py-2 text-xs font-mono font-semibold uppercase tracking-wider rounded transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button 
                onClick={confirmBaseModelSwap}
                className="flex-1 bg-error text-on-primary py-2 text-xs font-mono font-bold uppercase tracking-wider rounded hover:brightness-110 transition-colors cursor-pointer"
              >
                Overwrite Workspace
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 5. INTERACTIVE DELETE UNDO BANNER TOAST */}
      {store.showUndoToast && (
        <div className="fixed bottom-12 right-6 z-50 bg-surface border border-border p-4 shadow-2xl max-w-md w-full animate-bounce rounded-md flex items-center justify-between">
          <div className="font-mono text-xs text-on-surface max-w-[70%]">
            {store.toastMessage}
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={store.triggerUndo}
              className="font-mono text-xs font-bold text-primary underline uppercase tracking-wider hover:brightness-125 cursor-pointer"
            >
              Undo
            </button>
            <button 
              onClick={store.dismissToast}
              className="font-mono text-xs text-text-muted uppercase tracking-wider hover:text-on-surface cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* 6. REAL-TIME EVALUATION MODAL */}
      <EvaluationModal 
        isOpen={isEvalOpen}
        onClose={() => setIsEvalOpen(false)}
        referenceDetections={store.workingDetections}
        predictedDetections={getPredictedDetsForEval()}
        modelName={evalTargetModel === 'DL' ? 'DocLayoutYOLO' : 'Nemotron-Parse'}
        pageWidth={currentPage.image_size[0]}
        pageHeight={currentPage.image_size[1]}
      />

      {/* 7. ISSUE 4: UNSAVED CHANGES NAVIGATION GUARD MODAL */}
      {showUnsavedModal && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-surface border border-border p-6 rounded-lg shadow-2xl max-w-md w-full font-sans">
            <h3 className="font-mono text-sm font-semibold text-warning flex items-center gap-2 uppercase tracking-wide">
              <AlertTriangle className="w-5 h-5 text-warning" />
              Unsaved Changes on Page {currentPage.page}
            </h3>
            <p className="text-sm text-text-muted mt-3 leading-relaxed">
              You have unsaved edits on the current page. What would you like to do before navigating away?
            </p>
            <div className="flex flex-col gap-2 mt-6">
              <button 
                onClick={handleSaveAndContinue}
                disabled={isSaving}
                className="w-full bg-primary text-on-primary py-2.5 text-xs font-mono font-bold uppercase tracking-wider rounded hover:brightness-110 transition-all cursor-pointer flex items-center justify-center gap-2 disabled:opacity-60"
              >
                <Save className="w-4 h-4" />
                {isSaving ? 'Saving...' : 'Save & Continue'}
              </button>
              <button 
                onClick={handleDiscardAndContinue}
                disabled={isSaving}
                className="w-full bg-error/80 text-on-primary py-2.5 text-xs font-mono font-bold uppercase tracking-wider rounded hover:brightness-110 transition-all cursor-pointer flex items-center justify-center gap-2 disabled:opacity-60"
              >
                <Trash2 className="w-4 h-4" />
                Discard & Continue
              </button>
              <button 
                onClick={handleCancelNavigation}
                disabled={isSaving}
                className="w-full bg-surface-container-high hover:bg-surface-bright border border-border text-on-surface py-2.5 text-xs font-mono font-semibold uppercase tracking-wider rounded transition-colors cursor-pointer flex items-center justify-center gap-2 disabled:opacity-60"
              >
                <XCircle className="w-4 h-4" />
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
