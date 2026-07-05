import { create } from 'zustand';
import { PageData, Detection, DocLayClass } from '../types';

interface PipelineLog {
  step: string;
  page?: number;
  model?: string;
  message: string;
  percent: number;
}

type BaseModelChoice = 'DL' | 'NM' | 'blank';

interface AnnotationState {
  // Session details
  sessionId: string | null;
  status: 'idle' | 'processing' | 'results';
  modelsFound: string[];
  pages: PageData[];
  currentPageIndex: number;
  activeModelTab: 'DL' | 'NM' | 'GT';
  workingDetections: Detection[];
  hoveredDetectionIndex: number | null;
  
  // Pipeline status
  isUploading: boolean;
  uploadPercent: number;
  pipelineStep: number; // 1: Upload, 2: Extract, 3: YOLO, 4: Nemotron
  pipelineLogs: string[];
  
  // Intake configuration
  pdfFile: File | null;
  selectedPageRange: string;
  uploadedPdfName: string;
  pdfPageCount: number | null;
  
  // Toast notifications & Undo
  undoStack: { detection: Detection; index: number }[];
  showUndoToast: boolean;
  toastMessage: string;
  
  // Issue 3: Base model selection per page
  baseModelPerPage: Record<number, BaseModelChoice>;
  
  // Issue 4: Dirty state tracking
  dirtyPages: number[];
  savedEditsPerPage: Record<number, Detection[]>;
  
  // Actions
  setPdfFile: (file: File) => void;
  setSelectedPageRange: (range: string) => void;
  startPipeline: () => Promise<void>;
  connectSSE: (sessionId: string) => void;
  loadSessionData: (sessionId: string) => Promise<void>;
  changePage: (index: number) => void;
  forceChangePage: (index: number) => void;
  setActiveModelTab: (tab: 'DL' | 'NM' | 'GT') => void;
  setHoveredDetectionIndex: (idx: number | null) => void;
  initWorkingDetections: (base: BaseModelChoice) => void;
  updateDetection: (idx: number, patch: Partial<Detection>) => void;
  deleteDetection: (idx: number) => void;
  addDetection: (type: DocLayClass, bbox: [number, number, number, number]) => void;
  confirmPage: () => Promise<void>;
  clearPage: () => Promise<void>;
  triggerUndo: () => void;
  dismissToast: () => void;
  importJsonData: (jsonText: string) => string | null;
  exportGroundTruth: () => void;
  resetAllData: () => Promise<void>;
  
  // Issue 4: Save/discard per page
  markPageDirty: () => void;
  savePageEdits: (pageIndex: number) => Promise<void>;
  discardPageEdits: (pageIndex: number) => void;
  isPageDirty: (pageIndex: number) => boolean;
}

const BACKEND_URL = 'http://127.0.0.1:8000';

export const useAnnotationStore = create<AnnotationState>((set, get) => {
  let sseSource: EventSource | null = null;
  
  return {
    // Initial State
    sessionId: null,
    status: 'idle',
    modelsFound: [],
    pages: [],
    currentPageIndex: 0,
    activeModelTab: 'DL',
    workingDetections: [],
    hoveredDetectionIndex: null,
    
    isUploading: false,
    uploadPercent: 0,
    pipelineStep: 0,
    pipelineLogs: [],
    
    pdfFile: null,
    selectedPageRange: '',
    uploadedPdfName: '',
    pdfPageCount: null,
    
    undoStack: [],
    showUndoToast: false,
    toastMessage: '',
    
    // Issue 3
    baseModelPerPage: {},
    
    // Issue 4
    dirtyPages: [],
    savedEditsPerPage: {},
    
    // Setters
    setPdfFile: (file: File) => {
      set({
        pdfFile: file,
        uploadedPdfName: file.name,
        // Mock page count detection from size / name or default (e.g. 30)
        pdfPageCount: 30 
      });
    },
    

    
    setSelectedPageRange: (range: string) => {
      set({ selectedPageRange: range });
    },
    
    // Session pipeline triggers
    startPipeline: async () => {
      const { pdfFile, selectedPageRange } = get();
      if (!pdfFile || !selectedPageRange) return;
      
      set({ 
        isUploading: true, 
        uploadPercent: 0,
        pipelineStep: 1, 
        pipelineLogs: ['[SYSTEM] Initiating file uploads...'] 
      });
      
      // Simulate frontend upload percentage animation
      let progress = 0;
      const uploadTimer = setInterval(() => {
        progress += Math.floor(Math.random() * 15) + 5;
        if (progress >= 100) {
          progress = 100;
          clearInterval(uploadTimer);
          set({ uploadPercent: 100 });
        } else {
          set({ uploadPercent: progress });
        }
      }, 100);
      
      try {
        const formData = new FormData();
        formData.append('pdf', pdfFile);
        formData.append('pages', selectedPageRange);
        
        const response = await fetch(`${BACKEND_URL}/api/session/start`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          throw new Error('Failed to start session pipeline.');
        }
        
        const data = await response.json();
        clearInterval(uploadTimer);
        
        set({
          sessionId: data.session_id,
          modelsFound: data.models_found,
          status: 'processing',
          isUploading: false,
          uploadPercent: 100,
          pipelineStep: 2,
          pipelineLogs: [
            ...get().pipelineLogs,
            `[SYSTEM] Session started: ${data.session_id}`,
            `[SYSTEM] Models detected: ${data.models_found.join(', ')}`,
            `[SYSTEM] Launching sequential evaluation pipeline...`
          ]
        });
        
        // Connect to SSE for status updates
        get().connectSSE(data.session_id);
        
      } catch (err: any) {
        clearInterval(uploadTimer);
        set({
          isUploading: false,
          pipelineLogs: [...get().pipelineLogs, `[ERROR] Failed to start session: ${err.message}`]
        });
      }
    },
    
    connectSSE: (sessionId: string) => {
      if (sseSource) {
        sseSource.close();
      }
      
      sseSource = new EventSource(`${BACKEND_URL}/api/session/${sessionId}/status`);
      
      sseSource.onmessage = (event) => {
        try {
          const logData = JSON.parse(event.data) as PipelineLog;
          
          let stepNum = get().pipelineStep;
          if (logData.step.includes('model_a')) stepNum = 3;
          else if (logData.step.includes('model_b')) stepNum = 4;
          
          set((state) => ({
            pipelineStep: stepNum,
            pipelineLogs: [...state.pipelineLogs, logData.message]
          }));
          if (logData.step === 'complete') {
            if (sseSource) sseSource.close();
            set({ status: 'results' });
            get().loadSessionData(sessionId);
          } else if (logData.step === 'error') {
            if (sseSource) sseSource.close();
            set({ status: 'idle' });
            alert(`Pipeline failed: ${logData.message}`);
          }
        } catch (e) {
          console.error('Failed to parse SSE data', e);
        }
      };
      
      sseSource.onerror = (e) => {
        console.error('SSE Connection error', e);
        if (sseSource) sseSource.close();
        const { status, pipelineLogs } = get();
        if (status === 'processing') {
          set({
            status: 'idle',
            pipelineLogs: [...pipelineLogs, '[ERROR] SSE connection lost. The backend server might be unresponsive or has encountered an error.']
          });
          alert('Backend connection error: The annotation pipeline failed or disconnected. Please check the backend console.');
        }
      };
    },
    
    loadSessionData: async (sessionId: string) => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/session/${sessionId}/data`);
        if (!response.ok) {
          throw new Error('Failed to load session details.');
        }
        
        const data = await response.json();
        // data matches Record<string, PageData>
        const pagesList = Object.values(data) as PageData[];
        
        set({
          pages: pagesList,
          currentPageIndex: 0,
          status: 'results',
          activeModelTab: 'DL'
        });
        
        // Populate first page detections
        if (pagesList.length > 0) {
          const { savedEditsPerPage } = get();
          const firstPage = pagesList[0];
          
          // Issue 4: Restore from saved edits if available
          if (savedEditsPerPage[0]) {
            set({ workingDetections: savedEditsPerPage[0] });
          } else {
            set({
              workingDetections: firstPage.ground_truth !== null 
                ? firstPage.ground_truth 
                : firstPage.model_a.detections
            });
          }
        }
      } catch (err: any) {
        console.error(err);
        alert(`Failed to load session results: ${err.message}`);
      }
    },
    
    // changePage is now a "request" — Annotate.tsx intercepts for dirty guard
    changePage: (index: number) => {
      get().forceChangePage(index);
    },
    
    // Issue 4: Direct page change without guard (called after modal decision)
    forceChangePage: (index: number) => {
      const { pages, savedEditsPerPage } = get();
      if (index >= 0 && index < pages.length) {
        const targetPage = pages[index];
        
        // Issue 4: Restore saved edits if available
        let nextDetections: Detection[];
        if (savedEditsPerPage[index]) {
          nextDetections = savedEditsPerPage[index];
        } else {
          nextDetections = targetPage.ground_truth !== null 
            ? targetPage.ground_truth 
            : targetPage.model_a.detections;
        }
        
        set({
          currentPageIndex: index,
          workingDetections: nextDetections,
          hoveredDetectionIndex: null,
          undoStack: []
        });
      }
    },
    
    setActiveModelTab: (tab: 'DL' | 'NM' | 'GT') => {
      set({ activeModelTab: tab });
    },
    
    setHoveredDetectionIndex: (idx: number | null) => {
      set({ hoveredDetectionIndex: idx });
    },
    
    initWorkingDetections: (base: BaseModelChoice) => {
      const { pages, currentPageIndex } = get();
      const currentPage = pages[currentPageIndex];
      if (!currentPage) return;
      
      let nextDetections: Detection[] = [];
      if (base === 'DL') {
        nextDetections = currentPage.model_a.detections.map(d => ({ ...d, model: 'DocLayoutYOLO' }));
      } else if (base === 'NM') {
        nextDetections = currentPage.model_b.detections.map(d => ({ ...d, model: 'Nemotron-Parse-v1.1' }));
      } else {
        nextDetections = [];
      }
      
      set((state) => ({
        workingDetections: nextDetections,
        activeModelTab: 'GT', // automatically return focus to GT editing
        // Issue 3: Track base model selection per page
        baseModelPerPage: {
          ...state.baseModelPerPage,
          [currentPageIndex]: base
        },
        // Issue 4: Mark page dirty since we replaced detections
        dirtyPages: state.dirtyPages.includes(currentPageIndex)
          ? state.dirtyPages
          : [...state.dirtyPages, currentPageIndex],
        toastMessage: `Initialized workspace from ${base === 'blank' ? 'blank slate' : base}`,
        showUndoToast: true
      }));
      
      setTimeout(() => {
        if (get().toastMessage.includes(`Initialized workspace`)) {
          set({ showUndoToast: false });
        }
      }, 4000);
    },
    
    // Issue 4: Mark current page as dirty
    markPageDirty: () => {
      const { currentPageIndex, dirtyPages } = get();
      if (!dirtyPages.includes(currentPageIndex)) {
        set({ dirtyPages: [...dirtyPages, currentPageIndex] });
      }
    },
    
    updateDetection: (idx: number, patch: Partial<Detection>) => {
      set((state) => {
        const updated = [...state.workingDetections];
        updated[idx] = {
          ...updated[idx],
          ...patch,
          model: 'human' // Mark as human curated
        };
        return { workingDetections: updated };
      });
      // Issue 4: Mark dirty
      get().markPageDirty();
    },
    
    deleteDetection: (idx: number) => {
      const { workingDetections } = get();
      const target = workingDetections[idx];
      if (!target) return;
      
      set((state) => ({
        undoStack: [{ detection: target, index: idx }, ...state.undoStack],
        workingDetections: state.workingDetections.filter((_, i) => i !== idx),
        toastMessage: `Removed detection #${idx + 1} (${target.type})`,
        showUndoToast: true
      }));
      
      // Issue 4: Mark dirty
      get().markPageDirty();
      
      setTimeout(() => {
        if (get().toastMessage.includes(`Removed detection`)) {
          set({ showUndoToast: false });
        }
      }, 4000);
    },
    
    addDetection: (type: DocLayClass, bbox: [number, number, number, number]) => {
      const newDet: Detection = {
        id: `${Date.now()}-${Math.random()}`,
        type,
        bbox,
        model: 'human'
      };
      
      set((state) => ({
        workingDetections: [...state.workingDetections, newDet],
        toastMessage: `Added new ${type} detection`,
        showUndoToast: true
      }));
      
      // Issue 4: Mark dirty
      get().markPageDirty();
      
      setTimeout(() => {
        if (get().toastMessage.includes(`Added new`)) {
          set({ showUndoToast: false });
        }
      }, 4000);
    },
    
    // Issue 4: Save page edits to backend and local cache
    savePageEdits: async (pageIndex: number) => {
      const { sessionId, pages, workingDetections } = get();
      if (!sessionId || pages.length === 0) return;
      
      const currentPage = pages[pageIndex];
      
      try {
        const response = await fetch(`${BACKEND_URL}/api/session/${sessionId}/save-page`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            page: currentPage.page,
            detections: workingDetections.map(d => ({ ...d, model: 'human' }))
          })
        });
        
        if (!response.ok) {
          throw new Error('Failed to save page edits.');
        }
        
        // Store in local cache and clear dirty flag
        set((state) => ({
          savedEditsPerPage: {
            ...state.savedEditsPerPage,
            [pageIndex]: workingDetections.map(d => ({ ...d, model: 'human' }))
          },
          dirtyPages: state.dirtyPages.filter(p => p !== pageIndex),
        }));
        
        // Also update pages array ground_truth
        set((state) => {
          const updatedPages = [...state.pages];
          updatedPages[pageIndex] = {
            ...updatedPages[pageIndex],
            ground_truth: workingDetections.map(d => ({ ...d, model: 'human' }))
          };
          return { pages: updatedPages };
        });
        
      } catch (err: any) {
        alert(`Save failed: ${err.message}`);
      }
    },
    
    // Issue 4: Discard edits, restore from saved or original
    discardPageEdits: (pageIndex: number) => {
      const { pages, savedEditsPerPage } = get();
      const targetPage = pages[pageIndex];
      if (!targetPage) return;
      
      let restored: Detection[];
      if (savedEditsPerPage[pageIndex]) {
        restored = savedEditsPerPage[pageIndex];
      } else {
        restored = targetPage.ground_truth !== null
          ? targetPage.ground_truth
          : targetPage.model_a.detections;
      }
      
      set((state) => ({
        workingDetections: restored,
        dirtyPages: state.dirtyPages.filter(p => p !== pageIndex),
      }));
    },
    
    // Issue 4: Check if a page is dirty
    isPageDirty: (pageIndex: number) => {
      return get().dirtyPages.includes(pageIndex);
    },
    
    confirmPage: async () => {
      const { sessionId, pages, currentPageIndex, workingDetections } = get();
      if (!sessionId || pages.length === 0) return;
      
      const currentPage = pages[currentPageIndex];
      const pageKey = String(currentPage.page);
      
      // Update local pages state
      const updatedPages = [...pages];
      updatedPages[currentPageIndex] = {
        ...currentPage,
        ground_truth: workingDetections.map(d => ({ ...d, model: 'human' }))
      };
      
      set({ pages: updatedPages });
      
      // Post to FastAPI endpoint
      try {
        const requestData: Record<string, { ground_truth: Detection[] }> = {};
        requestData[pageKey] = {
          ground_truth: workingDetections.map(d => ({ ...d, model: 'human' }))
        };
        
        const response = await fetch(`${BACKEND_URL}/api/session/${sessionId}/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pages_data: requestData })
        });
        
        if (!response.ok) {
          throw new Error('Failed to save to backend disk.');
        }
        
        // Also update savedEditsPerPage and clear dirty
        set((state) => ({
          savedEditsPerPage: {
            ...state.savedEditsPerPage,
            [currentPageIndex]: workingDetections.map(d => ({ ...d, model: 'human' }))
          },
          dirtyPages: state.dirtyPages.filter(p => p !== currentPageIndex),
          toastMessage: `Confirmed page ${currentPage.page} ground truth! Saved to disk.`,
          showUndoToast: true
        }));
        
        setTimeout(() => {
          if (get().toastMessage.includes(`Confirmed page`)) {
            set({ showUndoToast: false });
          }
        }, 4000);
      } catch (err: any) {
        alert(`Save failed: ${err.message}`);
      }
    },
    
    clearPage: async () => {
      const { sessionId, pages, currentPageIndex } = get();
      if (!sessionId || pages.length === 0) return;
      
      const currentPage = pages[currentPageIndex];
      const pageKey = String(currentPage.page);
      
      const updatedPages = [...pages];
      updatedPages[currentPageIndex] = {
        ...currentPage,
        ground_truth: null
      };
      
      set({
        pages: updatedPages,
        workingDetections: [],
      });
      
      try {
        const requestData: Record<string, { ground_truth: null }> = {};
        requestData[pageKey] = { ground_truth: null };
        
        await fetch(`${BACKEND_URL}/api/session/${sessionId}/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pages_data: requestData })
        });
        
        set({
          toastMessage: `Unconfirmed page ${currentPage.page} and cleared workspace.`,
          showUndoToast: true
        });
        
        setTimeout(() => {
          if (get().toastMessage.includes(`Unconfirmed page`)) {
            set({ showUndoToast: false });
          }
        }, 4000);
      } catch (err: any) {
        console.error(err);
      }
    },
    
    triggerUndo: () => {
      const { undoStack } = get();
      if (undoStack.length === 0) return;
      
      const [lastItem, ...remainingStack] = undoStack;
      
      set((state) => {
        const restored = [...state.workingDetections];
        restored.splice(lastItem.index, 0, lastItem.detection);
        return {
          undoStack: remainingStack,
          workingDetections: restored,
          showUndoToast: false,
          toastMessage: `Restored detection: ${lastItem.detection.type}`
        };
      });
    },
    
    dismissToast: () => {
      set({ showUndoToast: false });
    },
    
    importJsonData: (jsonText: string): string | null => {
      try {
        const parsed = JSON.parse(jsonText);
        if (typeof parsed !== 'object' || parsed === null) {
          return 'Invalid JSON format: root must be an object.';
        }
        
        // Map simple export back to our store pages
        set((state) => {
          const updatedPages = state.pages.map((p) => {
            const pageKey = String(p.page);
            if (pageKey in parsed) {
              return {
                ...p,
                ground_truth: parsed[pageKey]
              };
            }
            return p;
          });
          
          // Refresh working detections if current page updated
          const currentPage = updatedPages[state.currentPageIndex];
          const nextWorking = currentPage.ground_truth !== null 
            ? currentPage.ground_truth 
            : currentPage.model_a.detections;
            
          return {
            pages: updatedPages,
            workingDetections: nextWorking
          };
        });
        
        return null;
      } catch (e: any) {
        return `Failed to parse JSON: ${e.message || e}`;
      }
    },
    
    exportGroundTruth: () => {
      const { sessionId } = get();
      if (!sessionId) return;
      
      // Request download from backend export route
      const link = document.createElement('a');
      link.href = `${BACKEND_URL}/api/session/${sessionId}/export`;
      link.download = 'ground_truth.json';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    },
    
    resetAllData: async () => {
      const { sessionId } = get();
      if (sessionId) {
        try {
          await fetch(`${BACKEND_URL}/api/session/${sessionId}`, { method: 'DELETE' });
        } catch (e) {
          console.error('Failed to clear session on backend', e);
        }
      }
      
      if (sseSource) {
        sseSource.close();
        sseSource = null;
      }
      
      set({
        sessionId: null,
        status: 'idle',
        modelsFound: [],
        pages: [],
        currentPageIndex: 0,
        activeModelTab: 'DL',
        workingDetections: [],
        hoveredDetectionIndex: null,
        
        isUploading: false,
        uploadPercent: 0,
        pipelineStep: 0,
        pipelineLogs: [],
        
        pdfFile: null,
        selectedPageRange: '',
        uploadedPdfName: '',
        pdfPageCount: null,
        
        undoStack: [],
        showUndoToast: false,
        toastMessage: '',
        
        baseModelPerPage: {},
        dirtyPages: [],
        savedEditsPerPage: {},
      });
    }
  };
});
