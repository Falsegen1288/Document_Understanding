import React, { createContext, useContext } from 'react';
import { useAnnotationStore } from '../store/annotationStore';
import { PageData, Detection } from '../types';

interface AnnotationContextType {
  pages: PageData[];
  currentPageIndex: number;
  activeModelTab: 'DL' | 'NM' | 'GT';
  workingDetections: Detection[];
  hoveredDetectionIndex: number | null;
  undoStack: { detection: Detection; index: number }[];
  showUndoToast: boolean;
  toastMessage: string;

  // Actions
  changePage: (pageIndex: number) => void;
  setActiveModelTab: (tab: 'DL' | 'NM' | 'GT') => void;
  setHoveredDetectionIndex: (idx: number | null) => void;
  initWorkingDetections: (base: 'DL' | 'NM' | 'blank') => void;
  updateDetection: (idx: number, patch: Partial<Detection>) => void;
  deleteDetection: (idx: number) => void;
  addDetection: (type: Detection['type'], bbox: Detection['bbox']) => void;
  confirmPage: () => void;
  clearPage: () => void;
  triggerUndo: () => void;
  dismissToast: () => void;
  importJsonData: (jsonText: string) => string | null;
  exportGroundTruth: () => void;
  resetAllData: () => void;
}

const AnnotationContext = createContext<AnnotationContextType | undefined>(undefined);

export const AnnotationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const store = useAnnotationStore();

  const contextValue: AnnotationContextType = {
    pages: store.pages,
    currentPageIndex: store.currentPageIndex,
    activeModelTab: store.activeModelTab,
    workingDetections: store.workingDetections,
    hoveredDetectionIndex: store.hoveredDetectionIndex,
    undoStack: store.undoStack,
    showUndoToast: store.showUndoToast,
    toastMessage: store.toastMessage,

    changePage: store.changePage,
    setActiveModelTab: store.setActiveModelTab,
    setHoveredDetectionIndex: store.setHoveredDetectionIndex,
    initWorkingDetections: store.initWorkingDetections,
    updateDetection: store.updateDetection,
    deleteDetection: store.deleteDetection,
    addDetection: store.addDetection,
    confirmPage: store.confirmPage,
    clearPage: store.clearPage,
    triggerUndo: store.triggerUndo,
    dismissToast: store.dismissToast,
    importJsonData: store.importJsonData,
    exportGroundTruth: store.exportGroundTruth,
    resetAllData: store.resetAllData,
  };

  return (
    <AnnotationContext.Provider value={contextValue}>
      {children}
    </AnnotationContext.Provider>
  );
};

export const useAnnotation = () => {
  const context = useContext(AnnotationContext);
  if (!context) {
    throw new Error('useAnnotation must be used within an AnnotationProvider');
  }
  return context;
};
