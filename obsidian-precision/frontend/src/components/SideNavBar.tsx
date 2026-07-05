/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { 
  Layers, Languages, Table, Eye, Activity, Terminal, Plus, 
  Radio, Compass, Home, Sliders, ClipboardList 
} from 'lucide-react';
import { ActiveView } from '../types';

interface SideNavBarProps {
  activeView: ActiveView;
  setActiveView: (view: ActiveView) => void;
  triggerNewAnalysis: () => void;
}

export const SideNavBar: React.FC<SideNavBarProps> = ({
  activeView,
  setActiveView,
  triggerNewAnalysis,
}) => {
  return (
    <aside className="fixed left-0 top-16 bottom-0 w-64 bg-surface-container-low border-r border-r-outline-variant flex flex-col p-4 z-40 select-none">
      {/* Engine Status Head */}
      <div className="flex flex-col gap-1 mb-6 px-2 select-none">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center text-primary border border-primary/20">
            {activeView === 'vision' ? (
              <Eye className="w-5 h-5 text-secondary animate-pulse" />
            ) : activeView === 'landing' ? (
              <Compass className="w-5 h-5 text-cyan-400 rotate-45" />
            ) : (
              <Layers className="w-5 h-5 text-primary" />
            )}
          </div>
          <div>
            <h3 className="font-headline text-sm font-bold text-on-surface leading-tight">
              {activeView === 'landing' ? 'Obsidian Hub' : activeView === 'vision' ? 'Vision Engine' : 'Precision Suite'}
            </h3>
            <p className="text-[10px] custom-font-mono text-secondary tracking-widest font-black uppercase">
              v4.2.0 ACTIVE
            </p>
          </div>
        </div>
      </div>

      {/* Directory Menu Links */}
      <nav className="flex-1 space-y-1 overflow-y-auto custom-scrollbar pr-0.5">
        <button
          onClick={() => setActiveView('landing')}
          className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl font-sans text-xs font-semibold uppercase tracking-wider transition-all duration-150 select-none cursor-pointer ${
            activeView === 'landing'
              ? 'bg-secondary-container dark:bg-secondary-container text-on-secondary-container shadow-lg shadow-secondary/10'
              : 'text-on-surface-variant hover:bg-surface-variant/50 hover:text-on-surface'
          }`}
        >
          <Home className="w-4 h-4 flex-shrink-0 text-cyan-400" />
          <span>Platform Overview</span>
        </button>

        <button
          onClick={() => setActiveView('ingestion')}
          className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl font-sans text-xs font-semibold uppercase tracking-wider transition-all duration-150 select-none cursor-pointer ${
            activeView === 'ingestion'
              ? 'bg-secondary-container dark:bg-secondary-container text-on-secondary-container shadow-lg shadow-secondary/10'
              : 'text-on-surface-variant hover:bg-surface-variant/50 hover:text-on-surface'
          }`}
        >
          <Radio className="w-4 h-4 flex-shrink-0 text-primary animate-pulse" />
          <span>Document Ingestion</span>
        </button>

        <div className="pt-2 pb-1 border-t border-outline-variant/50 my-1 text-[9px] custom-font-mono text-outline font-black tracking-widest uppercase px-4">
          ANALYTICS ENGINES
        </div>

        <button
          onClick={() => setActiveView('layout')}
          className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl font-sans text-xs font-semibold uppercase tracking-wider transition-all duration-150 select-none cursor-pointer ${
            activeView === 'layout'
              ? 'bg-secondary-container dark:bg-secondary-container text-on-secondary-container shadow-lg shadow-secondary/10'
              : 'text-on-surface-variant hover:bg-surface-variant/50 hover:text-on-surface'
          }`}
        >
          <Layers className="w-4 h-4 flex-shrink-0" />
          <span>Layout Segmentation</span>
        </button>

        <button
          onClick={() => setActiveView('ocr')}
          className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl font-sans text-xs font-semibold uppercase tracking-wider transition-all duration-150 select-none cursor-pointer ${
            activeView === 'ocr'
              ? 'bg-secondary-container dark:bg-secondary-container text-on-secondary-container shadow-lg shadow-secondary/10'
              : 'text-on-surface-variant hover:bg-surface-variant/50 hover:text-on-surface'
          }`}
        >
          <Languages className="w-4 h-4 flex-shrink-0" />
          <span>OCR Deep Scan</span>
        </button>

        <button
          onClick={() => setActiveView('table')}
          className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl font-sans text-xs font-semibold uppercase tracking-wider transition-all duration-150 select-none cursor-pointer ${
            activeView === 'table'
              ? 'bg-secondary-container dark:bg-secondary-container text-on-secondary-container shadow-lg shadow-secondary/10'
              : 'text-on-surface-variant hover:bg-surface-variant/50 hover:text-on-surface'
          }`}
        >
          <Table className="w-4 h-4 flex-shrink-0" />
          <span>Table Extraction</span>
        </button>

        <button
          onClick={() => setActiveView('vision')}
          className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl font-sans text-xs font-semibold uppercase tracking-wider transition-all duration-150 select-none cursor-pointer ${
            activeView === 'vision'
              ? 'bg-secondary-container dark:bg-secondary-container text-on-secondary-container shadow-lg shadow-secondary/10'
              : 'text-on-surface-variant hover:bg-surface-variant/50 hover:text-on-surface'
          }`}
        >
          <Eye className="w-4 h-4 flex-shrink-0" />
          <span>Vision Grounding</span>
        </button>
      </nav>

      {/* Action panel & system footers */}
      <div className="mt-auto border-t border-outline-variant pt-4 space-y-3">
        <button
          onClick={triggerNewAnalysis}
          className="w-full bg-secondary text-on-secondary py-3 px-4 rounded-xl custom-font-headline text-xs font-black tracking-widest uppercase flex items-center justify-center gap-2 hover:brightness-110 active:scale-95 transition-all outline-none cursor-pointer"
        >
          <Plus className="w-4 h-4 stroke-[3]" />
          New Analysis
        </button>

        <div className="space-y-1 pt-1">
          <button
            onClick={() => alert(`Obsidian Precision Diagnostics Engine status check:\nAll 6 sub-stages are fully active with green latency parameters.`)}
            className="w-full flex items-center gap-3 text-on-surface-variant px-4 py-2 text-xs font-semibold hover:text-on-surface transition-colors select-none text-left cursor-pointer"
          >
            <Activity className="w-4 h-4 text-emerald-400" />
            <span>Telemetry Check</span>
          </button>
          <button
            onClick={() => alert(`API Blueprint Specs are fully active on secure channel SSL. Use endpoint: /api/v4/spec`)}
            className="w-full flex items-center gap-3 text-on-surface-variant px-4 py-2 text-xs font-semibold hover:text-on-surface transition-colors select-none text-left cursor-pointer"
          >
            <Terminal className="w-4 h-4 text-primary" />
            <span>Blueprint API spec</span>
          </button>
        </div>
      </div>
    </aside>
  );
};
