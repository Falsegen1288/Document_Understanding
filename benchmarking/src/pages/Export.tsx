import React, { useState } from 'react';
import { useAnnotationStore } from '../store/annotationStore';
import { CLASS_COLORS, CLASS_LABELS, DocLayClass } from '../types';
import { 
  Download, 
  AlertTriangle, 
  CheckCircle, 
  FileJson, 
  X, 
  ExternalLink,
  Search,
  ClipboardCheck,
  Clipboard
} from 'lucide-react';

interface ExportProps {
  onNavigateToTab: (tab: 'dashboard' | 'annotate' | 'compare' | 'export') => void;
}

export const Export: React.FC<ExportProps> = ({ onNavigateToTab }) => {
  const store = useAnnotationStore();

  const [filterText, setFilterText] = useState('');
  const [dismissWarning, setDismissWarning] = useState(false);
  const [copied, setCopied] = useState(false);

  // JSON Import States
  const [importText, setImportText] = useState('');
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  // Computations
  const totalPages = store.pages.length;
  const confirmedPagesList = store.pages.filter(p => p.ground_truth !== null);
  const confirmedCount = confirmedPagesList.length;
  const unconfirmedCount = totalPages - confirmedCount;
  const unconfirmedPagesNumbers = store.pages.filter(p => p.ground_truth === null).map(p => p.page);

  // Count detections across confirmed GT or fallbacks (ADE Model C)
  const totalDetections = store.pages.reduce((sum, p) => {
    const list = p.ground_truth || p.model_c.detections;
    return sum + list.length;
  }, 0);

  // Calculate dynamic class distribution
  const computeClassDistribution = () => {
    const counts: Record<DocLayClass, number> = {
      title: 0,
      section_header: 0,
      text: 0,
      list_item: 0,
      table: 0,
      picture: 0,
      caption: 0,
      footnote: 0,
      formula: 0,
      page_header: 0,
      page_footer: 0,
    };

    let total = 0;
    store.pages.forEach(p => {
      const detections = p.ground_truth || p.model_c.detections;
      detections.forEach(d => {
        if (d.type in counts) {
          counts[d.type]++;
          total++;
        }
      });
    });

    return { counts, total };
  };

  const { counts: classCounts, total: classTotal } = computeClassDistribution();

  // Filter Table Rows
  const filteredPages = store.pages.filter(p => {
    if (!filterText) return true;
    const txt = filterText.toLowerCase();
    const sourceLabel = p.ground_truth ? 'manual_curation' : 'ade_dpt2';
    return (
      p.displayName.toLowerCase().includes(txt) ||
      sourceLabel.includes(txt) ||
      `page-${p.page}`.includes(txt)
    );
  });

  const handleImportSubmit = () => {
    setImportError(null);
    if (!importText.trim()) {
      setImportError('Please paste some valid JSON content first.');
      return;
    }

    const err = store.importJsonData(importText);
    if (err) {
      setImportError(err);
    } else {
      setShowImportDialog(false);
      setImportText('');
    }
  };

  // Clipboard copy handler
  const handleCopyToClipboard = () => {
    const output: Record<string, any> = {};
    store.pages.forEach(p => {
      output[String(p.page)] = p.ground_truth;
    });
    
    navigator.clipboard.writeText(JSON.stringify(output, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex-1 overflow-y-auto p-6">
      
      {/* 1. Amber Warning Banner (if pages are missing ground_truth labels) */}
      {!dismissWarning && unconfirmedCount > 0 && (
        <div className="bg-warning/10 border border-warning/40 p-4 mb-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 rounded-xs">
          <div className="flex items-center gap-3 text-warning">
            <AlertTriangle className="w-5 h-5 text-warning shrink-0" strokeWidth={2.5} />
            <span className="font-mono text-xs font-bold uppercase tracking-wider">
              {unconfirmedCount} pages ({unconfirmedPagesNumbers.map(n => `#${n}`).join(', ')}) have no ground truth confirmed yet — export anyway?
            </span>
          </div>
          <div className="flex gap-4">
            <button 
              onClick={() => {
                const pendingIdx = store.pages.findIndex(p => p.ground_truth === null);
                if (pendingIdx !== -1) {
                  store.changePage(pendingIdx);
                  onNavigateToTab('annotate');
                }
              }}
              className="text-warning font-mono text-xs uppercase underline hover:opacity-80 font-bold cursor-pointer"
            >
              Review Missing
            </button>
            <button 
              onClick={() => setDismissWarning(true)}
              className="text-warning font-mono text-xs uppercase hover:opacity-80 font-bold cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* 2. Page Header Title and Download Action */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-end gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold font-sans text-on-surface tracking-tight uppercase">Export &amp; Summary Report</h1>
          <p className="text-xs text-text-muted font-mono uppercase mt-1 tracking-wider">
            SESSION ID: <span className="text-primary font-bold">{store.sessionId?.slice(0, 8) || 'N/A'}</span> | BATCH: 04-B
          </p>
        </div>
        <div className="flex gap-2 w-full lg:w-auto">
          <button 
            id="open_import_dialog_btn"
            onClick={() => {
              setImportError(null);
              setShowImportDialog(true);
            }}
            className="flex-1 lg:flex-none bg-surface-container-high hover:bg-surface-bright border border-border text-on-surface px-4 py-2.5 rounded-xs font-bold font-mono text-xs uppercase tracking-wider flex items-center justify-center gap-2 transition-all cursor-pointer"
          >
            <FileJson className="w-4 h-4 text-secondary" />
            Import JSON Dataset
          </button>
          <button 
            id="trigger_download_json_btn"
            onClick={store.exportGroundTruth}
            className="flex-1 lg:flex-none bg-primary text-on-primary px-6 py-2.5 rounded-xs hover:brightness-110 flex items-center justify-center gap-2 font-bold font-mono text-xs uppercase tracking-wider shadow-lg shadow-primary/20 transition-all cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Download ground_truth.json
          </button>
        </div>
      </div>

      {/* 3. Stats Section */}
      <div className="grid grid-cols-12 gap-6 mb-8">
        
        {/* Class Distributions Stacked bar */}
        <div className="col-span-12 lg:col-span-5 bg-surface-container p-5 border border-border flex flex-col justify-between">
          <div className="flex justify-between items-center mb-4 pb-2 border-b border-border/40">
            <h3 className="font-mono text-[10px] text-text-muted uppercase tracking-widest font-bold">Class Distribution Matrix</h3>
            <span className="font-mono text-[9px] text-primary">{classTotal} total labels</span>
          </div>
          
          <div className="overflow-y-auto max-h-48 pr-1 space-y-2.5">
            {Object.entries(classCounts).map(([cls, count]) => {
              const share = classTotal > 0 ? (count / classTotal) * 100 : 0;
              if (count === 0) return null;
              return (
                <div key={cls} className="flex items-center justify-between font-mono text-xs">
                  <div className="flex items-center gap-2">
                    <span 
                      className="w-2 h-2 rounded-xs" 
                      style={{ backgroundColor: CLASS_COLORS[cls as DocLayClass] }} 
                    />
                    <span className="text-on-surface-variant truncate uppercase max-w-[120px] text-[10px]">{CLASS_LABELS[cls as DocLayClass]}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-20 bg-background h-1 rounded-full overflow-hidden shrink-0">
                      <div 
                        className="h-full rounded-full" 
                        style={{ width: `${share}%`, backgroundColor: CLASS_COLORS[cls as DocLayClass] }}
                      />
                    </div>
                    <span className="text-on-surface font-semibold shrink-0 min-w-[32px] text-right text-[10px]">{Math.round(share)}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Aggregate details boxes */}
        <div className="col-span-12 lg:col-span-7 grid grid-cols-1 md:grid-cols-3 gap-6">
          
          <div className="bg-surface-container p-5 border border-border flex flex-col justify-between">
            <h3 className="font-mono text-[10px] text-text-muted uppercase tracking-widest font-semibold">Total Verified Labels</h3>
            <div className="mt-4">
              <div className="text-3xl font-mono font-bold text-primary">{totalDetections}</div>
              <div className="text-[9px] text-primary/60 font-mono mt-1 uppercase tracking-wider">+12% over model baseline</div>
            </div>
          </div>

          <div className="bg-surface-container p-5 border border-border flex flex-col justify-between">
            <h3 className="font-mono text-[10px] text-text-muted uppercase tracking-widest font-semibold">Confirmed Ground Truth</h3>
            <div className="mt-4">
              <div className="text-3xl font-mono font-bold text-on-surface">
                {confirmedCount}<span className="text-base text-text-muted">/{totalPages}</span>
              </div>
              <div className="w-full bg-surface-container-highest h-1.5 mt-3 rounded-full overflow-hidden">
                <div 
                  className="bg-primary h-full rounded-full transition-all duration-500" 
                  style={{ width: `${(confirmedCount / totalPages) * 100}%` }}
                />
              </div>
            </div>
          </div>

          <div className="bg-surface-container p-5 border border-border flex flex-col justify-between">
            <h3 className="font-mono text-[10px] text-text-muted uppercase tracking-widest font-semibold">Pending Verification</h3>
            <div className="mt-4">
              <div className={`text-3xl font-mono font-bold ${unconfirmedCount > 0 ? 'text-warning' : 'text-primary'}`}>
                {String(unconfirmedCount).padStart(2, '0')}
              </div>
              <div className="flex gap-1.5 mt-3">
                <span className={`w-2 h-2 rounded-full ${unconfirmedCount > 0 ? 'bg-warning animate-pulse' : 'bg-primary'}`}></span>
                <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider">{unconfirmedCount > 0 ? 'Needs Attention' : 'Cleared & Secured'}</span>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* 4. Dataset Breakdown Summary Table */}
      <div className="bg-surface-container border border-border rounded-xs overflow-hidden">
        
        {/* Table Toolbar */}
        <div className="px-6 py-4 border-b border-border bg-surface-container-high flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <h3 className="font-mono text-xs text-on-surface uppercase font-bold tracking-widest">Validated Dataset Breakdown</h3>
          <div className="flex gap-2 w-full md:w-auto">
            <div className="relative flex-1 md:flex-initial">
              <input 
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                className="bg-surface-container-lowest border border-border text-xs font-mono px-3 py-1.5 pl-8 text-on-surface rounded-xs focus:outline-none focus:border-primary placeholder-text-muted w-full md:w-56" 
                placeholder="FILTER BY SOURCE OR DISPLAY..." 
                type="text"
              />
              <Search className="w-3.5 h-3.5 text-text-muted absolute left-2.5 top-1/2 -translate-y-1/2" />
            </div>
            
            <button 
              onClick={handleCopyToClipboard}
              className="bg-surface-container-lowest border border-border px-3 py-1.5 hover:bg-surface-container-high text-on-surface rounded-xs transition-colors flex items-center justify-center gap-1.5 cursor-pointer font-mono text-[11px] font-bold uppercase"
            >
              {copied ? <ClipboardCheck className="w-3.5 h-3.5 text-primary" /> : <Clipboard className="w-3.5 h-3.5" />}
              {copied ? 'Copied!' : 'Copy JSON'}
            </button>
          </div>
        </div>

        {/* Detailed Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs border-collapse">
            <thead>
              <tr className="text-text-muted border-b border-border bg-surface-container-high/40 uppercase text-[9px] tracking-wider select-none">
                <th className="px-6 py-3 font-semibold">Page #</th>
                <th className="px-6 py-3 font-semibold">Source Base</th>
                <th className="px-6 py-3 font-semibold text-right"># Verified Labels</th>
                <th className="px-6 py-3 font-semibold">Class Breakdown</th>
                <th className="px-6 py-3 font-semibold">Verification status</th>
                <th className="px-6 py-3 font-semibold text-right">Labeling Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filteredPages.map((pageData) => {
                const isConfirmed = pageData.ground_truth !== null;
                const detections = pageData.ground_truth || pageData.model_c.detections;

                // Simple mini representation of categories as colored dots
                const classColorsList = detections.map(d => CLASS_COLORS[d.type] || '#888888');

                return (
                  <tr 
                    key={pageData.page}
                    className={`hover:bg-surface-container-highest/20 transition-colors group ${
                      !isConfirmed ? 'bg-warning/[0.02]' : ''
                    }`}
                  >
                    <td className="px-6 py-4 text-on-surface-variant font-semibold">
                      Page {String(pageData.page).padStart(2, '0')}
                    </td>
                    <td className="px-6 py-4">
                      {isConfirmed ? (
                        <span className="bg-primary/10 border border-primary/20 text-primary text-[10px] px-2 py-0.5 rounded-sm uppercase tracking-wider font-semibold">
                          MANUAL_CURATION
                        </span>
                      ) : (
                        <span className="bg-surface-container-highest text-text-muted text-[10px] px-2 py-0.5 rounded-sm uppercase tracking-wider">
                          ADE_DPT2_PROXY
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-right font-medium">
                      {detections.length}
                    </td>
                    <td className="px-6 py-4">
                      {/* Tiny colored dots representing layout elements */}
                      <div className="flex flex-wrap gap-1 max-w-[200px]">
                        {classColorsList.slice(0, 15).map((color, idx) => (
                          <span 
                            key={idx} 
                            style={{ backgroundColor: color }} 
                            className="w-1.5 h-1.5 rounded-full inline-block shrink-0" 
                          />
                        ))}
                        {classColorsList.length > 15 && (
                          <span className="text-[8px] text-text-muted leading-none font-bold">+{classColorsList.length - 15}</span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      {isConfirmed ? (
                        <span className="flex items-center gap-1.5 text-primary text-[10px] font-semibold tracking-wider">
                          <CheckCircle className="w-3.5 h-3.5" />
                          CONFIRMED
                        </span>
                      ) : (
                        <span className="flex items-center gap-1.5 text-warning text-[10px] font-semibold tracking-wider">
                          <AlertTriangle className="w-3.5 h-3.5 animate-pulse" />
                          MISSING_GT
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button
                        onClick={() => {
                          const idx = store.pages.findIndex(p => p.page === pageData.page);
                          store.changePage(idx);
                          onNavigateToTab('annotate');
                        }}
                        className="text-text-muted group-hover:text-primary transition-colors cursor-pointer"
                        title="Edit Page"
                      >
                        <ExternalLink className="w-4 h-4 inline" />
                      </button>
                    </td>
                  </tr>
                );
              })}

              {filteredPages.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-text-muted italic">
                    No page files matched your criteria filter.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* JSON Import Dialog Popup Modal */}
      {showImportDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 animate-fade-in">
          <div className="bg-surface border border-border p-6 rounded-xs shadow-2xl max-w-lg w-full flex flex-col gap-4">
            <div className="flex justify-between items-center pb-2 border-b border-border">
              <h3 className="font-mono text-sm font-semibold text-primary flex items-center gap-2 uppercase tracking-wide">
                <FileJson className="w-5 h-5 text-secondary" /> Import ground_truth.json
              </h3>
              <button 
                onClick={() => setShowImportDialog(false)}
                className="text-text-muted hover:text-on-surface cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-xs text-text-muted leading-relaxed">
              Paste the structured labeling JSON schema (mapping page index strings to arrays of detections) to import ground truth validation runs:
            </p>

            <textarea
              id="import_json_textarea"
              value={importText}
              onChange={(e) => setImportText(e.target.value)}
              placeholder='{ "7": [ { "type": "title", "bbox": [120, 52, 420, 104], "model": "human" } ] }'
              className="bg-background border border-border text-xs font-mono p-3 rounded-xs h-48 focus:outline-none focus:border-primary text-on-surface overflow-y-auto"
            />

            {importError && (
              <div className="bg-error/15 border border-error/30 text-error p-3 rounded-xs text-xs font-mono">
                {importError}
              </div>
            )}

            <div className="flex gap-3 mt-2 justify-end">
              <button 
                onClick={() => setShowImportDialog(false)}
                className="bg-surface-container-high hover:bg-surface-bright border border-border text-on-surface px-4 py-2 text-xs font-mono font-semibold uppercase tracking-wider rounded-xs transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button 
                id="submit_import_json_btn"
                onClick={handleImportSubmit}
                className="bg-primary text-on-primary px-5 py-2 text-xs font-mono font-bold uppercase tracking-wider rounded-xs hover:brightness-110 transition-colors cursor-pointer"
              >
                Load Coordinates
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
