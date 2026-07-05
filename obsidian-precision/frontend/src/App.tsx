/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { ActiveView, RunItem } from './types';
import { INITIAL_RUNS } from './data';
import { TopNavBar } from './components/TopNavBar';
import { SideNavBar } from './components/SideNavBar';
import { OCRAnalysisView } from './components/OCRAnalysisView';
import { TableExtractionView } from './components/TableExtractionView';
import { VisionEngineView } from './components/VisionEngineView';
import { LandingPageView } from './components/LandingPageView';
import { IngestionView } from './components/IngestionView';
import { LayoutSegmentationView } from './components/LayoutSegmentationView';

export default function App() {
  const [activeView, setActiveView] = useState<ActiveView>('landing');
  const [runs, setRuns] = useState<RunItem[]>(() => {
    // Attempt local storage synchronization for persistence guidelines
    const saved = localStorage.getItem('obsidian_precision_runs');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return INITIAL_RUNS;
      }
    }
    return INITIAL_RUNS;
  });

  const [systemStatus, setSystemStatus] = useState<string>('SYSTEM_READY');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  // Push run list updates to localStorage on change
  useEffect(() => {
    localStorage.setItem('obsidian_precision_runs', JSON.stringify(runs));
  }, [runs]);

  // Resume any stuck running runs in the background on mount
  useEffect(() => {
    const runningRuns = runs.filter(r => r.status === 'running');
    if (runningRuns.length > 0) {
      setSystemStatus('PIPELINE_PENDING_JOBS');
    }
    
    runningRuns.forEach(run => {
      if (run.jobId) {
        // This is a real backend job — start polling for it
        pollJobProgress(run.id, run.jobId);
      } else {
        // Legacy/simulation run — auto-complete it
        let currentProgress = run.progress;
        const interval = setInterval(() => {
          currentProgress += Math.floor(5 + Math.random() * 12);
          if (currentProgress >= 100) {
            currentProgress = 100;
            clearInterval(interval);
            setRuns((prev) =>
              prev.map((r) =>
                r.id === run.id
                  ? {
                      ...r,
                      status: 'done',
                      progress: 100,
                      duration: 'Duration: 5.4s',
                      message: r.message || 'Legacy layout pipeline recovered and parsed into active memory.',
                    }
                  : r
              )
            );
            setSystemStatus('SYSTEM_READY');
          } else {
            setRuns((prev) =>
              prev.map((r) =>
                r.id === run.id
                  ? {
                      ...r,
                      progress: currentProgress,
                      duration: `${((currentProgress * 5.4) / 100).toFixed(1)}s elapsed`,
                    }
                  : r
              )
            );
          }
        }, 500);
      }
    });
  }, []);

  // Poll a specific backend job for progress updates every 3 seconds
  const pollJobProgress = (runKey: string, jobId: string) => {
    const startTime = Date.now();

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;

        const job = await res.json();
        const total = job.total_pages || 1;
        const done = job.pages_done || 0;
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

        if (job.status === 'done') {
          clearInterval(interval);
          const finalElapsed = job.completed_at && job.created_at
            ? Math.round((new Date(job.completed_at).getTime() - new Date(job.created_at).getTime()) / 1000)
            : Math.round((Date.now() - startTime) / 1000);

          setRuns((prev) =>
            prev.map((r) =>
              r.id === runKey
                ? {
                    ...r,
                    status: 'done' as const,
                    progress: 100,
                    duration: `Duration: ${finalElapsed}s`,
                    message: `Successfully processed ${total} pages. Full extraction pipeline completed.`,
                  }
                : r
            )
          );
          setSystemStatus('SYSTEM_READY');
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setRuns((prev) =>
            prev.map((r) =>
              r.id === runKey
                ? {
                    ...r,
                    status: 'failed' as const,
                    progress: 100,
                    duration: `Terminated @ ${elapsed}s`,
                    message: job.error_message || 'Pipeline execution failed.',
                  }
                : r
            )
          );
          setSystemStatus('SYSTEM_READY');
        } else {
          // Still running — update progress
          const progress = Math.min(Math.round((done / total) * 100), 99);
          setRuns((prev) =>
            prev.map((r) =>
              r.id === runKey
                ? {
                    ...r,
                    progress: progress > 0 ? progress : r.progress,
                    duration: `${elapsed}s elapsed`,
                    message: done > 0 ? `Page ${done}/${total} processed` : r.message,
                  }
                : r
            )
          );
        }
      } catch (err) {
        console.error('Poll error:', err);
      }
    }, 3000);
  };

  // Method to launch a real pipeline execution or simulation fallback
  const startAnalysis = async (
    fileName: string, 
    selectedAlgos: string[],
    algoSpec?: { layout_algo: string; ocr_algo: string; table_algo: string; figure_algo: string },
    file?: File
  ) => {
    // Generate randomized unique run identifiers
    const runIdNum = Math.floor(1000 + Math.random() * 9000);
    const suffix = String.fromCharCode(65 + Math.floor(Math.random() * 26));
    const runId = `#${runIdNum}-${suffix}`;
    const runKey = `run-${Date.now()}`;

    const newRun: RunItem = {
      id: runKey,
      runId,
      fileName,
      status: 'running',
      progress: 0,
      duration: '0s elapsed',
      timestamp: new Date().toISOString(),
      layoutAlgo: algoSpec?.layout_algo,
      ocrAlgo: algoSpec?.ocr_algo,
      tableAlgo: algoSpec?.table_algo,
      figureAlgo: algoSpec?.figure_algo,
    };

    setRuns((prev) => [newRun, ...prev]);
    setSystemStatus('PIPELINE_PENDING_JOBS');

    if (file) {
      // ═══════════════════════════════════════════════
      // REAL BACKEND PIPELINE — Upload PDF + Poll Progress
      // ═══════════════════════════════════════════════
      try {
        // 1. Upload PDF to backend
        const formData = new FormData();
        formData.append('file', file);
        formData.append('doc_type', 'auto');
        formData.append('layout_algo', algoSpec?.layout_algo || 'doclayout_yolo');
        formData.append('ocr_algo', algoSpec?.ocr_algo || 'easyocr');
        formData.append('table_algo', algoSpec?.table_algo || 'docling_tableformer');
        formData.append('figure_algo', algoSpec?.figure_algo || 'groq');

        const uploadRes = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });

        if (!uploadRes.ok) {
          let errorMsg = 'Upload failed';
          try { const ed = await uploadRes.json(); errorMsg = ed.detail || errorMsg; } catch {}
          throw new Error(errorMsg);
        }

        const uploadData = await uploadRes.json();
        const jobId = uploadData.job_id;

        // Store jobId on the run and start polling
        setRuns((prev) =>
          prev.map((r) => r.id === runKey ? { ...r, jobId } : r)
        );

        // Auto-select the newly created job
        setSelectedJobId(jobId);

        // 2. Start polling for progress every 3 seconds
        pollJobProgress(runKey, jobId);

      } catch (uploadErr: any) {
        setRuns((prev) =>
          prev.map((r) =>
            r.id === runKey
              ? {
                  ...r,
                  status: 'failed' as const,
                  progress: 100,
                  duration: 'Failed',
                  message: uploadErr.message || 'Upload to backend failed.',
                }
              : r
          )
        );
        setSystemStatus('SYSTEM_READY');
      }
    } else {
      // ═══════════════════════════════════════════════════
      // SIMULATION FALLBACK — For preset files without real File objects
      // ═══════════════════════════════════════════════════
      setSelectedJobId(null);
      let currentProgress = 0;
      const interval = setInterval(() => {
        currentProgress += Math.floor(5 + Math.random() * 15);
        if (currentProgress >= 100) {
          currentProgress = 100;
          clearInterval(interval);
          setRuns((prev) =>
            prev.map((r) =>
              r.runId === runId
                ? {
                    ...r,
                    status: 'done' as const,
                    progress: 100,
                    duration: 'Duration: 4.8s',
                    message: `Simulated ${selectedAlgos.length} analytical algorithms with preset configuration.`,
                  }
                : r
            )
          );
          setSystemStatus('SYSTEM_READY');
        } else {
          setRuns((prev) =>
            prev.map((r) =>
              r.runId === runId
                ? {
                    ...r,
                    progress: currentProgress,
                    duration: `${((currentProgress * 4.8) / 100).toFixed(1)}s elapsed`,
                  }
                : r
            )
          );
        }
      }, 450);
    }
  };

  // Trigger quick preset simulation workflow instantly
  const triggerPresetSimulation = (fileName: string) => {
    // Determine typical sub-algorithms and their configuration
    let algos = ['segmentation', 'ocr'];
    let algoSpec = {
      layout_algo: 'doclayout_yolo',
      ocr_algo: 'easyocr',
      table_algo: 'tatr',
      figure_algo: 'groq_llama',
    };

    if (fileName.includes('AUDIT')) {
      algos.push('table');
      algoSpec.table_algo = 'tatr';
    } else if (fileName.includes('TX-091')) {
      algos = ['segmentation', 'ocr', 'linking'];
      algoSpec.ocr_algo = 'easyocr';
      algoSpec.figure_algo = 'groq_llama';
    } else {
      algoSpec.layout_algo = 'doclayout_yolo';
      algoSpec.ocr_algo = 'tesseract';
    }

    startAnalysis(fileName, algos, algoSpec);
  };


  const triggerNewAnalysis = () => {
    // Focus or route user to ingestion view and highlight upload parameters selection
    setActiveView('ingestion');
  };

  // Helper method to resolve active view layout contents
  const renderActiveView = () => {
    switch (activeView) {
      case 'landing':
        return (
          <LandingPageView 
            setActiveView={setActiveView} 
            runsCount={runs.length} 
          />
        );
      case 'ingestion':
        return (
          <IngestionView 
            runs={runs} 
            startAnalysis={startAnalysis}
            triggerPresetSimulation={triggerPresetSimulation}
            selectedJobId={selectedJobId}
            onSelectJob={setSelectedJobId}
          />
        );
      case 'layout':
        return <LayoutSegmentationView jobId={selectedJobId} />;
      case 'ocr':
        return <OCRAnalysisView jobId={selectedJobId} />;
      case 'table':
        return <TableExtractionView jobId={selectedJobId} />;
      case 'vision':
        return <VisionEngineView jobId={selectedJobId} />;
      default:
        return (
          <LandingPageView 
            setActiveView={setActiveView} 
            runsCount={runs.length} 
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-background text-on-surface flex flex-col font-sans transition-colors duration-200">
      <TopNavBar 
        activeView={activeView} 
        setActiveView={setActiveView} 
        systemStatus={systemStatus}
      />
      
      <div className="flex flex-1 pt-16">
        <SideNavBar 
          activeView={activeView} 
          setActiveView={setActiveView} 
          triggerNewAnalysis={triggerNewAnalysis}
        />
        
        {/* Workspace Central Board and main content panels */}
        <main className="flex-1 ml-64 p-8 overflow-y-auto max-w-[1720px] custom-scrollbar">
          {renderActiveView()}
        </main>
      </div>
    </div>
  );
}
