/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { 
  Upload, CheckSquare, Square, RefreshCw, AlertTriangle, Play, 
  CheckCircle, Radio, Sparkles, FileText, Check, Loader2, Info, ChevronDown, CheckCircle2
} from 'lucide-react';
import { RunItem } from '../types';

interface IngestionViewProps {
  runs: RunItem[];
  startAnalysis: (
    fileName: string, 
    selectedAlgos: string[],
    algoSpec?: { layout_algo: string; ocr_algo: string; table_algo: string; figure_algo: string },
    file?: File
  ) => void;
  triggerPresetSimulation: (presetKey: string) => void;
  selectedJobId?: string | null;
  onSelectJob?: (jobId: string | null) => void;
}

export const IngestionView: React.FC<IngestionViewProps> = ({
  runs,
  startAnalysis,
  triggerPresetSimulation,
  selectedJobId,
  onSelectJob,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Staged File state
  const [stagedFile, setStagedFile] = useState<{ name: string; size: number; type: string } | null>(null);
  const [actualFile, setActualFile] = useState<File | null>(null);

  // Analytical engine preset mode
  const [presetOption, setPresetOption] = useState<'all' | 'structure_ocr' | 'custom'>('all');

  // Manual configuration selection state
  const [layoutEnabled, setLayoutEnabled] = useState(true);
  const [layoutAlgo, setLayoutAlgo] = useState<'doclayout_yolo' | 'nemotron_parse' | 'landingai_ade'>('doclayout_yolo');

  const [ocrEnabled, setOcrEnabled] = useState(true);
  const [ocrAlgo, setOcrAlgo] = useState<'easyocr' | 'tesseract'>('easyocr');

  const [tableEnabled, setTableEnabled] = useState(true);
  const [tableAlgo, setTableAlgo] = useState<'tatr' | 'docling_tableformer'>('docling_tableformer');

  const [figureEnabled, setFigureEnabled] = useState(true);
  const [figureAlgo, setFigureAlgo] = useState<'groq_llama' | 'groq_qwen' | 'local_qwen' | 'local_moondream'>('groq_llama');


  const getStepState = (activeRun: RunItem, key: string): 'completed' | 'processing' | 'pending' | 'skipped' => {
    const progress = activeRun.progress;

    if (key === 'ingest') {
      return progress > 0 ? 'completed' : 'processing';
    }

    // Check if skipped in active pipeline runs
    let isEnabled = true;
    if (key === 'segmentation' && !activeRun.layoutAlgo) isEnabled = false;
    if (key === 'ocr' && !activeRun.ocrAlgo) isEnabled = false;
    if (key === 'table' && !activeRun.tableAlgo) isEnabled = false;
    if (key === 'linking' && !activeRun.figureAlgo) isEnabled = false;

    if (!isEnabled) {
      return 'skipped';
    }

    if (key === 'segmentation') {
      if (progress === 0) return 'pending';
      if (progress > 0 && progress <= 25) return 'processing';
      return 'completed';
    }

    if (key === 'ocr') {
      if (progress <= 25) return 'pending';
      if (progress > 25 && progress <= 50) return 'processing';
      return 'completed';
    }

    if (key === 'table') {
      if (progress <= 50) return 'pending';
      if (progress > 50 && progress <= 75) return 'processing';
      return 'completed';
    }

    if (key === 'linking') {
      if (progress <= 75) return 'pending';
      if (progress > 75 && progress < 100) return 'processing';
      return 'completed';
    }

    return 'pending';
  };

  const handlePresetChange = (val: 'all' | 'structure_ocr' | 'custom') => {
    setPresetOption(val);
    if (val === 'all') {
      setLayoutEnabled(true);
      setLayoutAlgo('doclayout_yolo');
      setOcrEnabled(true);
      setOcrAlgo('easyocr');
      setTableEnabled(true);
      setTableAlgo('docling_tableformer');
      setFigureEnabled(true);
      setFigureAlgo('groq_llama');
    } else if (val === 'structure_ocr') {
      setLayoutEnabled(true);
      setLayoutAlgo('doclayout_yolo');
      setOcrEnabled(true);
      setOcrAlgo('easyocr');
      setTableEnabled(false);
      setTableAlgo('docling_tableformer');
      setFigureEnabled(false);
      setFigureAlgo('groq_llama');
    }
  };


  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      setActualFile(files[0]);
      handleFileSelected(files[0].name, files[0].size, files[0].type);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setActualFile(files[0]);
      handleFileSelected(files[0].name, files[0].size, files[0].type);
    }
  };

  const handleFileSelected = (fileName: string, size?: number, type?: string) => {
    setStagedFile({
      name: fileName,
      size: size || Math.floor(250000 + Math.random() * 8000000),
      type: type || 'application/pdf',
    });
  };

  const handleClearStaged = (e: React.MouseEvent) => {
    e.stopPropagation();
    setStagedFile(null);
    setActualFile(null);
  };

  const handleExecutePipeline = () => {
    if (!stagedFile) return;

    const activeAlgos: string[] = [];
    if (layoutEnabled) activeAlgos.push('segmentation');
    if (ocrEnabled) activeAlgos.push('ocr');
    if (tableEnabled) activeAlgos.push('table');
    if (figureEnabled) activeAlgos.push('linking');

    const algoSpec = {
      layout_algo: layoutEnabled ? layoutAlgo : '',
      ocr_algo: ocrEnabled ? ocrAlgo : '',
      table_algo: tableEnabled ? tableAlgo : '',
      figure_algo: figureEnabled ? figureAlgo : '',
    };

    startAnalysis(stagedFile.name, activeAlgos, algoSpec, actualFile || undefined);
    setStagedFile(null);
    setActualFile(null);
  };

  const handlePresetClick = (preset: { name: string; display: string; tag: string }) => {
    setStagedFile({
      name: preset.name,
      size: preset.name.includes('AUDIT') ? 859231 : preset.name.includes('SCHEMATIC') ? 3140592 : 1248592,
      type: 'application/pdf',
    });

    setPresetOption('custom');
    if (preset.name.includes('AUDIT')) {
      setLayoutEnabled(true);
      setLayoutAlgo('doclayout_yolo');
      setOcrEnabled(true);
      setOcrAlgo('easyocr');
      setTableEnabled(true);
      setTableAlgo('tatr');
      setFigureEnabled(false);
    } else if (preset.name.includes('TX-091')) {
      setLayoutEnabled(true);
      setLayoutAlgo('doclayout_yolo');
      setOcrEnabled(true);
      setOcrAlgo('easyocr');
      setTableEnabled(false);
      setFigureEnabled(true);
      setFigureAlgo('groq_llama');
    } else {
      setLayoutEnabled(true);
      setLayoutAlgo('doclayout_yolo');
      setOcrEnabled(true);
      setOcrAlgo('tesseract');
      setTableEnabled(false);
      setFigureEnabled(false);
    }

  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getFileTypeLabel = (name: string, type: string) => {
    if (type.includes('pdf') || name.toLowerCase().endsWith('.pdf')) return 'PDF Document';
    if (type.includes('png') || name.toLowerCase().endsWith('.png')) return 'PNG Image';
    if (type.includes('jpeg') || type.includes('jpg') || name.toLowerCase().endsWith('.jpg') || name.toLowerCase().endsWith('.jpeg')) return 'JPEG Image';
    if (type.includes('json') || name.toLowerCase().endsWith('.json')) return 'JSON Keymap Schema';
    return type || 'Binary Stream';
  };

  const presetFiles = [
    { name: 'SOURCE_DOC_089.PDF', display: 'Surgical Actuator Blueprint', tag: 'Acoustics Spec' },
    { name: 'FIN-Q4-AUDIT_v2.pdf', display: 'Q4 Audits Corporate Spec', tag: 'Table Metrics' },
    { name: 'TX-091_SCHEMATIC.PDF', display: 'Robotic Joint Linkage System', tag: 'Vision Grounding' },
  ];

  // Find active running run
  const activeRun = runs.find((r) => r.status === 'running');

  return (
    <div className="grid grid-cols-12 gap-6 items-start animate-fadeIn select-none">
      {/* File Ingestion Stage (Left/Middle 8 columns) */}
      <section className="col-span-12 lg:col-span-8 flex flex-col gap-6">
        <div className="flex justify-between items-end">
          <div>
            <h1 className="font-headline text-3xl font-black tracking-tight text-primary">
              Document Ingestion Node
            </h1>
            <p className="text-on-surface-variant text-sm mt-1 font-sans font-medium">
              Deploy and marshal raw unstructured PDF/schematic files into active memory parsing pipelines.
            </p>
          </div>
          <div className="flex gap-2 text-outline-variant font-black">
            <span className="px-3 py-1 rounded bg-surface-container text-primary font-mono text-[9px] tracking-wider border border-outline-variant flex items-center gap-1.5 shadow-sm">
              <Radio className="w-3 h-3 text-primary animate-pulse" />
              <span>INGESTION_ALPHA</span>
            </span>
          </div>
        </div>

        {/* 1. Dynamic horizontal stepper if an active run is running */}
        {activeRun ? (
          <div className="bg-[#030712] border border-primary/25 p-6 rounded-2xl glass-panel-glow shadow-md flex flex-col gap-6 animate-fadeIn relative overflow-hidden">
            <div className="absolute inset-0 bg-primary/2 opacity-10 pointer-events-none" />
            <div className="flex justify-between items-center pb-3 border-b border-outline-variant/50 relative z-10">
              <div>
                <span className="px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20 text-[9px] font-mono font-extrabold uppercase tracking-wide">
                  Active Pipeline Execution
                </span>
                <h3 className="font-headline text-lg font-black text-on-surface truncate max-w-[280px] sm:max-w-md mt-1.5" title={activeRun.fileName}>
                  {activeRun.fileName}
                </h3>
              </div>
              <div className="text-right">
                <span className="text-[9px] font-mono tracking-widest text-outline uppercase font-bold">Overall progress</span>
                <h3 className="font-mono text-xl font-black text-secondary mt-0.5 animate-pulse">
                  {activeRun.progress}%
                </h3>
              </div>
            </div>

            {/* Micro progress line */}
            <div className="h-2 w-full bg-slate-950 border border-slate-900 rounded-full overflow-hidden relative z-10">
              <div 
                className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-300 ease-out"
                style={{ width: `${activeRun.progress}%` }}
              />
            </div>

            {/* Detailed Horizontal Stepper */}
            <div className="flex flex-col md:flex-row items-center md:items-start justify-between w-full relative gap-6 md:gap-2 pt-2 pb-2 z-10 select-none">
              {[
                { name: 'File Ingested', key: 'ingest' },
                { name: 'Layout Segmentation', key: 'segmentation' },
                { name: 'OCR Transcription', key: 'ocr' },
                { name: 'Table Parsing', key: 'table' },
                { name: 'Vision Grounding', key: 'linking' },
              ].map((step, idx, arr) => {
                const isLast = idx === arr.length - 1;
                const state = getStepState(activeRun, step.key);
                const isCompleted = state === 'completed';
                const isProcessing = state === 'processing';
                const isPending = state === 'pending';
                const isSkipped = state === 'skipped';

                let circleClass = "";
                let textClass = "";
                let indicatorText = "";
                let icon = null;

                if (isCompleted) {
                  circleClass = "bg-emerald-950/40 border-emerald-500 text-emerald-400 border-2 shadow-[0_0_12px_rgba(16,185,129,0.3)]";
                  textClass = "text-emerald-400 font-bold";
                  indicatorText = "Completed";
                  icon = <Check className="w-5 h-5 stroke-[3]" />;
                } else if (isProcessing) {
                  circleClass = "bg-cyan-950/40 border-cyan-400 text-cyan-400 border-2 shadow-[0_0_15px_rgba(34,211,238,0.45)] animate-pulse";
                  textClass = "text-cyan-400 font-extrabold";
                  indicatorText = "Processing...";
                  icon = <Loader2 className="w-4 h-4 animate-spin text-cyan-400 stroke-[2.5]" />;
                } else if (isSkipped) {
                  circleClass = "bg-slate-900/30 border-slate-800 text-slate-500 border-dashed border-2 opacity-50";
                  textClass = "text-slate-500 font-medium line-through decoration-slate-700 decoration-1";
                  indicatorText = "Skipped";
                  icon = <span className="text-[10px] font-mono leading-none">-</span>;
                } else {
                  circleClass = "bg-slate-950 border-slate-800 text-slate-600 border-2 opacity-60";
                  textClass = "text-slate-600 font-medium";
                  indicatorText = "Queued";
                  icon = <span className="text-xs font-mono font-bold">{idx + 1}</span>;
                }

                return (
                  <React.Fragment key={step.key}>
                    <div className="flex flex-col items-center flex-1 text-center group">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${circleClass}`}>
                        {icon}
                      </div>
                      
                      <span className={`text-[10px] font-headline mt-3 transition-colors ${textClass}`}>
                        {step.name}
                      </span>
                      
                      <span className="text-[8px] font-mono uppercase tracking-wider text-outline mt-0.5">
                        {indicatorText}
                      </span>
                    </div>

                    {!isLast && (
                       <div className="hidden md:block flex-1 h-[2.5px] self-start mt-5 relative min-w-[15px]">
                        <div className={`absolute inset-0 rounded transition-all duration-300 ${
                          state === 'completed' && getStepState(activeRun, arr[idx + 1].key) !== 'skipped'
                            ? 'bg-gradient-to-r from-emerald-500 to-cyan-400'
                            : getStepState(activeRun, arr[idx + 1].key) === 'skipped'
                            ? 'border-t-2 border-dashed border-slate-800 bg-transparent'
                            : state === 'completed' && getStepState(activeRun, arr[idx + 1].key) === 'skipped'
                            ? 'border-t-2 border-dashed border-slate-800 bg-transparent'
                            : state === 'processing'
                            ? 'bg-gradient-to-r from-cyan-400 to-slate-850 animate-pulse'
                            : 'bg-slate-850'
                        }`} />
                      </div>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        ) : stagedFile ? (
          /* Staged File Details Card View */
          <div className="relative w-full rounded-2xl bg-[#030712] border border-primary/20 p-6 flex flex-col gap-4 shadow-xl animate-fadeIn font-sans font-medium">
            <div className="absolute inset-0 bg-primary/2 opacity-10 pointer-events-none rounded-2xl" />
            <div className="flex items-start gap-4 relative z-10">
              <div className="w-12 h-12 rounded-xl bg-primary/10 border border-primary/30 flex items-center justify-center text-primary flex-shrink-0 shadow-[0_0_12px_rgba(34,211,238,0.15)] animate-pulse">
                <FileText className="w-6 h-6 text-primary" />
              </div>
              <div className="flex-grow min-w-0">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded-full text-[8px] font-mono tracking-wider font-extrabold uppercase bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                    Validated &amp; Staged
                  </span>
                </div>
                <p className="font-headline text-base font-black text-on-surface truncate break-all mt-1.5" title={stagedFile.name}>
                  {stagedFile.name}
                </p>
                <div className="mt-1 flex items-center gap-3 text-xs text-on-surface-variant font-medium">
                  <span>Type: <strong className="text-on-surface">{getFileTypeLabel(stagedFile.name, stagedFile.type)}</strong></span>
                  <span className="text-outline-variant">•</span>
                  <span>Size: <strong className="text-on-surface">{formatFileSize(stagedFile.size)}</strong></span>
                </div>
              </div>
              
              <button 
                onClick={handleClearStaged}
                className="px-3 py-1.5 border border-outline-variant text-[10px] font-mono font-bold tracking-wider text-on-surface-variant hover:text-error hover:border-error/40 rounded-lg transition-colors cursor-pointer focus:outline-none"
              >
                Change File
              </button>
            </div>

            <div className="border-t border-outline-variant/60 pt-4 mt-2 relative z-10 w-full">
              <button 
                onClick={handleExecutePipeline}
                className="w-full bg-gradient-to-r from-primary to-secondary text-on-primary font-headline text-xs font-black tracking-widest uppercase py-3.5 rounded-xl hover:brightness-110 shadow-[0_0_20px_rgba(34,211,238,0.35)] hover:shadow-[0_0_25px_rgba(34,211,238,0.55)] transition-all duration-150 active:scale-95 flex items-center justify-center gap-2 outline-none border-none"
              >
                <Play className="w-4 h-4 fill-current stroke-[2.5]" />
                <span>Initiate Pipeline Analysis</span>
              </button>
            </div>
          </div>
        ) : (
          /* Fancy Drag & Drop Region */
          <div 
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative h-64 w-full rounded-2xl flex flex-col items-center justify-center neon-dash group cursor-pointer transition-all duration-350 ${
              isDragOver ? 'scale-[1.01] bg-secondary-container/5 border-secondary/50' : 'bg-surface-container-low/60 hover:bg-surface-container-low'
            }`}
          >
            <input 
              type="file" 
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden" 
              accept=".pdf,.png,.jpg,.jpeg,.tiff,.json"
            />
            <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl pointer-events-none" />
            
            <Upload className={`w-12 h-12 text-primary mb-4 transition-transform duration-350 ${
              isDragOver ? '-translate-y-2 text-secondary' : 'group-hover:-translate-y-1'
            }`} />
            
            <p className="font-headline text-lg font-black text-on-surface mb-1">
              Drop physical blueprints, tables or PDFs here
            </p>
            <p className="custom-font-mono text-[9px] text-on-surface-variant font-bold tracking-widest uppercase mb-1">
              PDF, PNG, JPEG, TIFF, or JSON up to 120MB
            </p>
            
            <button className="mt-5 px-6 py-3 bg-primary text-on-primary font-headline text-xs font-black tracking-widest uppercase rounded-lg hover:brightness-110 active:scale-95 duration-100 shadow-md">
              Browse Core Files
            </button>
          </div>
        )}

        {/* Global Pipeline Configurations & Model Selecting Panels */}
        <div className="bg-[#020617] p-6 rounded-2xl border border-outline-variant/60 flex flex-col gap-6 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-48 h-48 bg-primary/5 blur-[50px] pointer-events-none rounded-full" />
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-outline-variant/65 pb-4">
            <div className="flex items-center gap-2.5">
              <Sparkles className="w-5 h-5 text-primary flex-shrink-0 animate-pulse" />
              <div>
                <h3 className="font-headline text-sm font-black text-on-surface">
                  Core Dispatcher Preset Configuration
                </h3>
                <p className="text-xs text-on-surface-variant leading-relaxed font-sans mt-0.5">
                  Select a template to configure layout parser models or customize options manually.
                </p>
              </div>
            </div>
            
            <select
              value={presetOption}
              onChange={(e) => handlePresetChange(e.target.value as any)}
              className="bg-slate-950 text-white border border-outline-variant/90 rounded-xl p-2.5 text-xs font-mono tracking-wide focus:ring-2 focus:ring-primary/45 focus:outline-none cursor-pointer hover:border-primary/50 transition-colors w-full sm:w-60"
            >
              <option value="all">Full Diagnostic Pipeline</option>
              <option value="structure_ocr">Structure &amp; OCR Only</option>
              <option value="custom">Custom Settings (Manual Mode)</option>
            </select>
          </div>

          {/* Algorithm Grid Selector */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Category 1: Layout Segmentation */}
            <div className={`p-5 rounded-2xl border transition-all duration-300 relative flex flex-col gap-4 bg-slate-950/40 hover:border-slate-700/80 ${
              layoutEnabled ? 'border-sky-500/30' : 'border-outline-variant/30 opacity-55'
            }`}>
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${layoutEnabled ? 'bg-[#06b6d4]' : 'bg-slate-700'}`} />
                  <span className="text-[10px] custom-font-mono text-slate-400 font-extrabold uppercase">
                    Stage 1: Layout Segmentation
                  </span>
                </div>
                
                <button
                  onClick={() => {
                    setPresetOption('custom');
                    setLayoutEnabled(!layoutEnabled);
                  }}
                  className="focus:outline-none border-none bg-transparent"
                >
                  {layoutEnabled ? (
                    <CheckSquare className="w-5 h-5 text-[#06b6d4]" />
                  ) : (
                    <Square className="w-5 h-5 text-slate-700 hover:text-slate-500" />
                  )}
                </button>
              </div>

              <div>
                <h4 className="font-headline text-sm font-bold text-slate-100 flex items-center justify-between">
                  <span>Layout Segmentation Model</span>
                  {layoutEnabled && (
                    <span className="text-[8px] font-mono bg-cyan-950/60 border border-cyan-800 text-cyan-300 px-2 py-0.5 rounded font-black uppercase">
                      Active
                    </span>
                  )}
                </h4>
                <p className="text-[11px] text-slate-400 leading-relaxed font-sans mt-1">
                  Identify block coordinates, margins, column matrices and heading nodes.
                </p>
              </div>

              {layoutEnabled ? (
                <div>
                  <label className="text-[9px] font-mono uppercase text-slate-500 tracking-wider font-extrabold">Active Architecture</label>
                  <select
                    value={layoutAlgo}
                    onChange={(e) => {
                      setPresetOption('custom');
                      setLayoutAlgo(e.target.value as any);
                    }}
                    className="w-full mt-1.5 bg-slate-950 text-slate-100 border border-slate-800 rounded-xl p-2.5 text-xs font-sans font-semibold focus:outline-none cursor-pointer hover:border-slate-700 transition"
                  >
                    <option value="doclayout_yolo">doclayout_yolo (DocLayout-YOLOv10)</option>
                    <option value="nemotron_parse">nemotron_parse (NVIDIA Nemotron-Parse-v1.1)</option>
                    <option value="landingai_ade">landingai_ade (LandingAI ADE-DPT2)</option>

                  </select>
                </div>
              ) : (
                <div className="h-[52px] bg-slate-900/30 rounded-xl flex items-center justify-center border border-dashed border-slate-900 border-spacing-2">
                  <span className="text-[9px] font-mono uppercase text-slate-500 tracking-wider">Bypassed</span>
                </div>
              )}
            </div>

            {/* Category 2: Optical Character Recognition */}
            <div className={`p-5 rounded-2xl border transition-all duration-300 relative flex flex-col gap-4 bg-slate-950/40 hover:border-slate-700/80 ${
              ocrEnabled ? 'border-indigo-500/30' : 'border-outline-variant/30 opacity-55'
            }`}>
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${ocrEnabled ? 'bg-indigo-400' : 'bg-slate-700'}`} />
                  <span className="text-[10px] custom-font-mono text-slate-400 font-extrabold uppercase">
                    Stage 2: OCR Deep Scan
                  </span>
                </div>
                
                <button
                  onClick={() => {
                    setPresetOption('custom');
                    setOcrEnabled(!ocrEnabled);
                  }}
                  className="focus:outline-none border-none bg-transparent"
                >
                  {ocrEnabled ? (
                    <CheckSquare className="w-5 h-5 text-indigo-400" />
                  ) : (
                    <Square className="w-5 h-5 text-slate-700 hover:text-slate-500" />
                  )}
                </button>
              </div>

              <div>
                <h4 className="font-headline text-sm font-bold text-slate-100 flex items-center justify-between">
                  <span>OCR Transcription Model</span>
                  {ocrEnabled && (
                    <span className="text-[8px] font-mono bg-indigo-950/60 border border-indigo-800 text-indigo-300 px-2 py-0.5 rounded font-black uppercase">
                      Active
                    </span>
                  )}
                </h4>
                <p className="text-[11px] text-slate-400 leading-relaxed font-sans mt-1">
                  Extract highly skewed metadata characters inside dense machine schematics.
                </p>
              </div>

              {ocrEnabled ? (
                <div>
                  <label className="text-[9px] font-mono uppercase text-slate-500 tracking-wider font-extrabold">Active Architecture</label>
                  <select
                    value={ocrAlgo}
                    onChange={(e) => {
                      setPresetOption('custom');
                      setOcrAlgo(e.target.value as any);
                    }}
                    className="w-full mt-1.5 bg-slate-950 text-slate-100 border border-slate-800 rounded-xl p-2.5 text-xs font-sans font-semibold focus:outline-none cursor-pointer hover:border-slate-700 transition"
                  >
                    <option value="easyocr">easyocr (EasyOCR engine)</option>
                    <option value="tesseract">tesseract (Tesseract OCR engine)</option>

                  </select>
                </div>
              ) : (
                <div className="h-[52px] bg-slate-900/30 rounded-xl flex items-center justify-center border border-dashed border-slate-900 border-spacing-2">
                  <span className="text-[9px] font-mono uppercase text-slate-500 tracking-wider">Bypassed</span>
                </div>
              )}
            </div>

            {/* Category 3: Table Structure Extraction */}
            <div className={`p-5 rounded-2xl border transition-all duration-300 relative flex flex-col gap-4 bg-slate-950/40 hover:border-slate-700/80 ${
              tableEnabled ? 'border-fuchsia-500/30' : 'border-outline-variant/30 opacity-55'
            }`}>
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${tableEnabled ? 'bg-fuchsia-400' : 'bg-slate-700'}`} />
                  <span className="text-[10px] custom-font-mono text-slate-400 font-extrabold uppercase">
                    Stage 3: Table Extraction
                  </span>
                </div>
                
                <button
                  onClick={() => {
                    setPresetOption('custom');
                    setTableEnabled(!tableEnabled);
                  }}
                  className="focus:outline-none border-none bg-transparent"
                >
                  {tableEnabled ? (
                    <CheckSquare className="w-5 h-5 text-fuchsia-400" />
                  ) : (
                    <Square className="w-5 h-5 text-slate-700 hover:text-slate-500" />
                  )}
                </button>
              </div>

              <div>
                <h4 className="font-headline text-sm font-bold text-slate-100 flex items-center justify-between">
                  <span>Table Parser Model</span>
                  {tableEnabled && (
                    <span className="text-[8px] font-mono bg-fuchsia-950/65 border border-fuchsia-800 text-fuchsia-300 px-2 py-0.5 rounded font-black uppercase">
                      Active
                    </span>
                  )}
                </h4>
                <p className="text-[11px] text-slate-400 leading-relaxed font-sans mt-1">
                  Reconstruct column structures, double-spanning margins and rules.
                </p>
              </div>

              {tableEnabled ? (
                <div>
                  <label className="text-[9px] font-mono uppercase text-slate-500 tracking-wider font-extrabold">Active Architecture</label>
                  <select
                    value={tableAlgo}
                    onChange={(e) => {
                      setPresetOption('custom');
                      setTableAlgo(e.target.value as any);
                    }}
                    className="w-full mt-1.5 bg-slate-950 text-slate-100 border border-slate-800 rounded-xl p-2.5 text-xs font-sans font-semibold focus:outline-none cursor-pointer hover:border-slate-700 transition"
                  >
                    <option value="tatr">tatr (Table Transformer - TATR Model)</option>
                    <option value="docling_tableformer">docling_tableformer (Docling Framework)</option>

                  </select>
                </div>
              ) : (
                <div className="h-[52px] bg-slate-900/30 rounded-xl flex items-center justify-center border border-dashed border-slate-900 border-spacing-2">
                  <span className="text-[9px] font-mono uppercase text-slate-500 tracking-wider">Bypassed</span>
                </div>
              )}
            </div>

            {/* Category 4: Multimodal Vision Grounding */}
            <div className={`p-5 rounded-2xl border transition-all duration-300 relative flex flex-col gap-4 bg-slate-950/40 hover:border-slate-700/80 ${
              figureEnabled ? 'border-amber-500/30' : 'border-outline-variant/30 opacity-55'
            }`}>
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${figureEnabled ? 'bg-amber-400' : 'bg-slate-700'}`} />
                  <span className="text-[10px] custom-font-mono text-slate-400 font-extrabold uppercase">
                    Stage 4: Vision Grounding
                  </span>
                </div>
                
                <button
                  onClick={() => {
                    setPresetOption('custom');
                    setFigureEnabled(!figureEnabled);
                  }}
                  className="focus:outline-none border-none bg-transparent"
                >
                  {figureEnabled ? (
                    <CheckSquare className="w-5 h-5 text-amber-400" />
                  ) : (
                    <Square className="w-5 h-5 text-slate-700 hover:text-slate-500" />
                  )}
                </button>
              </div>

              <div>
                <h4 className="font-headline text-sm font-bold text-slate-100 flex items-center justify-between">
                  <span>Vision Captioner &amp; Grounding</span>
                  {figureEnabled && (
                    <span className="text-[8px] font-mono bg-amber-950/60 border border-amber-800 text-amber-300 px-2 py-0.5 rounded font-black uppercase">
                      Active
                    </span>
                  )}
                </h4>
                <p className="text-[11px] text-slate-400 leading-relaxed font-sans mt-1">
                  Anchor text annotations with figure Crops to execute multimodal index queries.
                </p>
              </div>

              {figureEnabled ? (
                <div>
                  <label className="text-[9px] font-mono uppercase text-slate-500 tracking-wider font-extrabold">Active Architecture</label>
                  <select
                    value={figureAlgo}
                    onChange={(e) => {
                      setPresetOption('custom');
                      setFigureAlgo(e.target.value as any);
                    }}
                    className="w-full mt-1.5 bg-slate-950 text-slate-100 border border-slate-800 rounded-xl p-2.5 text-xs font-sans font-semibold focus:outline-none cursor-pointer hover:border-slate-700 transition"
                  >
                    <option value="groq_llama">groq_llama (Groq: Llama 4 Scout 17B)</option>
                    <option value="groq_qwen">groq_qwen (Groq: Qwen 3.6 27B)</option>
                    <option value="local_qwen">local_qwen (Local Ollama: Qwen2.5-VL 3B)</option>
                    <option value="local_moondream">local_moondream (Local Ollama: Moondream)</option>

                  </select>
                </div>
              ) : (
                <div className="h-[52px] bg-slate-900/30 rounded-xl flex items-center justify-center border border-dashed border-slate-900 border-spacing-2">
                  <span className="text-[9px] font-mono uppercase text-slate-500 tracking-wider">Bypassed</span>
                </div>
              )}
            </div>

          </div>
        </div>

        {/* Presets and Hotlinks */}
        <div className="p-5 bg-surface-container-low/40 rounded-2xl border border-outline-variant/50">
          <div className="flex items-center gap-2 text-primary mb-3">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="font-headline text-xs font-black uppercase tracking-widest text-primary">
              Quick Simulation Presets
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {presetFiles.map((preset) => (
              <button
                key={preset.name}
                onClick={() => handlePresetClick(preset)}
                className={`p-3.5 bg-surface-container border hover:border-primary rounded-xl text-left transition-all group duration-155 relative h-full flex flex-col justify-between cursor-pointer focus:outline-none ${
                  stagedFile?.name === preset.name ? 'border-primary bg-primary/5 ring-1 ring-primary/25 shadow-md' : 'border-outline-variant/80'
                }`}
              >
                <div>
                  <span className="text-[8px] custom-font-mono uppercase font-black text-primary bg-primary/10 px-1.5 py-0.5 rounded">
                    {preset.tag}
                  </span>
                  <p className="font-headline text-xs font-black text-on-surface group-hover:text-primary mt-2">
                    {preset.display}
                  </p>
                </div>
                <div className="flex items-center gap-2 mt-4 text-[9px] custom-font-mono text-on-surface-variant">
                  <Play className="w-3 h-3 text-secondary" />
                  <span>Stage and Tune Preset</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Runs history panel sidebar layout (4 columns) */}
      <aside className="col-span-12 lg:col-span-4 flex flex-col gap-4">
        <div className="glass-panel-glow p-6 rounded-2xl flex flex-col gap-4 h-full min-h-[500px]">
          <div className="flex justify-between items-center pb-2 border-b border-outline-variant">
            <h2 className="font-headline text-base font-black text-primary">
              Runs Pipeline Queue
            </h2>
            <span className="text-[9px] custom-font-mono text-on-surface-variant font-black uppercase">
              Live Channels
            </span>
          </div>

          <div className="space-y-3 max-h-[520px] overflow-y-auto pr-1">
            {runs.map((run) => {
              const isSelected = selectedJobId ? (run.jobId === selectedJobId) : (run.id === selectedJobId);
              return (
                <div 
                  key={run.id}
                  onClick={() => onSelectJob && onSelectJob(run.jobId || run.id)}
                  className={`p-4 rounded-xl border transition-all duration-155 select-none cursor-pointer ${
                    isSelected
                      ? 'border-primary bg-primary/10 shadow-lg' 
                      : run.status === 'running' 
                      ? 'border-primary/40 bg-surface-container-high/40 shadow-inner' 
                      : run.status === 'failed' 
                      ? 'border-error/20 bg-error-container/5 hover:border-error/40' 
                      : 'border-outline-variant bg-surface-container/35 hover:border-secondary/40'
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center gap-2">
                      {run.status === 'running' ? (
                        <span className="w-2 h-2 rounded-full bg-primary pulse-running" />
                      ) : run.status === 'failed' ? (
                        <span className="w-2 h-2 rounded-full bg-error" />
                      ) : (
                        <CheckCircle className="w-3.5 h-3.5 text-secondary" />
                      )}
                      <span className="font-sans text-xs font-bold text-on-surface">
                        {run.runId}
                      </span>
                    </div>
                    <span className={`text-[9px] font-mono font-black ${
                      run.status === 'running' ? 'text-primary' : run.status === 'failed' ? 'text-error' : 'text-secondary'
                    }`}>
                      {run.status === 'running' ? `${run.progress}%` : run.status === 'failed' ? 'FAILED' : 'DONE'}
                    </span>
                  </div>

                  <p className="text-on-surface-variant font-sans text-xs font-semibold break-all truncate" title={run.fileName}>
                    {run.fileName}
                  </p>

                  {/* Selected Active Models list display inside Queue cards */}
                  {(run.layoutAlgo || run.ocrAlgo || run.tableAlgo || run.figureAlgo) && (
                    <div className="mt-2.5 p-2 bg-slate-950/80 border border-slate-900 rounded-lg text-[9px] font-mono flex flex-col gap-1 text-slate-400 select-text">
                      <span className="text-primary-fixed-dim font-bold text-[8px] uppercase tracking-wider mb-0.5">Pipeline Dispatch Details</span>
                      {run.layoutAlgo ? (
                        <span className="flex items-center justify-between"><strong className="text-cyan-400">Layout:</strong> <span>{run.layoutAlgo}</span></span>
                      ) : (
                        <span className="text-slate-600 line-through">● Layout Segment Bypassed</span>
                      )}
                      {run.ocrAlgo ? (
                        <span className="flex items-center justify-between"><strong className="text-indigo-400">OCR:</strong> <span>{run.ocrAlgo}</span></span>
                      ) : (
                        <span className="text-slate-600 line-through">● OCR Text Bypassed</span>
                      )}
                      {run.tableAlgo ? (
                        <span className="flex items-center justify-between"><strong className="text-fuchsia-400">Table:</strong> <span>{run.tableAlgo}</span></span>
                      ) : (
                        <span className="text-slate-600 line-through">● Table Structure Bypassed</span>
                      )}
                      {run.figureAlgo ? (
                        <span className="flex items-center justify-between"><strong className="text-amber-400">Grounding:</strong> <span>{run.figureAlgo}</span></span>
                      ) : (
                        <span className="text-slate-600 line-through">● Vision Grounding Bypassed</span>
                      )}
                    </div>
                  )}

                  {run.status === 'running' && (
                    <div className="h-1.5 w-full bg-surface-container-highest rounded-full overflow-hidden mt-3 mb-2">
                      <div 
                        className="h-full bg-primary transition-all duration-500 ease-out" 
                        style={{ width: `${run.progress}%` }}
                      />
                    </div>
                  )}

                  {run.message && (
                    <p className={`text-[10px] font-sans mt-2 italic leading-relaxed font-semibold ${
                      run.status === 'failed' ? 'text-error' : 'text-on-surface-variant'
                    }`}>
                      {run.message}
                    </p>
                  )}

                  <div className="flex justify-between items-center text-[9px] custom-font-mono text-outline mt-3">
                    <span>{new Date(run.timestamp).toLocaleTimeString()}</span>
                    <span>{run.duration}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-auto pt-4 border-t border-outline-variant">
            <div className="p-3 bg-error-container/10 border border-error-container/30 rounded-xl flex items-center gap-3">
              <AlertTriangle className="w-4 h-4 text-error pulse-running flex-shrink-0" />
              <p className="text-[9px] text-error font-sans leading-relaxed font-semibold">
                Latency warning on Node-Delta-04 due to high-density table mapping stream congestion.
              </p>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
};
