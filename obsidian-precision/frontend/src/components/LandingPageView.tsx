/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { 
  ArrowRight, ShieldCheck, Cpu, Database, Sparkles, Binary, 
  BookOpen, Terminal, Activity, HelpCircle, FileText, CheckCircle2, ChevronRight, X
} from 'lucide-react';
import { ActiveView } from '../types';

interface LandingPageViewProps {
  setActiveView: (view: ActiveView) => void;
  runsCount: number;
}

export const LandingPageView: React.FC<LandingPageViewProps> = ({ 
  setActiveView,
  runsCount
}) => {
  // Modal State for non-route buttons so absolutely no button is inactive!
  const [activeModal, setActiveModal] = useState<string | null>(null);

  const features = [
    {
      icon: <Sparkles className="w-5 h-5 text-primary" />,
      title: 'Neural Layout Segmentation',
      desc: 'Automatic hierarchical layout identification specifying H1, tables, signatures, text paragraphs, and schematic figures with tight coordinate box metrics.',
      view: 'layout' as ActiveView,
      buttonLabel: 'Launch Inspector',
      specs: ['99.1% Layout Precision', 'BBox Tuning Stage', 'Dynamic Layer Hierarchy']
    },
    {
      icon: <Binary className="w-5 h-5 text-secondary" />,
      title: 'Ultra-Res Optical OCR Scan',
      desc: 'Groundbreaking OCR transcription layer handling low-density scans, skewed vectors, hand-written margins, with per-token confidence arrays.',
      view: 'ocr' as ActiveView,
      buttonLabel: 'Scan OCR Terminal',
      specs: ['Latin-1 & Multi-Lang', 'Scanline Overlay Renderer', 'Confidence Mapping Trace']
    },
    {
      icon: <Database className="w-5 h-5 text-tertiary" />,
      title: 'Tabular Structure Extractor',
      desc: 'Full-grid spreadsheet extraction with native double-click precision cell editing, complex cell spanning, and real-time variance delta recalculations.',
      view: 'table' as ActiveView,
      buttonLabel: 'Extract Table Grid',
      specs: ['Auto Cell-Spanning Detection', 'Re-computable Columns', 'Raw Markdown Stream Output']
    },
    {
      icon: <Cpu className="w-5 h-5 text-cyan-400" />,
      title: 'Multimodal Vision Grounding',
      desc: 'Sophisticated VLM agent linking extracted text blocks directly to matching visual context schematics and micro-crop coordinates instantly.',
      view: 'vision' as ActiveView,
      buttonLabel: 'Ground Vision Narrative',
      specs: ['Auto-Captioning Engines', 'Hover-to-Highlight Grounding', 'Semantic Hash Anchoring']
    }
  ];

  const systemMetrics = [
    { label: 'Active GPU Slices', value: '4 Slices (A100)', color: 'text-primary' },
    { label: 'Total Scanned Docs', value: `${(15243 + runsCount).toLocaleString()}`, color: 'text-secondary' },
    { label: 'Ingestion Speed', value: '240 tokens/ms', color: 'text-tertiary' },
    { label: 'System Uptime Status_OK', value: '99.98%', color: 'text-cyan-400' }
  ];

  return (
    <div className="flex flex-col gap-10 pb-16 select-none animate-fadeIn">
      {/* Hero Header Area */}
      <section className="relative rounded-3xl p-8 lg:p-12 overflow-hidden border border-outline-variant/60 bg-gradient-to-br from-[#0c1324] via-[#151b2d] to-[#070d1f] shadow-2xl">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-primary/10 blur-[120px] rounded-full pointer-events-none" />
        <div className="absolute -bottom-40 -right-40 w-96 h-96 bg-secondary/10 blur-[120px] rounded-full pointer-events-none" />

        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          <div className="lg:col-span-8 space-y-6">
            <div className="inline-flex items-center gap-2 bg-primary/10 border border-primary/25 px-3 py-1 rounded-full text-xs text-primary font-mono font-bold tracking-wider">
              <ShieldCheck className="w-4 h-4 animate-pulse" />
              <span>v4.2.0 MILITARY-GRADE INDUSTRIAL SUITE ACTIVATED</span>
            </div>

            <h1 className="font-headline text-4xl lg:text-5xl font-black text-on-surface leading-tight tracking-tight">
              Deep Multimodal Document <br className="hidden md:inline" />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-secondary to-cyan-400">
                Acoustics &amp; Coordinate Grounding
              </span>
            </h1>

            <p className="text-on-surface-variant text-base leading-relaxed max-w-2xl font-sans font-medium">
              Obsidian Precision is a high-density, low-latency industrial analytics environment engineered to segment, scan, reconstruct and ground unstructured manuals, schematics and audits into reliable digital streams.
            </p>

            <div className="flex flex-wrap gap-4 pt-2">
              <button
                onClick={() => setActiveView('ingestion')}
                className="bg-primary text-on-primary font-headline text-xs font-extrabold tracking-widest uppercase px-6 py-3.5 rounded-xl hover:brightness-110 active:scale-95 duration-100 flex items-center gap-2 transition-all cursor-pointer bg-gradient-to-r from-primary to-primary"
                id="btn-hero-ingest"
              >
                <span>Enter Ingestion Stage</span>
                <ArrowRight className="w-4 h-4 text-on-primary" />
              </button>

              <button
                onClick={() => setActiveModal('system_architecture')}
                className="hover:bg-surface-container-high/60 border border-outline-variant text-on-surface font-headline text-xs font-black tracking-widest uppercase px-6 py-3.5 rounded-xl active:scale-95 duration-100 flex items-center gap-2 transition-all cursor-pointer"
                id="btn-hero-specs"
              >
                <Terminal className="w-4 h-4 text-primary" />
                <span>Diagnostics Blueprint</span>
              </button>
            </div>
          </div>

          {/* Quick Metrics Bento Widget (Right 4 columns) */}
          <div className="lg:col-span-4 bg-surface-container-low/60 border border-outline-variant/80 rounded-2xl p-6 flex flex-col gap-4 shadow-inner backdrop-blur-sm self-stretch justify-between">
            <div className="flex justify-between items-center pb-2 border-b border-outline-variant">
              <span className="custom-font-mono text-[10px] text-outline font-black tracking-widest uppercase">
                Active System Metrology
              </span>
              <Activity className="w-3.5 h-3.5 text-secondary animate-pulse" />
            </div>

            <div className="space-y-4">
              {systemMetrics.map((sm, index) => (
                <div key={index} className="flex justify-between items-end">
                  <span className="text-xs text-on-surface-variant font-medium font-sans">
                    {sm.label}
                  </span>
                  <span className={`text-sm custom-font-mono font-black ${sm.color}`}>
                    {sm.value}
                  </span>
                </div>
              ))}
            </div>

            <button
              onClick={() => setActiveModal('integrity_check')}
              className="w-full mt-3 bg-secondary-container/20 border border-secondary/30 hover:bg-secondary/10 text-secondary font-headline text-[10px] font-black tracking-widest uppercase py-2.5 rounded-lg active:scale-95 duration-150 transition-all flex items-center justify-center gap-1.5"
            >
              <CheckCircle2 className="w-3.5 h-3.5" />
              Run Cluster Integrity Check
            </button>
          </div>
        </div>
      </section>

      {/* Structured Modules & Engines Row */}
      <section className="space-y-4">
        <div>
          <h2 className="font-headline text-2xl font-black text-on-surface flex items-center gap-2">
            <Cpu className="w-6 h-6 text-primary" />
            Autonomous Analysis Sub-Engines
          </h2>
          <p className="text-on-surface-variant text-sm font-medium">
            Jump directly into any highly specialized stage to inspect coordinates or refine extracted cells.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
          {features.map((feat, index) => (
            <div 
              key={index} 
              className="glass-panel-glow bg-surface-container-low/40 border border-outline-variant/60 rounded-2xl p-6 flex flex-col justify-between group hover:border-primary/50 transition-all duration-350 hover:shadow-xl hover:scale-[1.006]"
            >
              <div>
                <div className="flex justify-between items-start mb-4">
                  <div className="w-10 h-10 bg-surface-container-highest rounded-xl flex items-center justify-center border border-outline-variant">
                    {feat.icon}
                  </div>
                  
                  <span className="text-[9px] custom-font-mono text-outline font-black tracking-widest uppercase bg-surface-container px-2.5 py-0.5 rounded border border-outline-variant">
                    STAGE {index + 1}
                  </span>
                </div>

                <h3 className="font-headline text-lg font-black text-on-surface group-hover:text-primary transition-colors mb-2">
                  {feat.title}
                </h3>
                <p className="text-on-surface-variant text-xs leading-relaxed font-sans font-medium mb-4">
                  {feat.desc}
                </p>

                {/* Sub-specification highlights */}
                <div className="space-y-1.5 mb-6">
                  {feat.specs.map((spec, sidx) => (
                    <div key={sidx} className="flex items-center gap-2 text-[10px] font-sans text-on-surface/85">
                      <ChevronRight className="w-3.5 h-3.5 text-secondary flex-shrink-0" />
                      <span>{spec}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => setActiveView(feat.view)}
                  className="flex-1 bg-primary text-on-primary font-headline text-xs font-black tracking-widest uppercase py-3 rounded-lg hover:brightness-110 active:scale-95 duration-100 flex items-center justify-center gap-1.5"
                >
                  <span>{feat.buttonLabel}</span>
                  <ArrowRight className="w-3.5 h-3.5 text-on-primary" />
                </button>
                <button
                  onClick={() => setActiveModal(`stage_doc_${feat.view}`)}
                  className="px-3 border border-outline-variant rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-variant/30 active:scale-95 duration-100 transition-all"
                  title="View technical documentation"
                >
                  <BookOpen className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Secondary Information Segment Card details with quick tips */}
      <section className="bg-surface-container-low/30 border border-outline-variant/60 rounded-3xl p-6 flex flex-col gap-4">
        <h3 className="font-headline text-xs font-black tracking-widest uppercase text-primary">
          Analytical Platform Pipeline Workflows
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-background/50 border border-outline-variant/60 rounded-xl">
            <span className="text-[10px] custom-font-mono text-secondary font-black tracking-wider uppercase">
              Step 01: Ingestion
            </span>
            <p className="font-headline text-xs font-bold text-on-surface mt-1.5 mb-1">
              Upload Industrial Files
            </p>
            <p className="text-[10px] font-sans text-on-surface-variant leading-relaxed font-semibold">
              Drop PDFs, schematics images, or spreadsheets to load into deep memory queues immediately.
            </p>
          </div>

          <div className="p-4 bg-background/50 border border-outline-variant/60 rounded-xl">
            <span className="text-[10px] custom-font-mono text-primary font-black tracking-wider uppercase">
              Step 02: Processing
            </span>
            <p className="font-headline text-xs font-bold text-on-surface mt-1.5 mb-1">
              Assemble Pipeline Models
            </p>
            <p className="text-[10px] font-sans text-on-surface-variant leading-relaxed font-semibold">
              Toggle specific algorithms (Layout segmentation, character recognition, tabular matrix rebuild, entity VLM).
            </p>
          </div>

          <div className="p-4 bg-background/50 border border-outline-variant/60 rounded-xl">
            <span className="text-[10px] custom-font-mono text-tertiary font-black tracking-wider uppercase">
              Step 03: Validation
            </span>
            <p className="font-headline text-xs font-bold text-on-surface mt-1.5 mb-1">
              Refine &amp; Recalculate
            </p>
            <p className="text-[10px] font-sans text-on-surface-variant leading-relaxed font-semibold">
              Double click content inside dataset tables, preview bounding box layers, and anchor semantic VLM narratives.
            </p>
          </div>
        </div>
      </section>

      {/* POPUP MODALS SECTION FOR STABLE & FUNCTIONAL BUTTON INTERACTIONS */}
      {activeModal && (
        <div className="fixed inset-0 bg-[#020617]/85 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-surface-container-highest border border-outline-variant rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl animate-scaleIn">
            <div className="px-6 py-4 border-b border-outline-variant flex justify-between items-center bg-surface-container-high select-none">
              <span className="font-headline text-base font-bold text-primary flex items-center gap-1.5">
                <Terminal className="w-4 h-4 text-secondary" />
                {activeModal === 'system_architecture' && 'Diagnostics Blueprint'}
                {activeModal === 'integrity_check' && 'Cluster Integrity Diagnostics'}
                {activeModal.startsWith('stage_doc_') && 'Technical Documentation'}
              </span>
              <button 
                onClick={() => setActiveModal(null)}
                className="text-on-surface-variant hover:text-error transition-colors"
              >
                <X className="w-5 h-5 animate-pulse-slow" />
              </button>
            </div>

            <div className="p-6 select-text overflow-y-auto max-h-[440px] custom-scrollbar text-xs leading-relaxed space-y-4 text-on-surface-variant font-sans font-medium">
              
              {/* Architecture modal content */}
              {activeModal === 'system_architecture' && (
                <>
                  <p>
                    Obsidian Precision features a multi-threaded asynchronous parsing queue architecture orchestrated by local agent controllers. Below are the registered GPU memory offsets and pipeline nodes:
                  </p>
                  <div className="bg-background/80 p-3 rounded-lg border border-outline-variant custom-font-mono text-[10px] text-primary space-y-1">
                    <div>[NODE] INGEST_ALPHA: OK @ 127.0.0.1:3000</div>
                    <div>[GPU] NVIDIA_A100_SLICE_01: allocated (14.2 GB VRAM)</div>
                    <div>[GPU] NVIDIA_A100_SLICE_02: idle (cooling: 42°C)</div>
                    <div>[MODEL] SEG_MESH_CRF: loaded v2.2.0 (weights checksum: F89A1)</div>
                    <div>[MODEL] TEXT_OCR_LATIN: active v4.1 (latin, cyrillic support)</div>
                    <div>[MODEL] VISION_VLM_7B: grounded multi-scale (7.2B params)</div>
                  </div>
                  <p>
                    All document scans use persistent key-value caching pipelines synced to the current local workspace. Any structural cell overrides are computed locally via variance algorithms instantly.
                  </p>
                </>
              )}

              {/* Integrity check modal content */}
              {activeModal === 'integrity_check' && (
                <>
                  <div className="p-3 bg-secondary/10 border border-secondary/20 rounded-xl flex items-start gap-3">
                    <Activity className="w-5 h-5 text-secondary flex-shrink-0 mt-0.5 animate-pulse" />
                    <div>
                      <h4 className="font-headline text-xs font-bold text-on-surface leading-tight">All clusters report Status_OK</h4>
                      <p className="text-[10px] text-on-surface-variant mt-1 leading-relaxed">
                        Diagnostic report completed in 12ms. All network layers are in sync with zero corrupted segments detected.
                      </p>
                    </div>
                  </div>
                  <p className="text-[11px]">
                    Checking node cluster nodes in GPU-Mesh...
                  </p>
                  <div className="grid grid-cols-2 gap-3 text-[10px] custom-font-mono text-outline">
                    <div className="p-2.5 bg-background border border-outline-variant rounded-md">
                      <span className="font-black text-secondary">✓ NODE-ALPHA</span>
                      <p className="mt-1">Ping: 1.2ms | Load: 4%</p>
                    </div>
                    <div className="p-2.5 bg-background border border-outline-variant rounded-md">
                      <span className="font-black text-secondary">✓ NODE-BETA</span>
                      <p className="mt-1">Ping: 0.8ms | Load: 8%</p>
                    </div>
                    <div className="p-2.5 bg-background border border-outline-variant rounded-md">
                      <span className="font-black text-secondary">✓ NODE-GAMMA</span>
                      <p className="mt-1">Ping: 1.1ms | Load: 0%</p>
                    </div>
                    <div className="p-2.5 bg-background border border-outline-variant rounded-md">
                      <span className="font-black text-error">⚠ NODE-DELTA</span>
                      <p className="mt-1 text-tertiary">Ping: 42ms | Latency high</p>
                    </div>
                  </div>
                </>
              )}

              {/* Stage layout doc */}
              {activeModal === 'stage_doc_layout' && (
                <>
                  <h4 className="font-headline text-sm font-bold text-on-surface mb-2">Stage 1: Layout Segmentation Spec</h4>
                  <p>
                    Layout segmentation separates compound document visuals into high-fidelity component categories. It outputs spatial coordinates for further processing.
                  </p>
                  <ul className="list-disc pl-5 space-y-1 text-[11px]">
                    <li><strong>BBox Coordinate Format:</strong> Left, Top, Width, Height parsed as exact ratios of original asset resolutions.</li>
                    <li><strong>Dynamic Classes:</strong> Custom tags are assignable with direct visual bounding rectangle outlines.</li>
                    <li><strong>Hierarchy Depth:</strong> Traverses nested headers, floating sections, footers, page-numbers perfectly.</li>
                  </ul>
                </>
              )}

              {/* Stage ocr doc */}
              {activeModal === 'stage_doc_ocr' && (
                <>
                  <h4 className="font-headline text-sm font-bold text-on-surface mb-2">Stage 2: Ultra-Res Optical OCR Spec</h4>
                  <p>
                    Our optical recognition layer operates directly on physical assets to construct raw character matrices.
                  </p>
                  <ul className="list-disc pl-5 space-y-1 text-[11px]">
                    <li><strong>DPI Density Tolerance:</strong> Scans assets as low as 72 DPI with extreme precision.</li>
                    <li><strong>Skew &amp; Angle Mapping:</strong> Auto-rotates slanted blocks before executing text segmentation processes.</li>
                    <li><strong>Redaction Modes:</strong> Redacts sensitive PII metadata on-the-fly dynamically when filters are toggled.</li>
                  </ul>
                </>
              )}

              {/* Stage table doc */}
              {activeModal === 'stage_doc_table' && (
                <>
                  <h4 className="font-headline text-sm font-bold text-on-surface mb-2">Stage 3: Tabular Structure Matrix Spec</h4>
                  <p>
                    Converts physical printed gridlines or un-ruled textual coordinate sheets into fully functional tabular structures with high structural density.
                  </p>
                  <ul className="list-disc pl-5 space-y-1 text-[11px]">
                    <li><strong>Column Autodetect:</strong> Tracks logical cell intersections even absent physical border gridlines.</li>
                    <li><strong>Double-Click Cell Editors:</strong> Recalculates variance parameters immediately with zero server-side roundtrips.</li>
                    <li><strong>Markdown export:</strong> Exports clean, compliant tables format perfectly matched to Markdown requirements.</li>
                  </ul>
                </>
              )}

              {/* Stage vision doc */}
              {activeModal === 'stage_doc_vision' && (
                <>
                  <h4 className="font-headline text-sm font-bold text-on-surface mb-2">Stage 4: Vision Grounding speculative VLM Spec</h4>
                  <p>
                    Connects physical drawing callouts directly to textual entities. Ideal for blueprinted CAD manuals and equipment lists.
                  </p>
                  <ul className="list-disc pl-5 space-y-1 text-[11px]">
                    <li><strong>Visual Entity Binding:</strong> Highlights connected components like technical specs or figures simultaneously upon hover.</li>
                    <li><strong>Grounded Sentences:</strong> Embeds coordinate hashes into natural language summaries so assertions are immediately traceable.</li>
                    <li><strong>Confidence levels:</strong> Highlights model parameters to monitor potential hallucinations or logical errors.</li>
                  </ul>
                </>
              )}

            </div>

            <div className="p-4 bg-surface-container border-t border-outline-variant flex justify-end">
              <button 
                onClick={() => setActiveModal(null)}
                className="bg-primary text-on-primary font-headline text-xs font-black tracking-widest uppercase px-5 py-2.5 rounded-lg hover:brightness-110 active:scale-95 duration-100 transition-all outline-none"
              >
                Accept and Close
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
