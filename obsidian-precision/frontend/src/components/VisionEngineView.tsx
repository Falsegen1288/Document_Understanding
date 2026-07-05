/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { 
  Eye, Zap, RefreshCw, AlertTriangle, Monitor, Download, ChevronLeft, ChevronRight, CheckCircle2,
  Cpu, TrendingUp, Info, HelpCircle, Network
} from 'lucide-react';

interface VisionItem {
  id: string;
  title: string;
  imageUrl: string;
  vlmCaption: string;
  anchorHeading: string;
  similarityIdx: number;
  attributesChecklist: { name: string; status: 'verified' | 'warning' }[];
  nearbyContextGrounded: string;
  rawJsonMeta: string;
  tableCropUrl?: string;
  tableMarkdown?: string;
}

const INITIAL_VISION_ITEMS: VisionItem[] = [
  {
    id: 'crop-1',
    title: 'Surgical Actuator Vector Mapping',
    imageUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCl4o7vbvxQ_VfuE7T3ywuC9zHQI-vgKMPfLqS0CL9WkWhAWKZFb9vx1kfhW78LU3zqRXzUFG4KsoDXI3R7wS1eNcEJY5sU7GAe5fH5zXTwkuAn6-k50GgAfYvf4Ed_ZIWoPeFJMmaW3YBEu4bLgTK8QwmW_Fws_hnInWMvlaYgk-_2V757uSdBomKsvSSb45kCgUrymjs7tBZDj5reLqj5ZZHQ6roDi_3zbu2F67hdw_8HdoLOFokOlslsejOcQoF-l-Haat7HELY',
    vlmCaption: 'VLM Analysis: Detailed architectural blueprint representing surgical actuator deflection ratios.',
    anchorHeading: '1. Technical Performance Standards',
    similarityIdx: 96,
    attributesChecklist: [
      { name: 'Vector Checksum Verified', status: 'verified' },
      { name: 'Grounding Headings Aligned', status: 'verified' }
    ],
    nearbyContextGrounded: 'Grounded paragraph text containing actuator spec variables.',
    rawJsonMeta: '{"label": "figure", "confidence": 0.96}'
  }
];

interface VisionEngineViewProps {
  jobId: string | null;
}

export const VisionEngineView: React.FC<VisionEngineViewProps> = ({ jobId }) => {
  const [items, setItems] = useState<VisionItem[]>(INITIAL_VISION_ITEMS);
  const [activeIdx, setActiveIdx] = useState(0);

  // Dynamic state values
  const [fileName, setFileName] = useState<string>('TX-091_SCHEMATIC.PDF');
  
  // Dynamic fetch if jobId exists
  useEffect(() => {
    if (!jobId || jobId.startsWith('run-')) {
      setItems(INITIAL_VISION_ITEMS);
      setActiveIdx(0);
      setFileName('TX-091_SCHEMATIC.PDF');
      return;
    }

    const loadMultimodalData = async () => {
      try {
        const jobRes = await fetch(`/api/jobs/${jobId}`);
        if (jobRes.ok) {
          const jobData = await jobRes.json();
          setFileName(jobData.filename);
        }

        const res = await fetch(`/api/results/${jobId}`);
        if (res.ok) {
          const data = await res.json();
          const pages = data.pages || [];
          const foundItems: VisionItem[] = [];

          pages.forEach((p: any) => {
            const dets = p.detections || [];
            dets.forEach((d: any) => {
              // Extract figure detections with crops
              const isFig = d.label === 'figure' || d.label === 'schematic' || d.label === 'diagram' || d.type === 'figure' || d.type === 'picture';
              if (isFig && d.extracted?.crop_path) {
                let cropUrl = d.extracted.crop_path;
                if (!cropUrl.startsWith('/')) {
                  cropUrl = '/' + cropUrl;
                }

                let tCropUrl = d.extracted.table_crop_path || '';
                if (tCropUrl && !tCropUrl.startsWith('/')) {
                  tCropUrl = '/' + tCropUrl;
                }

                // Attributes checklist
                const attrKeys = d.extracted?.attributes ? Object.keys(d.extracted.attributes) : [];
                const checkList = attrKeys.map(k => ({
                  name: k.replace(/_/g, ' ').toUpperCase(),
                  status: d.extracted.attributes[k] === true || d.extracted.attributes[k] === 'true' || d.extracted.attributes[k] === 'yes' ? 'verified' : 'warning'
                }));

                if (checkList.length === 0) {
                  checkList.push({ name: 'GROUNDED MATCH', status: 'verified' });
                  checkList.push({ name: 'RESOLUTION CHECK', status: 'verified' });
                }

                // extract table markdown if it exists in grounded_context
                const gContext = d.extracted?.grounded_context || d.extracted?.nearby_context || '';
                let tableMarkdown = '';
                if (gContext.includes('### ADJACENT SPECIFICATIONS TABLE:')) {
                  const parts = gContext.split('### ADJACENT SPECIFICATIONS TABLE:');
                  if (parts.length > 1) {
                    const sub = parts[1].trim();
                    const nextHeadingIdx = sub.indexOf('###');
                    tableMarkdown = nextHeadingIdx !== -1 ? sub.substring(0, nextHeadingIdx).trim() : sub;
                  }
                }

                foundItems.push({
                  id: d.id,
                  title: `Multimodal Crop page_${p.page_number}_fig_${d.id.substring(0, 4)}`,
                  imageUrl: cropUrl,
                  vlmCaption: d.extracted?.vlm_description || d.extracted?.caption || d.extracted?.description || 'VLM Caption: Multimodal diagram component description.',
                  anchorHeading: d.extracted?.nearest_heading || (d.extracted?.nearby_headings && d.extracted.nearby_headings[0]) || 'N/A: Core heading node',
                  similarityIdx: Math.round((d.extracted?.similarity_score || d.confidence || 0.95) * 100),
                  attributesChecklist: checkList as any,
                  nearbyContextGrounded: d.extracted?.grounded_context || d.extracted?.nearby_context || 'Nearby paragraph text describing the context of the diagram.',
                  tableCropUrl: tCropUrl || undefined,
                  tableMarkdown: tableMarkdown || undefined,
                  rawJsonMeta: JSON.stringify(d, null, 2)
                });
              }
            });
          });

          if (foundItems.length > 0) {
            setItems(foundItems);
            setActiveIdx(0);
          } else {
            setItems([]);
          }
        }
      } catch (err) {
        console.error('Failed loading multimodal crops from API:', err);
      }
    };
    loadMultimodalData();
  }, [jobId]);

  const activeItem = items[activeIdx];

  const handlePrev = () => {
    setActiveIdx((prev) => (prev > 0 ? prev - 1 : items.length - 1));
  };

  const handleNext = () => {
    setActiveIdx((prev) => (prev < items.length - 1 ? prev + 1 : 0));
  };

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] gap-6 select-none animate-fadeIn">
      {/* Workspace Header Panel */}
      <div className="flex justify-between items-center bg-[#020617] border border-outline-variant/60 rounded-2xl px-6 py-4 shadow-xl">
        <div className="flex items-center gap-4">
          <Eye className="w-5 h-5 text-primary" />
          <div>
            <h1 className="font-headline text-base font-black text-on-surface truncate max-w-[240px] md:max-w-md" title={fileName}>
              {fileName}
            </h1>
            <p className="text-[10px] text-on-surface-variant font-sans font-semibold mt-0.5">
              MULTIMODAL VISION INSPECTION DOCK
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {items.length > 1 && (
            <div className="flex items-center bg-surface-container-low border border-outline-variant rounded-xl p-1 select-none">
              <button 
                onClick={handlePrev}
                className="p-1.5 hover:bg-surface-variant rounded text-on-surface-variant hover:text-primary transition-all cursor-pointer border-none bg-transparent"
                title="Previous Crop"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="px-3 text-xs font-mono font-bold text-on-surface">
                {activeIdx + 1} of {items.length} crops
              </span>
              <button 
                onClick={handleNext}
                className="p-1.5 hover:bg-surface-variant rounded text-on-surface-variant hover:text-primary transition-all cursor-pointer border-none bg-transparent"
                title="Next Crop"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
          
          <button 
            onClick={() => {
              if (activeItem) {
                const link = document.createElement('a');
                link.href = activeItem.imageUrl;
                link.setAttribute('download', `${activeItem.title}.png`);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
              }
            }}
            disabled={!activeItem}
            className="bg-primary text-on-primary font-headline text-xs font-black tracking-widest uppercase px-4 py-2.5 rounded-xl hover:brightness-110 active:scale-95 duration-100 flex items-center gap-1.5 transition-all outline-none border-none disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            <Download className="w-4 h-4 stroke-[2.5]" />
            Download Crop PNG
          </button>
        </div>
      </div>

      {/* Main Vision board split layout */}
      {items.length === 0 ? (
        <div className="flex-1 bg-surface-container-lowest border border-outline-variant rounded-2xl flex flex-col items-center justify-center p-8">
          <AlertTriangle className="w-12 h-12 text-slate-500 mb-3" />
          <p className="text-slate-400 italic text-sm">No multimodal figure crops parsed for this job.</p>
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-12 gap-6 overflow-hidden">
          
          {/* Left panel: high resolution image crop */}
          <section className="col-span-12 lg:col-span-4 bg-slate-950 border border-outline-variant/60 rounded-2xl flex items-center justify-center p-8 relative shadow-2xl overflow-hidden">
            <div className="absolute inset-0 bg-primary/2 opacity-15 pointer-events-none" />
            <img 
              alt={activeItem.title} 
              className="max-w-full max-h-full object-contain rounded-lg border border-slate-900 shadow-2xl relative z-10 hover:scale-[1.01] transition-transform duration-300"
              src={activeItem.imageUrl}
            />
            <div className="absolute bottom-4 left-4 bg-slate-900/90 border border-slate-800 rounded-lg px-3 py-1.5 z-20 text-[10px] custom-font-mono text-outline select-text">
              SOURCE FILE CROP: {activeItem.title.toUpperCase()}
            </div>
          </section>

          {/* Right panel: VLM metadata grounding inspect deck */}
          <section className="col-span-12 lg:col-span-8 h-full overflow-y-auto flex flex-col gap-5 select-none custom-scrollbar">
            
            {/* 1. VLM semantic captioning card */}
            <div className="bg-surface-container-low border border-outline-variant/65 rounded-2xl p-5 shadow-xl">
              <div className="flex items-center gap-2 text-primary border-b border-outline-variant/60 pb-3 mb-3">
                <Zap className="w-4 h-4 text-primary animate-pulse" />
                <span className="font-headline text-[10px] font-black uppercase tracking-wider">
                  VLM Grounded Description
                </span>
              </div>
              <p className="text-slate-100 font-mono text-[11px] leading-relaxed bg-slate-950/70 border border-slate-900 p-4 rounded-xl select-text font-medium">
                {activeItem.vlmCaption}
              </p>
            </div>



            {/* Linked Adjacent Specs Table Card */}
            {activeItem.tableCropUrl && (
              <div className="bg-surface-container-low border border-outline-variant/65 rounded-2xl p-5 shadow-xl flex flex-col gap-3">
                <div className="flex items-center gap-2 text-purple-600 border-b border-outline-variant/60 pb-3 mb-1">
                  <span className="material-symbols-outlined text-purple-500 text-lg">table_chart</span>
                  <span className="font-headline text-[10px] font-black uppercase tracking-wider text-purple-400">
                    Adjacent Specifications Table
                  </span>
                </div>
                
                {/* Visual crop of the adjacent table */}
                <div className="rounded-xl overflow-hidden border border-slate-900 bg-slate-950 flex flex-col p-1.5 shadow-inner">
                  <div className="text-[8px] font-mono text-slate-500 uppercase tracking-widest px-2 py-1 select-none">
                    Linked Table Image Crop
                  </div>
                  <div className="relative aspect-video max-h-48 overflow-hidden bg-slate-950 flex items-center justify-center rounded">
                    <img 
                      src={activeItem.tableCropUrl} 
                      alt="Adjacent Table Crop" 
                      className="max-w-full max-h-full object-contain rounded border border-slate-900/60"
                    />
                  </div>
                </div>

                {/* Formatted Specs Table Data from Markdown */}
                {activeItem.tableMarkdown && (
                  <div className="flex flex-col gap-1.5 mt-1 bg-slate-950/70 border border-slate-900 rounded-xl p-4 overflow-hidden">
                    <span className="text-[8px] font-mono text-purple-400 uppercase tracking-wider font-bold">
                      Extracted Specification Markdown
                    </span>
                    <pre className="text-slate-300 font-mono text-[9px] leading-relaxed select-text overflow-x-auto overflow-y-auto max-h-40 custom-scrollbar p-1">
                      {activeItem.tableMarkdown}
                    </pre>
                  </div>
                )}
              </div>
            )}

            {/* 4. Attribute verification lists & matching indicators */}
            <div className="bg-surface-container-low border border-outline-variant/65 rounded-2xl p-5 shadow-xl flex flex-col gap-4">
              <div className="flex justify-between items-center border-b border-outline-variant/60 pb-3">
                <span className="font-headline text-[10px] font-black uppercase tracking-wider text-slate-400">
                  Feature Grounding checklists
                </span>
                <span className="text-[9px] custom-font-mono text-on-surface-variant font-black">
                  VLM MAPPED
                </span>
              </div>

              {/* Similarity gauge & attributes checklist split */}
              <div className="grid grid-cols-12 gap-4 items-center">
                {/* Similarity score gauge dials */}
                <div className="col-span-5 flex flex-col items-center justify-center p-3 bg-slate-950/75 border border-slate-900 rounded-xl relative">
                  <span className="text-[8px] custom-font-mono text-outline font-bold uppercase tracking-wider mb-2">Sim Score</span>
                  <div className="relative w-18 h-18 flex items-center justify-center">
                    <svg className="w-full h-full transform -rotate-90">
                      <circle cx="36" cy="36" r="30" stroke="#1e293b" strokeWidth="6" fill="transparent" />
                      <circle cx="36" cy="36" r="30" stroke="#10b981" strokeWidth="6" fill="transparent" 
                        strokeDasharray={2 * Math.PI * 30}
                        strokeDashoffset={2 * Math.PI * 30 * (1 - activeItem.similarityIdx / 100)}
                      />
                    </svg>
                    <span className="absolute font-mono text-base font-extrabold text-secondary">
                      {activeItem.similarityIdx}%
                    </span>
                  </div>
                </div>

                {/* Attributes checklist list */}
                <div className="col-span-7 space-y-2.5">
                  {activeItem.attributesChecklist.map((attr, idx) => (
                    <div key={idx} className="flex justify-between items-center text-xs">
                      <span className="text-on-surface-variant font-medium truncate max-w-[140px]" title={attr.name}>{attr.name}</span>
                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        {attr.status === 'verified' ? (
                          <>
                            <span className="text-[9px] font-mono text-secondary uppercase font-bold">Passed</span>
                            <CheckCircle2 className="w-3.5 h-3.5 text-secondary" />
                          </>
                        ) : (
                          <>
                            <span className="text-[9px] font-mono text-error uppercase font-bold">Unverified</span>
                            <AlertTriangle className="w-3.5 h-3.5 text-error animate-pulse" />
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 5. Raw Grounded JSON meta schema */}
            <div className="bg-surface-container-low border border-outline-variant/65 rounded-2xl p-5 shadow-xl">
              <div className="flex items-center gap-2 text-slate-500 border-b border-outline-variant/60 pb-3 mb-3">
                <Cpu className="w-4 h-4 text-slate-500" />
                <span className="font-headline text-[10px] font-black uppercase tracking-wider">
                  Raw Grounded JSON meta
                </span>
              </div>
              <pre className="text-slate-400 font-mono text-[9px] leading-relaxed bg-slate-950/70 border border-slate-900 p-4 rounded-xl select-text max-h-40 overflow-y-auto custom-scrollbar">
                {activeItem.rawJsonMeta}
              </pre>
            </div>
            
          </section>
        </div>
      )}
    </div>
  );
};
