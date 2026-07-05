/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { 
  ZoomIn, ZoomOut, Layers, Crop, Check, Edit3, Trash2, 
  Plus, Play, AlertTriangle, Monitor, Download, Eye, Layers3, Activity,
  ChevronLeft, ChevronRight
} from 'lucide-react';

interface LayoutBlock {
  id: string;
  label: 'HEADING' | 'PARAGRAPH' | 'TABLE_REGION' | 'SCHEMATIC_FIG' | 'METADATA' | 'SIGNATURE';
  confidence: number;
  top: string; // percentage css pos
  left: string; // percentage css pos
  width: string; // percentage css width
  height: string; // percentage css height
  description: string;
}

const INITIAL_LAYOUT_BLOCKS: LayoutBlock[] = [
  {
    id: 'seg-1',
    label: 'HEADING',
    confidence: 99.4,
    top: '4%',
    left: '12%',
    width: '42%',
    height: '6%',
    description: 'Main drawing title segment: SURGICAL_ACTUATOR_SPEC'
  },
  {
    id: 'seg-2',
    label: 'SCHEMATIC_FIG',
    confidence: 97.2,
    top: '12%',
    left: '12%',
    width: '76%',
    height: '42%',
    description: 'Central multi-axis engineering core vector mapping'
  },
  {
    id: 'seg-3',
    label: 'TABLE_REGION',
    confidence: 94.8,
    top: '56%',
    left: '12%',
    width: '76%',
    height: '24%',
    description: 'Performance parameters tabular indexing matrix'
  },
  {
    id: 'seg-4',
    label: 'METADATA',
    confidence: 99.1,
    top: '82%',
    left: '12%',
    width: '50%',
    height: '8%',
    description: 'Approval checksum, validation hashes, author signatures'
  },
  {
    id: 'seg-5',
    label: 'SIGNATURE',
    confidence: 78.4,
    top: '82%',
    left: '68%',
    width: '20%',
    height: '8%',
    description: 'Validation officer digital credential sign-off zone'
  }
];

interface LayoutSegmentationViewProps {
  jobId: string | null;
}

export const LayoutSegmentationView: React.FC<LayoutSegmentationViewProps> = ({ jobId }) => {
  const [blocks, setBlocks] = useState<LayoutBlock[]>(INITIAL_LAYOUT_BLOCKS);
  const [activeSegId, setActiveSegId] = useState<string | null>('seg-2');
  const [zoomScale, setZoomScale] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);

  // States to add or edit bounding box boundaries!
  const [isAddingBox, setIsAddingBox] = useState(false);
  const [isEditingBox, setIsEditingBox] = useState(false);

  // Input editing temporary state
  const [tempLabel, setTempLabel] = useState<LayoutBlock['label']>('PARAGRAPH' as any);
  const [tempDesc, setTempDesc] = useState('');
  const [tempConf, setTempConf] = useState(95);
  const [tempLeft, setTempLeft] = useState('15%');
  const [tempTop, setTempTop] = useState('15%');
  const [tempWidth, setTempWidth] = useState('20%');
  const [tempHeight, setTempHeight] = useState('10%');

  // Pagination & Image Paths
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [fileName, setFileName] = useState('SOURCE_DOC_089.PDF');
  const [pageImageUrl, setPageImageUrl] = useState('https://lh3.googleusercontent.com/aida-public/AB6AXuCl4o7vbvxQ_VfuE7T3ywuC9zHQI-vgKMPfLqS0CL9WkWhAWKZFb9vx1kfhW78LU3zqRXzUFG4KsoDXI3R7wS1eNcEJY5sU7GAe5fH5zXTwkuAn6-k50GgAfYvf4Ed_ZIWoPeFJMmaW3YBEu4bLgTK8QwmW_Fws_hnInWMvlaYgk-_2V757uSdBomKsvSSb45kCgUrymjs7tBZDj5reLqj5ZZHQ6roDi_3zbu2F67hdw_8HdoLOFokOlslsejOcQoF-l-Haat7HELY');

  // Retrieve Job Metadata & Page Count
  useEffect(() => {
    if (!jobId || jobId.startsWith('run-')) {
      setBlocks(INITIAL_LAYOUT_BLOCKS);
      setActiveSegId('seg-2');
      setFileName('SOURCE_DOC_089.PDF');
      setTotalPages(48);
      setCurrentPage(1);
      setPageImageUrl('https://lh3.googleusercontent.com/aida-public/AB6AXuCl4o7vbvxQ_VfuE7T3ywuC9zHQI-vgKMPfLqS0CL9WkWhAWKZFb9vx1kfhW78LU3zqRXzUFG4KsoDXI3R7wS1eNcEJY5sU7GAe5fH5zXTwkuAn6-k50GgAfYvf4Ed_ZIWoPeFJMmaW3YBEu4bLgTK8QwmW_Fws_hnInWMvlaYgk-_2V757uSdBomKsvSSb45kCgUrymjs7tBZDj5reLqj5ZZHQ6roDi_3zbu2F67hdw_8HdoLOFokOlslsejOcQoF-l-Haat7HELY');
      return;
    }

    const loadJobInfo = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (res.ok) {
          const job = await res.json();
          setFileName(job.filename);
          setTotalPages(job.total_pages || 1);
        }
      } catch (err) {
        console.error('Error fetching job details:', err);
      }
    };
    loadJobInfo();
  }, [jobId]);

  // Load single page detections & PNG
  useEffect(() => {
    if (!jobId || jobId.startsWith('run-')) return;

    const loadPageData = async () => {
      setPageImageUrl(`/api/pages/${jobId}/${currentPage}`);

      try {
        const res = await fetch(`/api/results/${jobId}/page/${currentPage}`);
        if (res.ok) {
          const pageResult = await res.json();
          const detections = pageResult.detections || [];
          const width = pageResult.page_width_px || 612;
          const height = pageResult.page_height_px || 792;

          if (detections.length === 0) {
            setBlocks([]);
            setActiveSegId(null);
            return;
          }

          const mappedBlocks = detections.map((d: any) => {
            const x0 = d.bbox[0];
            const y0 = d.bbox[1];
            const x1 = d.bbox[2];
            const y1 = d.bbox[3];

            const top = `${(y0 / height) * 100}%`;
            const left = `${(x0 / width) * 100}%`;
            const w = `${((x1 - x0) / width) * 100}%`;
            const h = `${((y1 - y0) / height) * 100}%`;

            let mappedLabel: LayoutBlock['label'] = 'PARAGRAPH';
            const dl = (d.label || '').toUpperCase();
            if (dl.includes('HEADING') || dl.includes('TITLE')) mappedLabel = 'HEADING';
            else if (dl.includes('TABLE')) mappedLabel = 'TABLE_REGION';
            else if (dl.includes('FIGURE') || dl.includes('SCHEMATIC') || dl.includes('FIG')) mappedLabel = 'SCHEMATIC_FIG';
            else if (dl.includes('SIGNATURE')) mappedLabel = 'SIGNATURE';
            else if (dl.includes('METADATA') || dl.includes('HEADER') || dl.includes('FOOTER')) mappedLabel = 'METADATA';

            return {
              id: d.id,
              label: mappedLabel,
              confidence: Math.round((d.confidence || 0.95) * 1000) / 10,
              top,
              left,
              width: w,
              height: h,
              description: d.extracted?.content || d.content || `${mappedLabel} block detected by spatial analysis.`
            };
          });

          setBlocks(mappedBlocks);
          setActiveSegId(mappedBlocks[0]?.id || null);
        }
      } catch (err) {
        console.error('Error fetching page coordinate detections:', err);
      }
    };
    loadPageData();
  }, [jobId, currentPage]);

  const selectedBlock = blocks.find((b) => b.id === activeSegId);

  const handleZoomIn = () => setZoomScale((prev) => Math.min(prev + 0.15, 1.8));
  const handleZoomOut = () => setZoomScale((prev) => Math.max(prev - 0.15, 0.7));

  const handlePrevPage = () => {
    setCurrentPage((prev) => Math.max(prev - 1, 1));
  };

  const handleNextPage = () => {
    setCurrentPage((prev) => Math.min(prev + 1, totalPages));
  };

  // Switch to editing state
  const handleStartEdit = (b: LayoutBlock) => {
    setActiveSegId(b.id);
    setTempLabel(b.label);
    setTempDesc(b.description);
    setTempConf(b.confidence);
    setTempLeft(b.left);
    setTempTop(b.top);
    setTempWidth(b.width);
    setTempHeight(b.height);
    setIsEditingBox(true);
    setIsAddingBox(false);
  };

  // Switch to adding block state
  const handleStartAdd = () => {
    setTempLabel('PARAGRAPH' as any);
    setTempDesc('Custom user-anchored layout block mapping');
    setTempConf(99.9);
    setTempLeft('20%');
    setTempTop('45%');
    setTempWidth('40%');
    setTempHeight('15%');
    setIsAddingBox(true);
    setIsEditingBox(false);
  };

  const handleSaveAdd = () => {
    const newBlock: LayoutBlock = {
      id: `seg-${Date.now()}`,
      label: tempLabel,
      confidence: parseFloat(Number(tempConf).toFixed(1)) || 99.9,
      top: tempTop,
      left: tempLeft,
      width: tempWidth,
      height: tempHeight,
      description: tempDesc || 'Custom bounding box coordinate mapping'
    };

    setBlocks((prev) => [...prev, newBlock]);
    setActiveSegId(newBlock.id);
    setIsAddingBox(false);
  };

  const handleSaveEdit = () => {
    if (!activeSegId) return;
    setBlocks((prev) =>
      prev.map((b) =>
        b.id === activeSegId
          ? {
              ...b,
              label: tempLabel,
              confidence: parseFloat(Number(tempConf).toFixed(1)) || b.confidence,
              top: tempTop,
              left: tempLeft,
              width: tempWidth,
              height: tempHeight,
              description: tempDesc || b.description
            }
          : b
      )
    );
    setIsEditingBox(false);
  };

  const handleDeleteBlock = (id: string) => {
    setBlocks((prev) => prev.filter((b) => b.id !== id));
    if (activeSegId === id) {
      setActiveSegId(blocks[0]?.id || null);
    }
    setIsEditingBox(false);
  };

  // Style class helper by Label values
  const getLabelColors = (label: LayoutBlock['label'], isActive: boolean) => {
    let border = 'border-[3px] border-slate-700 bg-slate-900/5 hover:border-slate-300';
    let textBadge = 'bg-slate-950 text-white font-black border border-slate-700 opacity-100';
    let labelText = 'text-slate-400';

    // Rich saturated solid colors supporting high-contrast white background overlay mapping
    const colorMap = {
      SCHEMATIC_FIG: { border: 'border-[#2563eb]', text: 'text-[#3b82f6]', bg: 'bg-[#0055ff]/5', shadow: 'shadow-[0_0_15px_rgba(37,99,235,0.7)]' }, // Electric Blue for figures
      HEADING: { border: 'border-[#06b6d4]', text: 'text-[#22d3ee]', bg: 'bg-[#06b6d4]/5', shadow: 'shadow-[0_0_15px_rgba(6,182,212,0.7)]' }, // Solid Cyan for headings
      TABLE_REGION: { border: 'border-[#d946ef]', text: 'text-[#d946ef]', bg: 'bg-[#d946ef]/5', shadow: 'shadow-[0_0_15px_rgba(217,70,239,0.7)]' }, // Magenta for tables
      SIGNATURE: { border: 'border-[#10b981]', text: 'text-[#10b981]', bg: 'bg-[#10b981]/5', shadow: 'shadow-[0_0_15px_rgba(16,185,129,0.7)]' }, // Emerald Green for signatures
      METADATA: { border: 'border-[#8b5cf6]', text: 'text-[#a78bfa]', bg: 'bg-[#8b5cf6]/5', shadow: 'shadow-[0_0_15px_rgba(139,92,246,0.65)]' },
      PARAGRAPH: { border: 'border-[#f59e0b]', text: 'text-[#fbbf24]', bg: 'bg-[#f59e0b]/5', shadow: 'shadow-[0_0_15px_rgba(245,158,11,0.65)]' }
    };

    const c = colorMap[label] || colorMap['PARAGRAPH'];

    if (isActive) {
      border = `border-[3px] ${c.border} ${c.bg} ring-4 ring-cyan-500/15 ${c.shadow} scale-[1.002] z-30`;
      textBadge = `bg-slate-950 text-amber-400 border-2 ${c.border} font-black opacity-100 shadow-[0_2px_12px_rgba(0,0,0,0.95)] z-40`;
      labelText = `${c.text} font-black`;
    } else {
      border = `border-[3px] ${c.border} ${c.bg} hover:border-white hover:z-20`;
      textBadge = `bg-slate-950 text-white border-[1.5px] ${c.border} font-black opacity-100 z-30`;
      labelText = `${c.text}`;
    }

    return { border, textBadge, labelText };
  };

  return (
    <div className="flex h-[calc(100vh-140px)] gap-6 overflow-hidden select-none animate-fadeIn">
      {/* Left Column: Visual Asset layout segment visualizer */}
      <section className="flex-1 bg-surface-container-lowest relative flex flex-col border border-outline-variant rounded-2xl overflow-hidden shadow-xl">
        <div className="h-12 flex items-center justify-between px-6 bg-surface-container-low border-b border-outline-variant">
          <div className="flex items-center gap-4">
            <Layers3 className="w-4.5 h-4.5 text-primary" />
            <span className="font-headline text-xs font-bold text-on-surface truncate max-w-[200px]">{fileName}</span>
            
            {/* Paging controller */}
            <div className="flex items-center gap-1.5 ml-2">
              <button 
                onClick={handlePrevPage}
                disabled={currentPage === 1}
                className="p-1 hover:bg-surface-variant/60 rounded text-on-surface-variant hover:text-primary disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed transition-colors border-none bg-transparent"
                title="Previous page"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-[10px] font-mono font-bold text-on-surface-variant">
                Page {currentPage} of {totalPages}
              </span>
              <button 
                onClick={handleNextPage}
                disabled={currentPage === totalPages}
                className="p-1 hover:bg-surface-variant/60 rounded text-on-surface-variant hover:text-primary disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed transition-colors border-none bg-transparent"
                title="Next page"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="flex gap-2">
            <button 
              onClick={handleZoomIn}
              className="p-1.5 rounded bg-surface-variant/40 text-on-surface-variant hover:text-primary transition-colors cursor-pointer"
              title="Zoom In scale"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button 
              onClick={handleZoomOut}
              className="p-1.5 rounded bg-surface-variant/40 text-on-surface-variant hover:text-primary transition-colors cursor-pointer"
              title="Zoom Out scale"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button 
              onClick={() => setShowOverlays((p) => !p)}
              className={`p-1.5 rounded transition-all cursor-pointer ${
                showOverlays ? 'bg-primary/25 text-primary' : 'bg-surface-variant/40 text-on-surface-variant hover:text-primary'
              }`}
              title="Toggle Layout Overlays"
            >
              <Layers className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* CAD Layout Blueprint display wrapper */}
        <div className="flex-1 overflow-auto p-12 flex justify-center items-start custom-scrollbar bg-[#020617] relative">
          <div 
            className="relative w-[600px] h-[780px] bg-slate-900 shadow-2xl origin-top transition-transform duration-300 ease-out"
            style={{ transform: `scale(${zoomScale})` }}
          >
            <img 
              alt="CAD Blueprint segment analysis canvas" 
              className="w-full h-full object-contain opacity-90" 
              src={pageImageUrl}
            />
            
            {/* Real-time bounding box layouts */}
            {showOverlays && blocks.map((block) => {
              const isActive = activeSegId === block.id;
              const styles = getLabelColors(block.label, isActive);

              return (
                <div
                  key={block.id}
                  onClick={() => {
                    setActiveSegId(block.id);
                    setIsEditingBox(false);
                  }}
                  style={{
                    top: block.top,
                    left: block.left,
                    width: block.width,
                    height: block.height,
                  }}
                  className={`absolute border transition-all duration-155 cursor-pointer flex items-start p-1.5 select-none z-10 ${styles.border}`}
                >
                  {/* Coordinate micro tag */}
                  <span className={`absolute -top-6 left-0 text-[9px] custom-font-mono px-2 py-0.5 rounded leading-none transition-all z-40 whitespace-nowrap ${styles.textBadge}`}>
                    {block.label} ({block.confidence}%)
                  </span>
                </div>
              );
            })}
            
            <div className="scanline absolute inset-0 opacity-10 pointer-events-none" />
          </div>
        </div>
      </section>

      {/* Right Column: Custom Coordinates editor & BBox Inspector */}
      <section className="w-[430px] flex flex-col bg-surface-container-lowest border border-outline-variant rounded-2xl overflow-hidden shadow-xl">
        <div className="h-12 flex items-center justify-between px-4 bg-surface-container-low border-b border-outline-variant">
          <div className="flex items-center gap-2">
            <Crop className="w-4 h-4 text-secondary" />
            <span className="text-on-surface custom-font-mono text-xs font-black tracking-wider uppercase">
              BBOX_RECON_INSPECTOR
            </span>
          </div>
          <button
            onClick={handleStartAdd}
            className="px-3 py-1 bg-secondary text-on-secondary hover:brightness-110 active:scale-95 duration-100 transition-all font-headline text-[9px] font-black tracking-wider uppercase rounded flex items-center gap-1 cursor-pointer outline-none"
          >
            <Plus className="w-3.5 h-3.5 stroke-[2.5]" />
            Add Box
          </button>
        </div>

        {/* Middle part: list of detected segments with action details */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
          {blocks.length === 0 ? (
            <div className="text-center text-slate-500 italic text-xs py-8">No segments parsed on this page.</div>
          ) : (
            blocks.map((b) => {
              const isActive = activeSegId === b.id;
              const styles = getLabelColors(b.label, isActive);
              
              return (
                <div
                  key={b.id}
                  onClick={() => {
                    setActiveSegId(b.id);
                    if (isActive && !isEditingBox) {
                      setIsEditingBox(false);
                    }
                  }}
                  className={`p-3.5 rounded-xl border transition-all duration-155 cursor-pointer ${
                    isActive 
                      ? 'border-primary bg-primary/5 shadow-inner' 
                      : 'border-outline-variant bg-surface-container/20 hover:border-secondary hover:bg-surface-container/40'
                  }`}
                >
                  <div className="flex justify-between items-center mb-1.5">
                    <span className={`text-[10px] custom-font-mono font-black uppercase ${styles.labelText}`}>
                      {b.label}
                    </span>
                    
                    <div className="flex items-center gap-2">
                      <span className={`text-[9px] custom-font-mono font-black ${b.confidence > 90 ? 'text-secondary' : 'text-error'}`}>
                        {b.confidence}% confidence
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStartEdit(b);
                        }}
                        className="p-1 hover:text-primary transition-colors hover:bg-surface-variant/40 rounded"
                        title="Edit Coordinates & Details"
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteBlock(b.id);
                        }}
                        className="p-1 hover:text-error transition-colors hover:bg-surface-variant/40 rounded"
                        title="Delete document block"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  <p className="text-on-surface-variant text-xs leading-relaxed font-sans font-medium select-text">
                    {b.description}
                  </p>

                  <div className="mt-2 text-[9px] custom-font-mono text-outline flex gap-3 select-text">
                    <span>LEFT: {b.left}</span>
                    <span>TOP: {b.top}</span>
                    <span>W: {b.width}</span>
                    <span>H: {b.height}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Coordinates Adding & Modifying custom panel */}
        {(isAddingBox || isEditingBox) && (
          <div className="p-4 bg-surface-container border-t border-outline-variant space-y-3 shrink-0 animate-scaleIn">
            <div className="flex justify-between items-center pb-2 border-b border-outline-variant/60">
              <span className="font-headline text-xs font-bold text-primary">
                {isAddingBox ? 'ANCHOR NEW BOUNDING BOX' : 'TUNE SEGMENT PARAMETERS'}
              </span>
              <button 
                onClick={() => {
                  setIsAddingBox(false);
                  setIsEditingBox(false);
                }} 
                className="text-on-surface-variant hover:text-error text-xs"
              >
                Cancel
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[9px] custom-font-mono text-outline font-bold uppercase">Block Label</label>
                <select
                  value={tempLabel}
                  onChange={(e) => setTempLabel(e.target.value as any)}
                  className="w-full mt-1 bg-background text-on-surface border border-outline-variant rounded p-1.5 text-xs text-sans font-medium focus:ring-1 focus:ring-primary focus:outline-none"
                >
                  <option value="HEADING">HEADING</option>
                  <option value="PARAGRAPH">PARAGRAPH</option>
                  <option value="TABLE_REGION">TABLE_REGION</option>
                  <option value="SCHEMATIC_FIG">SCHEMATIC_FIG</option>
                  <option value="METADATA">METADATA</option>
                  <option value="SIGNATURE">SIGNATURE</option>
                </select>
              </div>

              <div>
                <label className="text-[9px] custom-font-mono text-outline font-bold uppercase">Confidence (%)</label>
                <input
                  type="number"
                  step="0.1"
                  value={tempConf}
                  onChange={(e) => setTempConf(parseFloat(e.target.value) || 100)}
                  className="w-full mt-1 bg-background text-on-surface border border-outline-variant rounded p-1.5 text-xs custom-font-mono focus:ring-1 focus:ring-primary focus:outline-none"
                />
              </div>
            </div>

            {/* Micro coordinates manual sizing editor */}
            <div className="grid grid-cols-4 gap-2">
              <div>
                <label className="text-[8px] custom-font-mono text-outline uppercase">Left</label>
                <input
                  type="text"
                  value={tempLeft}
                  onChange={(e) => setTempLeft(e.target.value)}
                  className="w-full mt-1 bg-background text-on-surface border border-outline-variant rounded-md p-1.5 text-[10px] custom-font-mono text-center focus:ring-1 focus:ring-primary focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[8px] custom-font-mono text-outline uppercase">Top</label>
                <input
                  type="text"
                  value={tempTop}
                  onChange={(e) => setTempTop(e.target.value)}
                  className="w-full mt-1 bg-background text-on-surface border border-outline-variant rounded-md p-1.5 text-[10px] custom-font-mono text-center focus:ring-1 focus:ring-primary focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[8px] custom-font-mono text-outline uppercase">Width</label>
                <input
                  type="text"
                  value={tempWidth}
                  onChange={(e) => setTempWidth(e.target.value)}
                  className="w-full mt-1 bg-background text-on-surface border border-outline-variant rounded-md p-1.5 text-[10px] custom-font-mono text-center focus:ring-1 focus:ring-primary focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[8px] custom-font-mono text-outline uppercase">Height</label>
                <input
                  type="text"
                  value={tempHeight}
                  onChange={(e) => setTempHeight(e.target.value)}
                  className="w-full mt-1 bg-background text-on-surface border border-outline-variant rounded-md p-1.5 text-[10px] custom-font-mono text-center focus:ring-1 focus:ring-primary focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="text-[9px] custom-font-mono text-outline font-bold uppercase">Annotation / Description</label>
              <input
                type="text"
                value={tempDesc}
                onChange={(e) => setTempDesc(e.target.value)}
                placeholder="Brief structure description..."
                className="w-full mt-1 bg-background text-on-surface border border-outline-variant rounded p-1.5 text-xs text-sans font-medium focus:ring-1 focus:ring-primary focus:outline-none"
              />
            </div>

            <button
              onClick={isAddingBox ? handleSaveAdd : handleSaveEdit}
              className="w-full bg-primary text-on-primary font-headline text-xs font-black tracking-wider uppercase py-2.5 rounded-lg active:scale-95 hover:brightness-110 transition-all duration-100 flex items-center justify-center gap-1.5"
            >
              <Check className="w-4 h-4 text-on-primary stroke-[3]" />
              <span>Apply Changes</span>
            </button>
          </div>
        )}

        {/* Footer actions & metrics */}
        <div className="p-5 bg-surface-container border-t border-outline-variant shrink-0 flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-xl bg-surface-container-high border border-outline-variant/60">
              <p className="text-[9px] custom-font-mono text-outline font-black uppercase mb-1">
                Segments Mapped
              </p>
              <div className="flex items-end gap-1.5">
                <span className="text-lg font-headline font-black text-secondary leading-none">
                  {blocks.length}
                </span>
                <Monitor className="w-3.5 h-3.5 text-secondary mb-0.5" />
              </div>
            </div>
            
            <div className="p-3 rounded-xl bg-surface-container-high border border-outline-variant/60">
              <p className="text-[9px] custom-font-mono text-outline font-black uppercase mb-1">
                Avg Segment Conf.
              </p>
              <div className="flex items-end gap-1.5">
                <span className="text-lg font-headline font-black text-primary leading-none">
                  {(blocks.reduce((acc, b) => acc + b.confidence, 0) / (blocks.length || 1)).toFixed(1)}%
                </span>
                <Activity className="w-3.5 h-3.5 text-primary mb-0.5" />
              </div>
            </div>
          </div>

          <button 
            onClick={() => {
              const rawJson = JSON.stringify(blocks, null, 2);
              navigator.clipboard.writeText(rawJson);
              alert('JSON layout coordinates copied to clipboard!\n\n' + rawJson.slice(0, 200) + '...');
            }}
            className="w-full flex items-center justify-center gap-2 bg-secondary text-on-secondary py-3.5 rounded-xl custom-font-headline text-xs font-black tracking-widest uppercase hover:brightness-110 duration-100 active:scale-95 transition-all outline-none"
          >
            <Download className="w-4 h-4 stroke-[2.5]" />
            Export Layout JSON
          </button>
        </div>
      </section>
    </div>
  );
};
