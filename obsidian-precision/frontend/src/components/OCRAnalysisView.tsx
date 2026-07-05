/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { ZoomIn, ZoomOut, Layers, Copy, Terminal, ChevronLeft, ChevronRight, TrendingUp, Cpu, Download } from 'lucide-react';

interface OCRBox {
  id: string;
  index: string;
  label: string;
  confidence: number;
  pos: string;
  text: string;
  top: string;
  left: string;
  width: string;
  height: string;
  lang: string;
  entityType: string;
}

const INITIAL_OCR_BOXES: OCRBox[] = [
  {
    id: 'box-01',
    index: 'TEXT_BLOCK_01',
    label: 'TITLE_HEADING',
    confidence: 99.8,
    pos: 'X: 42 Y: 85',
    text: 'DEEP_STAGE SURGICAL INTERFACE CORE',
    top: '3%',
    left: '12%',
    width: '45%',
    height: '4%',
    lang: 'easyocr_v2.0',
    entityType: 'METADATA_HEADER'
  },
  {
    id: 'box-02',
    index: 'TEXT_BLOCK_02',
    label: 'SECTION_HEADER',
    confidence: 97.4,
    pos: 'X: 42 Y: 145',
    text: '1. TECHNICAL PERFORMANCE STANDARDS',
    top: '8%',
    left: '12%',
    width: '38%',
    height: '3%',
    lang: 'easyocr_v2.0',
    entityType: 'HEADING'
  },
  {
    id: 'box-03',
    index: 'TEXT_BLOCK_03',
    label: 'SPECIFICATION_ROW',
    confidence: 94.2,
    pos: 'X: 84 Y: 220',
    text: 'Multi-axis deflection ratio limits configured within 0.04% precision tolerance.',
    top: '14%',
    left: '12%',
    width: '76%',
    height: '6%',
    lang: 'easyocr_v2.0',
    entityType: 'PARAGRAPH'
  }
];

interface OCRAnalysisViewProps {
  jobId: string | null;
}

export const OCRAnalysisView: React.FC<OCRAnalysisViewProps> = ({ jobId }) => {
  const [ocrBoxes, setOcrBoxes] = useState<OCRBox[]>(INITIAL_OCR_BOXES);
  const [activeBoxId, setActiveBoxId] = useState<string | null>('box-03');
  const [zoomScale, setZoomScale] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);

  // Pagination & Image Paths
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [fileName, setFileName] = useState('SOURCE_DOC_089.PDF');
  const [pageImageUrl, setPageImageUrl] = useState('https://lh3.googleusercontent.com/aida-public/AB6AXuAOlvV9hWUhj5U3RQ9rGXHVUmi3-J6NDAnudof29YbuFje6AAtikcWSlY6UxUyolFS009DzwoidO7NOjb12zzRlzCxBNpuokiRvtuIAhPBOj5QDdbPDckI1-fuJENtGz0AeGMSGDxozuB75HVSVbu-3QYriP_oveoLBa-lcJ_flt2l0Oq64TXnvBVCCdcMPeHKv4hidChxXefm8TjLmjd93NngASWIWqM4tzs5DGorecrN3xHH63Q0saYpf8qopL06SblMcpZSeUTg');

  // Retrieve Job Metadata & Page Count
  useEffect(() => {
    if (!jobId || jobId.startsWith('run-')) {
      setOcrBoxes(INITIAL_OCR_BOXES);
      setActiveBoxId('box-03');
      setFileName('SOURCE_DOC_089.PDF');
      setTotalPages(48);
      setCurrentPage(1);
      setPageImageUrl('https://lh3.googleusercontent.com/aida-public/AB6AXuAOlvV9hWUhj5U3RQ9rGXHVUmi3-J6NDAnudof29YbuFje6AAtikcWSlY6UxUyolFS009DzwoidO7NOjb12zzRlzCxBNpuokiRvtuIAhPBOj5QDdbPDckI1-fuJENtGz0AeGMSGDxozuB75HVSVbu-3QYriP_oveoLBa-lcJ_flt2l0Oq64TXnvBVCCdcMPeHKv4hidChxXefm8TjLmjd93NngASWIWqM4tzs5DGorecrN3xHH63Q0saYpf8qopL06SblMcpZSeUTg');
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

          // We filter text types
          const textDets = detections.filter((d: any) => d.type === 'text' || d.type === 'plain text' || d.type === 'title' || d.type === 'section_header' || d.type === 'heading' || d.type === 'header' || d.type === 'paragraph' || d.type === 'caption');

          if (textDets.length === 0) {
            setOcrBoxes([]);
            setActiveBoxId(null);
            return;
          }

          const mappedBoxes = textDets.map((d: any, index: number) => {
            const x0 = d.bbox[0];
            const y0 = d.bbox[1];
            const x1 = d.bbox[2];
            const y1 = d.bbox[3];

            const top = `${(y0 / height) * 100}%`;
            const left = `${(x0 / width) * 100}%`;
            const w = `${((x1 - x0) / width) * 100}%`;
            const h = `${((y1 - y0) / height) * 100}%`;

            const content = d.extracted?.content || d.content || '';

            return {
              id: d.id,
              index: `TEXT_LINE_${String(index + 1).padStart(3, '0')}`,
              label: d.type.toUpperCase(),
              confidence: Math.round((d.confidence || 0.95) * 1000) / 10,
              pos: `X: ${Math.round(x0)} Y: ${Math.round(y0)}`,
              text: content,
              top,
              left,
              width: w,
              height: h,
              lang: d.extracted?.ocr_engine || 'pymupdf',
              entityType: d.type.toUpperCase(),
            };
          });

          setOcrBoxes(mappedBoxes);
          setActiveBoxId(mappedBoxes[0]?.id || null);
        }
      } catch (err) {
        console.error('Error fetching layout segments:', err);
      }
    };
    loadPageData();
  }, [jobId, currentPage]);

  const handlePrevPage = () => {
    setCurrentPage((prev) => Math.max(prev - 1, 1));
  };

  const handleNextPage = () => {
    setCurrentPage((prev) => Math.min(prev + 1, totalPages));
  };

  const handleZoomIn = () => {
    setZoomScale((prev) => Math.min(prev + 0.15, 1.8));
  };

  const handleZoomOut = () => {
    setZoomScale((prev) => Math.max(prev - 0.15, 0.7));
  };

  const handleCopyStream = () => {
    const rawStreamText = ocrBoxes.map((b) => `[INDEX: ${b.index}] ${b.text}`).join('\n');
    navigator.clipboard.writeText(rawStreamText);
    alert('OCR transcription stream copied to clipboard!');
  };

  const selectedBox = ocrBoxes.find((b) => b.id === activeBoxId);

  return (
    <div className="flex h-[calc(100vh-140px)] gap-6 overflow-hidden">
      {/* Left side: Original blueprint / document canvas stage */}
      <section className="flex-1 bg-surface-container-lowest relative flex flex-col border border-outline-variant rounded-2xl overflow-hidden shadow-xl">
        <div className="h-12 flex items-center justify-between px-6 bg-surface-container-low border-b border-outline-variant">
          <div className="flex items-center gap-4 select-none">
            <span className="font-headline text-xs font-bold text-on-surface">{fileName}</span>
            <div className="flex items-center gap-1.5">
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
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button 
              onClick={handleZoomOut}
              className="p-1.5 rounded bg-surface-variant/40 text-on-surface-variant hover:text-primary transition-colors cursor-pointer"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button 
              onClick={() => setShowOverlays((p) => !p)}
              className={`p-1.5 rounded transition-all cursor-pointer ${
                showOverlays ? 'bg-primary/25 text-primary' : 'bg-surface-variant/40 text-on-surface-variant hover:text-primary'
              }`}
              title="Toggle Text Highlight Overlays"
            >
              <Layers className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Blueprint display viewport */}
        <div className="flex-1 overflow-auto p-12 flex justify-center items-start custom-scrollbar bg-[#020617] relative">
          <div 
            className="relative w-[600px] h-[780px] bg-slate-900 shadow-2xl origin-top transition-transform duration-300 ease-out"
            style={{ transform: `scale(${zoomScale})` }}
          >
            <img 
              alt="CAD Blueprint Deep scan viewport" 
              className="w-full h-full object-contain opacity-90" 
              src={pageImageUrl}
            />
            
            {/* Draw character lines OCR overlays */}
            {showOverlays && ocrBoxes.map((box) => {
              const isActive = box.id === activeBoxId;
              return (
                <div
                  key={box.id}
                  onClick={() => setActiveBoxId(box.id)}
                  style={{
                    top: box.top,
                    left: box.left,
                    width: box.width,
                    height: box.height,
                  }}
                  className={`absolute border transition-all duration-150 cursor-pointer flex items-center justify-center select-none z-10 ${
                    isActive 
                      ? 'border-[#8b5cf6] bg-[#8b5cf6]/10 ring-2 ring-[#a78bfa]/20 shadow-[0_0_12px_rgba(139,92,246,0.6)] scale-[1.002] z-20' 
                      : 'border-indigo-400 bg-indigo-500/5 hover:border-white'
                  }`}
                  title={box.text}
                >
                  <span className={`absolute -top-4.5 left-0 text-[8px] custom-font-mono px-1 py-0.5 rounded leading-none transition-all ${
                    isActive 
                      ? 'bg-[#8b5cf6] text-white font-extrabold opacity-100 z-30' 
                      : 'bg-slate-950 text-indigo-300 font-bold opacity-0 hover:opacity-100'
                  }`}>
                    {box.index}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Right side: High-contrast OCR terminal details inspect panels */}
      <section className="w-[450px] flex flex-col bg-surface-container-lowest border border-outline-variant rounded-2xl overflow-hidden shadow-xl">
        <div className="h-12 flex items-center justify-between px-4 bg-surface-container-low border-b border-outline-variant">
          <div className="flex items-center gap-2 select-none">
            <Terminal className="w-4 h-4 text-primary" />
            <span className="text-on-surface custom-font-mono text-xs font-black tracking-wider uppercase">
              CHARACTER_OCR_DECK
            </span>
          </div>
          <button
            onClick={handleCopyStream}
            className="px-3 py-1 border border-outline-variant hover:bg-surface-variant hover:text-primary transition-all font-headline text-[9px] font-black tracking-wider uppercase rounded flex items-center gap-1 cursor-pointer outline-none"
          >
            <Copy className="w-3.5 h-3.5" />
            Copy Stream
          </button>
        </div>

        {/* Text OCR blocks log queues */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
          {ocrBoxes.length === 0 ? (
            <div className="text-center text-slate-500 italic text-xs py-8">No text blocks detected on this page.</div>
          ) : (
            ocrBoxes.map((b) => {
              const isActive = b.id === activeBoxId;
              return (
                <div
                  key={b.id}
                  onClick={() => setActiveBoxId(b.id)}
                  className={`p-4 rounded-xl border transition-all duration-155 cursor-pointer flex flex-col gap-2.5 ${
                    isActive 
                      ? 'border-[#8b5cf6] bg-[#8b5cf6]/5 shadow-inner' 
                      : 'border-outline-variant bg-surface-container/20 hover:border-secondary hover:bg-surface-container/40'
                  }`}
                >
                  <div className="flex justify-between items-center select-none">
                    <span className={`text-[10px] custom-font-mono font-black ${
                      isActive ? 'text-[#a78bfa]' : 'text-slate-400'
                    }`}>
                      {b.index}
                    </span>
                    <span className={`text-[9px] custom-font-mono font-black ${
                      b.confidence > 90 ? 'text-secondary' : 'text-error'
                    }`}>
                      {b.confidence}% character score
                    </span>
                  </div>

                  <p className="text-slate-100 font-mono text-[11px] leading-relaxed break-words bg-slate-950/70 border border-slate-900 p-2.5 rounded-lg select-text font-medium">
                    {b.text || '[NO_CHARACTER_TRANSCRIPTION]'}
                  </p>

                  <div className="flex justify-between items-center text-[9px] custom-font-mono text-outline select-none">
                    <span>ENGINE: {b.lang.toUpperCase()}</span>
                    <span>TYPE: {b.entityType}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* OCR summary stats and metadata */}
        <div className="p-5 bg-surface-container border-t border-outline-variant shrink-0 flex flex-col gap-4 select-none">
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-xl bg-surface-container-high border border-outline-variant/60">
              <p className="text-[9px] custom-font-mono text-outline font-black uppercase mb-1">
                Parsed Words Stream
              </p>
              <div className="flex items-end gap-1.5">
                <span className="text-lg font-headline font-black text-secondary leading-none">
                  {ocrBoxes.reduce((acc, b) => acc + (b.text?.split(/\s+/).length || 0), 0)}
                </span>
                <Cpu className="w-3.5 h-3.5 text-secondary mb-0.5" />
              </div>
            </div>

            <div className="p-3 rounded-xl bg-surface-container-high border border-outline-variant/60">
              <p className="text-[9px] custom-font-mono text-outline font-black uppercase mb-1">
                Page Density Rating
              </p>
              <div className="flex items-end gap-1.5">
                <span className="text-lg font-headline font-black text-primary leading-none">
                  {ocrBoxes.length > 25 ? 'HIGH_INDEX' : ocrBoxes.length > 5 ? 'MEDIUM_INDEX' : 'LOW_INDEX'}
                </span>
                <TrendingUp className="w-3.5 h-3.5 text-primary mb-0.5" />
              </div>
            </div>
          </div>

          <button 
            onClick={() => {
              const textContent = ocrBoxes.map(b => b.text).join('\n');
              const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8;' });
              const link = document.createElement('a');
              link.href = URL.createObjectURL(blob);
              link.setAttribute('download', `${fileName}_page_${currentPage}_ocr.txt`);
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
            }}
            className="w-full flex items-center justify-center gap-2 bg-[#8b5cf6] text-white py-3.5 rounded-xl font-headline text-xs font-black tracking-widest uppercase hover:brightness-110 duration-100 active:scale-95 transition-all outline-none border-none cursor-pointer"
          >
            <Download className="w-4 h-4 stroke-[2.5]" />
            Download Page TXT
          </button>
        </div>
      </section>
    </div>
  );
};
