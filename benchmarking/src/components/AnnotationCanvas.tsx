import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Detection, CLASS_COLORS, CLASS_LABELS, DocLayClass } from '../types';
import { useAnnotation } from '../context/AnnotationContext';
import { Move, PenTool } from 'lucide-react';

interface AnnotationCanvasProps {
  detections: Detection[];
  imagePath: string;
}

export const AnnotationCanvas: React.FC<AnnotationCanvasProps> = ({ detections, imagePath }) => {
  const { hoveredDetectionIndex, setHoveredDetectionIndex, addDetection, activeModelTab } = useAnnotation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Mode: 'pan' or 'draw'
  const [canvasMode, setCanvasMode] = useState<'pan' | 'draw'>('pan');
  const [selectedDrawClass, setSelectedDrawClass] = useState<DocLayClass>('text');

  // Zoom and Pan stored in useRef (no re-render triggers)
  const zoom = useRef<number>(1);
  const panX = useRef<number>(0);
  const panY = useRef<number>(0);
  const [zoomDisplay, setZoomDisplay] = useState<number>(100);

  // Drawing States — ALL in refs to prevent re-render during drag
  const isDrawingRef = useRef<boolean>(false);
  const drawStartRef = useRef<{ x: number; y: number } | null>(null);
  const drawingBboxRef = useRef<[number, number, number, number] | null>(null);

  // Interaction state
  const isDragging = useRef<boolean>(false);
  const lastWheel = useRef<number>(0);
  const rafId = useRef<number>(0);

  // Store latest props/state in refs for stable redraw access
  const detectionsRef = useRef<Detection[]>(detections);
  const hoveredIndexRef = useRef<number | null>(hoveredDetectionIndex);
  const selectedDrawClassRef = useRef<DocLayClass>(selectedDrawClass);
  const canvasModeRef = useRef<'pan' | 'draw'>(canvasMode);

  // Keep refs synced with latest values
  detectionsRef.current = detections;
  hoveredIndexRef.current = hoveredDetectionIndex;
  selectedDrawClassRef.current = selectedDrawClass;
  canvasModeRef.current = canvasMode;

  // Canvas dimensions managed via ref + direct DOM mutation (no React re-render)
  const canvasWidth = useRef<number>(600);
  const canvasHeight = useRef<number>(750);

  // ─── Utility ───────────────────────────────────────────
  const hexToRgba = (hex: string, alpha: number) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // ─── Core Redraw (reads ONLY from refs, zero React deps) ───
  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !image.complete || image.naturalWidth === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear background
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    // Apply zoom and pan transformations
    ctx.translate(panX.current, panY.current);
    ctx.scale(zoom.current, zoom.current);

    // 1. Draw PDF page document
    ctx.drawImage(image, 0, 0);

    // 2. Draw existing bboxes
    const dets = detectionsRef.current;
    const hovIdx = hoveredIndexRef.current;
    dets.forEach((det, idx) => {
      const [x0, y0, x1, y1] = det.bbox;
      const w = x1 - x0;
      const h = y1 - y0;
      const color = CLASS_COLORS[det.type] || '#46f1c5';
      const isHovered = hovIdx === idx;

      const alpha = isHovered ? 0.35 : 0.15;
      ctx.fillStyle = hexToRgba(color, alpha);
      ctx.fillRect(x0, y0, w, h);

      ctx.lineWidth = isHovered ? 3 : 1.5;
      ctx.strokeStyle = color;
      ctx.strokeRect(x0, y0, w, h);

      if (w > 30 || isHovered) {
        ctx.fillStyle = color;
        const fontHeight = Math.max(10, Math.floor(11 / zoom.current));
        ctx.font = `bold ${fontHeight}px JetBrains Mono, monospace`;
        const textStr = `${CLASS_LABELS[det.type] || det.type.toUpperCase()} [${idx}]`;
        const textWidth = ctx.measureText(textStr).width;

        const labelHeight = fontHeight + 4;
        ctx.fillRect(x0 - 0.75, y0 - labelHeight, textWidth + 8, labelHeight);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(textStr, x0 + 3, y0 - 3);
      }
    });

    // 3. Draw active drawing dashed rectangle
    const drawBbox = drawingBboxRef.current;
    if (drawBbox) {
      const [x0, y0, x1, y1] = drawBbox;
      const drawColor = CLASS_COLORS[selectedDrawClassRef.current] || '#46f1c5';
      ctx.fillStyle = hexToRgba(drawColor, 0.25);
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);

      ctx.lineWidth = 2;
      ctx.strokeStyle = drawColor;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]); // reset

      // floating cursor tooltip values
      ctx.fillStyle = drawColor;
      const drawFontHeight = Math.max(10, Math.floor(11 / zoom.current));
      ctx.font = `bold ${drawFontHeight}px JetBrains Mono, monospace`;
      const sizeTag = `${CLASS_LABELS[selectedDrawClassRef.current].toUpperCase()} (${Math.round(x1 - x0)}x${Math.round(y1 - y0)})`;
      const sizeTagWidth = ctx.measureText(sizeTag).width;
      ctx.fillRect(x0, y0 - drawFontHeight - 4, sizeTagWidth + 8, drawFontHeight + 4);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(sizeTag, x0 + 4, y0 - 4);
    }

    ctx.restore();
  }, []); // Zero deps — reads everything from refs

  // Schedule a single-shot rAF redraw (coalesces multiple calls per frame)
  const scheduleRedraw = useCallback(() => {
    cancelAnimationFrame(rafId.current);
    rafId.current = requestAnimationFrame(redrawCanvas);
  }, [redrawCanvas]);

  // ─── EFFECT 1: Load image ONCE when imagePath changes ─────
  useEffect(() => {
    if (!imagePath) return;
    setLoading(true);

    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      setLoading(false);
      resetZoom(img);
      scheduleRedraw();
    };
    img.onerror = (e) => {
      console.error('Failed to load canvas image:', imagePath, e);
      setLoading(false);
    };
    img.src = imagePath;
  }, [imagePath, scheduleRedraw]);

  const resetZoom = (loadedImg?: HTMLImageElement) => {
    const activeImage = loadedImg || imageRef.current;
    if (!activeImage || !containerRef.current) return;

    const w = containerRef.current.clientWidth || 600;
    const h = containerRef.current.clientHeight || 750;

    const scaleX = (w - 60) / activeImage.width;
    const scaleY = (h - 60) / activeImage.height;
    const fitScale = Math.min(scaleX, scaleY, 1);

    zoom.current = fitScale;
    panX.current = (w - activeImage.width * fitScale) / 2;
    panY.current = (h - activeImage.height * fitScale) / 2;
    setZoomDisplay(Math.round(fitScale * 100));
  };

  // ─── EFFECT 2: Resize canvas buffer via DOM (no React re-render) ───
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const newW = Math.floor(entry.contentRect.width);
        const newH = Math.floor(containerRef.current?.clientHeight || 650);

        // Only update if actually changed
        if (newW === canvasWidth.current && newH === canvasHeight.current) return;
        canvasWidth.current = newW;
        canvasHeight.current = newH;

        // Direct DOM mutation — no setState, no re-render
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.width = newW;
          canvas.height = newH;
          scheduleRedraw();
        }
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [scheduleRedraw]);

  // ─── EFFECT 3: Redraw when detections or hover changes ───
  useEffect(() => {
    scheduleRedraw();
  }, [detections, hoveredDetectionIndex, scheduleRedraw]);

  // ─── Canvas coordinate conversion ───
  const getCanvasCoords = (clientX: number, clientY: number) => {
    if (!canvasRef.current || !imageRef.current) return null;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const imgX = (x - panX.current) / zoom.current;
    const imgY = (y - panY.current) / zoom.current;
    return { x: imgX, y: imgY };
  };

  // ─── Mouse Handlers ───
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (canvasModeRef.current === 'pan') {
      if (isDragging.current) {
        const dx = e.clientX - (e.currentTarget as any).__lastX;
        const dy = e.clientY - (e.currentTarget as any).__lastY;
        panX.current += dx;
        panY.current += dy;
        (e.currentTarget as any).__lastX = e.clientX;
        (e.currentTarget as any).__lastY = e.clientY;
        scheduleRedraw();
        return;
      }

      const coords = getCanvasCoords(e.clientX, e.clientY);
      if (!coords) return;

      let foundIdx: number | null = null;
      const dets = detectionsRef.current;
      for (let i = dets.length - 1; i >= 0; i--) {
        const [x0, y0, x1, y1] = dets[i].bbox;
        if (coords.x >= x0 && coords.x <= x1 && coords.y >= y0 && coords.y <= y1) {
          foundIdx = i;
          break;
        }
      }

      if (foundIdx !== hoveredIndexRef.current) {
        setHoveredDetectionIndex(foundIdx);
      }
    } else if (canvasModeRef.current === 'draw') {
      if (isDrawingRef.current && drawStartRef.current) {
        const currentCoords = getCanvasCoords(e.clientX, e.clientY);
        if (currentCoords) {
          drawingBboxRef.current = [
            Math.min(drawStartRef.current.x, currentCoords.x),
            Math.min(drawStartRef.current.y, currentCoords.y),
            Math.max(drawStartRef.current.x, currentCoords.x),
            Math.max(drawStartRef.current.y, currentCoords.y)
          ];
          scheduleRedraw();
        }
      }
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (canvasModeRef.current === 'pan') {
      if (e.button === 0 && hoveredIndexRef.current === null) {
        isDragging.current = true;
        (e.currentTarget as any).__lastX = e.clientX;
        (e.currentTarget as any).__lastY = e.clientY;
      }
    } else if (canvasModeRef.current === 'draw') {
      if (activeModelTab !== 'GT') {
        alert('You must be on the "Current GT (Working)" tab to draw boxes.');
        return;
      }
      if (e.button === 0) {
        const coords = getCanvasCoords(e.clientX, e.clientY);
        if (coords) {
          isDrawingRef.current = true;
          drawStartRef.current = coords;
          drawingBboxRef.current = [coords.x, coords.y, coords.x, coords.y];
        }
      }
    }
  };

  const handleMouseUp = () => {
    if (canvasModeRef.current === 'pan') {
      isDragging.current = false;
    } else if (canvasModeRef.current === 'draw') {
      if (isDrawingRef.current && drawingBboxRef.current) {
        const [x0, y0, x1, y1] = drawingBboxRef.current;
        const w = x1 - x0;
        const h = y1 - y0;

        // Verify minimum drag threshold
        if (w > 6 && h > 6) {
          addDetection(selectedDrawClassRef.current, [x0, y0, x1, y1]);
          // Issue 1: Auto-close class picker & return to PAN mode
          setCanvasMode('pan');
        }
      }
      isDrawingRef.current = false;
      drawStartRef.current = null;
      drawingBboxRef.current = null;
      scheduleRedraw();
    }
  };

  const handleMouseLeave = () => {
    isDragging.current = false;
    if (isDrawingRef.current) {
      isDrawingRef.current = false;
      drawStartRef.current = null;
      drawingBboxRef.current = null;
      scheduleRedraw();
    }
    setHoveredDetectionIndex(null);
  };

  // ─── Scroll Zoom ───
  const ZOOM_STEP = 0.12;
  const ZOOM_MIN = 0.15;
  const ZOOM_MAX = 6.0;

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();

    // Throttle: max ~30 zoom steps/sec
    const now = performance.now();
    if (now - lastWheel.current < 32) return;
    lastWheel.current = now;

    // Normalize: use ONLY the direction sign, ignore magnitude
    const dir = Math.sign(e.deltaY);  // -1 = zoom in, +1 = zoom out

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    // Cursor position relative to canvas element
    const cursorX = e.clientX - rect.left;
    const cursorY = e.clientY - rect.top;

    const prevZoom = zoom.current;
    const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, prevZoom * (1 - dir * ZOOM_STEP)));

    // Zoom toward cursor: adjust pan so point under cursor stays fixed
    const ratio = newZoom / prevZoom;
    panX.current = cursorX - ratio * (cursorX - panX.current);
    panY.current = cursorY - ratio * (cursorY - panY.current);
    zoom.current = newZoom;

    setZoomDisplay(Math.round(newZoom * 100));
    scheduleRedraw();
  }, [scheduleRedraw]);

  // Attach wheel listener with cleanup
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [handleWheel]);

  // ─── Zoom button handlers ───
  const applyZoom = useCallback((newZoom: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const clamped = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newZoom));
    // Zoom toward canvas center when using buttons
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const ratio = clamped / zoom.current;
    panX.current = cx - ratio * (cx - panX.current);
    panY.current = cy - ratio * (cy - panY.current);
    zoom.current = clamped;
    setZoomDisplay(Math.round(clamped * 100));
    scheduleRedraw();
  }, [scheduleRedraw]);

  const handleZoomIn = () => applyZoom(zoom.current + ZOOM_STEP);
  const handleZoomOut = () => applyZoom(zoom.current - ZOOM_STEP);
  const handleZoomReset = () => {
    resetZoom();
    scheduleRedraw();
  };

  // ─── Cleanup rAF on unmount ───
  useEffect(() => {
    return () => cancelAnimationFrame(rafId.current);
  }, []);

  // Zoom button style constant
  const zoomBtnStyle: React.CSSProperties = {
    background: 'transparent',
    border: 'none',
    color: '#aaaaaa',
    cursor: 'pointer',
    padding: '4px',
    borderRadius: '5px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'color 0.12s ease',
    outline: 'none',
  };

  return (
    <div
      id="canvas_container_panel"
      ref={containerRef}
      className="flex-1 w-full bg-surface-dim border border-border relative overflow-hidden flex items-center justify-center bg-grid-subtle transition-all h-[650px]"
      style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden', background: '#111' }}
    >
      {loading ? (
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
          <p className="font-mono text-xs text-text-muted">LOADING SOURCE DOCUMENT IMAGES...</p>
        </div>
      ) : (
        <>
          <canvas
            id="clinical_annotation_canvas"
            ref={canvasRef}
            width={canvasWidth.current}
            height={canvasHeight.current}
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            style={{
              cursor: canvasMode === 'draw' ? 'crosshair' : isDragging.current ? 'grabbing' : hoveredDetectionIndex !== null ? 'pointer' : 'grab',
              display: 'block',
            }}
          />

          {/* ── Zoom Controls ─────────────────────────────── */}
          <div style={{
            position: 'absolute',
            bottom: '14px',
            right: '14px',
            display: 'flex',
            alignItems: 'center',
            gap: '2px',
            background: 'rgba(14, 14, 14, 0.88)',
            border: '1px solid #222',
            borderRadius: '8px',
            padding: '5px 8px',
            zIndex: 20,
            backdropFilter: 'blur(6px)',
            boxShadow: '0 2px 12px rgba(0,0,0,0.5)',
          }}>

            {/* Zoom Out button */}
            <button onClick={handleZoomOut} title="Zoom Out (−)"
              style={zoomBtnStyle} onMouseEnter={e => (e.currentTarget.style.color='#00ffcc')}
              onMouseLeave={e => (e.currentTarget.style.color='#aaa')}>
              <svg width="17" height="17" viewBox="0 0 24 24"
                   fill="none" stroke="currentColor" strokeWidth="2.2"
                   strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="7.5"/>
                <line x1="21" y1="21" x2="16.2" y2="16.2"/>
                <line x1="7.5" y1="11" x2="14.5" y2="11"/>
              </svg>
            </button>

            {/* Zoom percentage display */}
            <span style={{
              color: '#00ffcc',
              fontSize: '11px',
              fontFamily: '"JetBrains Mono", "Fira Code", monospace',
              fontWeight: 600,
              minWidth: '44px',
              textAlign: 'center',
              userSelect: 'none',
              letterSpacing: '0.03em',
            }}>
              {zoomDisplay}%
            </span>

            {/* Zoom In button */}
            <button onClick={handleZoomIn} title="Zoom In (+)"
              style={zoomBtnStyle} onMouseEnter={e => (e.currentTarget.style.color='#00ffcc')}
              onMouseLeave={e => (e.currentTarget.style.color='#aaa')}>
              <svg width="17" height="17" viewBox="0 0 24 24"
                   fill="none" stroke="currentColor" strokeWidth="2.2"
                   strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="7.5"/>
                <line x1="21" y1="21" x2="16.2" y2="16.2"/>
                <line x1="11" y1="7.5" x2="11" y2="14.5"/>
                <line x1="7.5" y1="11" x2="14.5" y2="11"/>
              </svg>
            </button>

            {/* Divider */}
            <div style={{ width:'1px', height:'16px', background:'#2a2a2a', margin:'0 3px' }}/>

            {/* Reset 1:1 button */}
            <button onClick={handleZoomReset} title="Reset Zoom (Fit)"
              style={{ ...zoomBtnStyle, fontSize:'10px', fontFamily:'monospace',
                       padding:'3px 7px', letterSpacing:'0.05em' }}
              onMouseEnter={e => (e.currentTarget.style.color='#00ffcc')}
              onMouseLeave={e => (e.currentTarget.style.color='#888')}>
              FIT
            </button>
          </div>
        </>
      )}

      {/* Modern Canvas Interaction Modes Floating Menu */}
      <div className="absolute top-4 left-4 flex gap-1 z-20 bg-surface/90 backdrop-blur border border-border p-1 shadow-md">
        <button
          id="pan_mode_toggle_btn"
          onClick={() => setCanvasMode('pan')}
          className={`flex items-center gap-1.5 px-3 py-1.5 font-mono text-xs cursor-pointer transition-all ${
            canvasMode === 'pan' ? 'bg-primary text-on-primary font-bold' : 'text-on-surface hover:bg-surface-container'
          }`}
        >
          <Move className="w-3.5 h-3.5" />
          PAN / ZOOM
        </button>
        <button
          id="draw_mode_toggle_btn"
          onClick={() => {
            if (activeModelTab !== 'GT') {
              alert('Please switch the active model layer to "Current GT (Working)" to draw new bboxes.');
              return;
            }
            setCanvasMode('draw');
          }}
          className={`flex items-center gap-1.5 px-3 py-1.5 font-mono text-xs cursor-pointer transition-all ${
            canvasMode === 'draw' ? 'bg-primary text-on-primary font-bold' : 'text-on-surface hover:bg-surface-container'
          }`}
        >
          <PenTool className="w-3.5 h-3.5" />
          DRAW BOX MODE
        </button>
      </div>

      {/* Draw Category Selector Tooltip */}
      {canvasMode === 'draw' && (
        <div className="absolute top-16 left-4 bg-surface-container-high/95 backdrop-blur border border-border p-2.5 shadow-xl max-w-sm flex flex-col gap-2 z-20">
          <span className="font-mono text-[9px] text-text-muted uppercase tracking-wider block font-bold">Category for drawn box:</span>
          <div className="grid grid-cols-2 gap-1 max-h-36 overflow-y-auto pr-1">
            {(['title', 'section_header', 'text', 'list_item', 'table', 'picture', 'caption', 'footnote', 'formula', 'page_header', 'page_footer'] as DocLayClass[]).map((cls) => (
              <button
                key={cls}
                onClick={() => setSelectedDrawClass(cls)}
                className={`flex items-center gap-1.5 px-2 py-1 border text-left font-mono text-[10px] cursor-pointer transition-all rounded-xs ${
                  selectedDrawClass === cls
                    ? 'border-primary bg-primary/10 text-primary font-bold shadow-sm'
                    : 'border-border/60 hover:border-primary text-on-surface'
                }`}
              >
                <div className="w-2 h-2 rounded-xs" style={{ backgroundColor: CLASS_COLORS[cls] }} />
                {CLASS_LABELS[cls]}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
