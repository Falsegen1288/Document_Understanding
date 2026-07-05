import React, { useMemo } from 'react';
import { X, Award, AlertTriangle, Eye, Activity } from 'lucide-react';
import { Detection, CLASS_COLORS, CLASS_LABELS } from '../types';

interface EvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
  referenceDetections: Detection[]; // Usually current Ground Truth (GT)
  predictedDetections: Detection[]; // e.g. ADE, DocLayoutYOLO, or Nemotron
  modelName: string;
  pageWidth: number;
  pageHeight: number;
}

export const EvaluationModal: React.FC<EvaluationModalProps> = ({
  isOpen,
  onClose,
  referenceDetections,
  predictedDetections,
  modelName,
  pageWidth = 1275,
  pageHeight = 1650
}) => {
  if (!isOpen) return null;

  // ── Helper: IoU calculation ─────────────────────────
  const computeIoU = (b1: [number, number, number, number], b2: [number, number, number, number]) => {
    const ix0 = Math.max(b1[0], b2[0]);
    const iy0 = Math.max(b1[1], b2[1]);
    const ix1 = Math.min(b1[2], b2[2]);
    const iy1 = Math.min(b1[3], b2[3]);
    
    const inter = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);
    if (inter === 0) return 0;
    
    const a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    const a2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    return inter / (a1 + a2 - inter);
  };

  // ── Helper: Reading Order Assignment ──────────────────
  const assignReadingOrder = (dets: Detection[], rowBand = 40) => {
    if (dets.length === 0) return [];
    
    const detsWithCentroids = dets.map((d, i) => {
      const [x0, y0, x1, y1] = d.bbox;
      return {
        det: d,
        idx: i,
        cx: (x0 + x1) / 2,
        cy: (y0 + y1) / 2
      };
    });
    
    // Sort by y center first
    detsWithCentroids.sort((a, b) => a.cy - b.cy);
    
    const ordered: typeof detsWithCentroids = [];
    let currentBandY: number | null = null;
    let currentBand: typeof detsWithCentroids = [];
    
    for (const item of detsWithCentroids) {
      if (currentBandY === null || Math.abs(item.cy - currentBandY) <= rowBand) {
        currentBand.push(item);
        currentBandY = currentBandY === null 
          ? item.cy 
          : (currentBandY * (currentBand.length - 1) + item.cy) / currentBand.length;
      } else {
        currentBand.sort((a, b) => a.cx - b.cx); // sort left to right
        ordered.push(...currentBand);
        currentBand = [item];
        currentBandY = item.cy;
      }
    }
    
    if (currentBand.length > 0) {
      currentBand.sort((a, b) => a.cx - b.cx);
      ordered.push(...currentBand);
    }
    
    return ordered.map((item, orderIdx) => ({
      det: item.det,
      originalIdx: item.idx,
      rank: orderIdx
    }));
  };

  const metrics = useMemo(() => {
    const refN = referenceDetections.length;
    const predN = predictedDetections.length;
    
    // 1. Build IoU Matrix
    const iouMatrix: number[][] = Array(refN).fill(0).map(() => Array(predN).fill(0));
    for (let i = 0; i < refN; i++) {
      for (let j = 0; j < predN; j++) {
        iouMatrix[i][j] = computeIoU(referenceDetections[i].bbox, predictedDetections[j].bbox);
      }
    }

    // ── Hungarian/Greedy Optimal Matching ──────────────────
    // Sort all matches by IoU in descending order
    const allPairs: { i: number; j: number; iou: number }[] = [];
    for (let i = 0; i < refN; i++) {
      for (let j = 0; j < predN; j++) {
        if (iouMatrix[i][j] >= 0.10) { // minimum threshold for matching
          allPairs.push({ i, j, iou: iouMatrix[i][j] });
        }
      }
    }
    allPairs.sort((a, b) => b.iou - a.iou);
    
    const matchedRef = new Set<number>();
    const matchedPred = new Set<number>();
    const matches: { i: number; j: number; iou: number }[] = [];
    
    for (const pair of allPairs) {
      if (!matchedRef.has(pair.i) && !matchedPred.has(pair.j)) {
        matchedRef.add(pair.i);
        matchedPred.add(pair.j);
        matches.push(pair);
      }
    }

    // ── Layer 1: Geometric & Categorical Alignment ─────────
    const matchesAt50 = matches.filter(m => m.iou >= 0.50);
    const precision = predN > 0 ? matchesAt50.length / predN : 0;
    const recall = refN > 0 ? matchesAt50.length / refN : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const meanIoU = matches.length > 0 ? matches.reduce((sum, m) => sum + m.iou, 0) / matches.length : 0;
    
    const classAgreements = matchesAt50.filter(m => referenceDetections[m.i].type === predictedDetections[m.j].type);
    const classAccuracy = matchesAt50.length > 0 ? classAgreements.length / matchesAt50.length : 0;

    // ── Layer 2: pyCOTe Spatial Quality (Raster Grid approx) ─
    // We create a fast grid of size 120 x 150 representing the document page
    const gridCols = 120;
    const gridRows = 150;
    const cellW = pageWidth / gridCols;
    const cellH = pageHeight / gridRows;
    const pageCells = gridCols * gridRows;
    
    // Helper: Rasterize box to cells list
    const getCells = (bbox: [number, number, number, number]) => {
      const startCol = Math.max(0, Math.floor(bbox[0] / cellW));
      const endCol = Math.min(gridCols - 1, Math.floor(bbox[2] / cellW));
      const startRow = Math.max(0, Math.floor(bbox[1] / cellH));
      const endRow = Math.min(gridRows - 1, Math.floor(bbox[3] / cellH));
      
      const cells: number[] = [];
      for (let r = startRow; r <= endRow; r++) {
        for (let c = startCol; c <= endCol; c++) {
          cells.push(r * gridCols + c);
        }
      }
      return cells;
    };
    
    const refBboxCells = referenceDetections.map(d => getCells(d.bbox));
    const predBboxCells = predictedDetections.map(d => getCells(d.bbox));
    
    // Combined union of reference cells
    const refUnion = new Set<number>();
    refBboxCells.forEach(cells => cells.forEach(c => refUnion.add(c)));
    
    // Assign each prediction to its best-matching SSU reference
    const predToSsu = Array(predN).fill(-1);
    for (let j = 0; j < predN; j++) {
      let bestSsu = -1;
      let maxOverlap = 0;
      const pm = new Set(predBboxCells[j]);
      
      for (let i = 0; i < refN; i++) {
        let overlap = 0;
        refBboxCells[i].forEach(c => {
          if (pm.has(c)) overlap++;
        });
        if (overlap > maxOverlap) {
          maxOverlap = overlap;
          bestSsu = i;
        }
      }
      if (bestSsu >= 0) {
        predToSsu[j] = bestSsu;
      }
    }
    
    // Coverage: % of ref cells covered by assigned predictions
    const coverageScores: number[] = [];
    for (let i = 0; i < refN; i++) {
      const ssuCells = refBboxCells[i];
      if (ssuCells.length === 0) continue;
      
      const ssuSet = new Set(ssuCells);
      const coveredSet = new Set<number>();
      
      // Get all predictions assigned to this reference
      const assignedPreds = predBboxCells.filter((_, j) => predToSsu[j] === i);
      assignedPreds.forEach(cells => cells.forEach(c => {
        if (ssuSet.has(c)) coveredSet.add(c);
      }));
      
      coverageScores.push(coveredSet.size / ssuCells.length);
    }
    const coverageVal = coverageScores.length > 0 ? coverageScores.reduce((sum, v) => sum + v, 0) / coverageScores.length : 0;
    
    // Overlap: redundant double-coverage inside SSU
    const overlapScores: number[] = [];
    for (let i = 0; i < refN; i++) {
      const ssuCells = refBboxCells[i];
      if (ssuCells.length === 0) continue;
      
      const ssuSet = new Set(ssuCells);
      const cellCount: Record<number, number> = {};
      
      const assignedPreds = predBboxCells.filter((_, j) => predToSsu[j] === i);
      if (assignedPreds.length < 2) {
        overlapScores.push(0);
        continue;
      }
      
      assignedPreds.forEach(cells => cells.forEach(c => {
        if (ssuSet.has(c)) {
          cellCount[c] = (cellCount[c] || 0) + 1;
        }
      }));
      
      const redundantCells = Object.values(cellCount).filter(count => count > 1).length;
      overlapScores.push(redundantCells / ssuCells.length);
    }
    const overlapVal = overlapScores.length > 0 ? overlapScores.reduce((sum, v) => sum + v, 0) / overlapScores.length : 0;
    
    // Trespass: prediction bleeding into other SSUs
    const trespassScores: number[] = [];
    for (let j = 0; j < predN; j++) {
      const ssuIdx = predToSsu[j];
      if (ssuIdx === -1) continue;
      
      const ssuCells = refBboxCells[ssuIdx];
      if (ssuCells.length === 0) continue;
      
      // Combine other SSU cells
      const otherSsuSet = new Set<number>();
      refBboxCells.forEach((cells, i) => {
        if (i !== ssuIdx) cells.forEach(c => otherSsuSet.add(c));
      });
      
      let trespassCount = 0;
      predBboxCells[j].forEach(c => {
        if (otherSsuSet.has(c)) trespassCount++;
      });
      
      trespassScores.push(trespassCount / ssuCells.length);
    }
    const trespassVal = trespassScores.length > 0 ? trespassScores.reduce((sum, v) => sum + v, 0) / trespassScores.length : 0;
    
    // Excess: predictions landing in non-SSU background space
    const predUnion = new Set<number>();
    predBboxCells.forEach(cells => cells.forEach(c => predUnion.add(c)));
    
    const nonSsuArea = pageCells - refUnion.size;
    let excessVal = 0;
    if (nonSsuArea > 0) {
      let excessCells = 0;
      predUnion.forEach(c => {
        if (!refUnion.has(c)) excessCells++;
      });
      excessVal = excessCells / nonSsuArea;
    }

    // ── Layer 3: LED Structural Errors ─────────────────────
    const ledErrors: Record<string, number> = {
      Missing: 0,
      Hallucination: 0,
      'Size-Error': 0,
      Split: 0,
      Merge: 0,
      Overlap: 0,
      Duplicate: 0,
      Misclassification: 0
    };
    
    // Duplicate: pred boxes overlapping >= 85%
    for (let j1 = 0; j1 < predN; j1++) {
      for (let j2 = j1 + 1; j2 < predN; j2++) {
        if (computeIoU(predictedDetections[j1].bbox, predictedDetections[j2].bbox) >= 0.85) {
          ledErrors.Duplicate++;
        }
      }
    }
    
    // Missing: ref box with max IoU < 0.10
    for (let i = 0; i < refN; i++) {
      let maxIoU = 0;
      for (let j = 0; j < predN; j++) {
        maxIoU = Math.max(maxIoU, iouMatrix[i][j]);
      }
      if (maxIoU < 0.10) ledErrors.Missing++;
    }
    
    // Hallucination: pred box with max IoU < 0.10
    for (let j = 0; j < predN; j++) {
      let maxIoU = 0;
      for (let i = 0; i < refN; i++) {
        maxIoU = Math.max(maxIoU, iouMatrix[i][j]);
      }
      if (maxIoU < 0.10) ledErrors.Hallucination++;
    }
    
    // Misclassification & Size-Error on optimal matches
    matches.forEach(m => {
      if (m.iou >= 0.50 && referenceDetections[m.i].type !== predictedDetections[m.j].type) {
        ledErrors.Misclassification++;
      }
      if (m.iou >= 0.10 && m.iou < 0.50) {
        ledErrors['Size-Error']++;
      }
    });
    
    // Split: ref mapped to multiple predictions
    const refToPredsCount = Array(refN).fill(0);
    matches.forEach(m => {
      if (m.iou >= 0.10) refToPredsCount[m.i]++;
    });
    // Count unmatched predictions overlapping ref
    for (let j = 0; j < predN; j++) {
      if (!matchedPred.has(j)) {
        for (let i = 0; i < refN; i++) {
          if (iouMatrix[i][j] >= 0.15) {
            refToPredsCount[i]++;
            break;
          }
        }
      }
    }
    ledErrors.Split = refToPredsCount.filter(cnt => cnt > 1).length;
    
    // Merge: pred overlapping multiple ref boxes
    const predToRefsCount = Array(predN).fill(0);
    matches.forEach(m => {
      if (m.iou >= 0.10) predToRefsCount[m.j]++;
    });
    for (let i = 0; i < refN; i++) {
      if (!matchedRef.has(i)) {
        for (let j = 0; j < predN; j++) {
          if (iouMatrix[i][j] >= 0.15) {
            predToRefsCount[j]++;
            break;
          }
        }
      }
    }
    ledErrors.Merge = predToRefsCount.filter(cnt => cnt > 1).length;
    
    // Overlap: pred overlapping pred (0.10 <= IoU < 0.85)
    for (let j1 = 0; j1 < predN; j1++) {
      for (let j2 = j1 + 1; j2 < predN; j2++) {
        const iou = computeIoU(predictedDetections[j1].bbox, predictedDetections[j2].bbox);
        if (iou >= 0.10 && iou < 0.85) {
          ledErrors.Overlap++;
        }
      }
    }

    // ── Layer 4: Reading Order ─────────────────────────────
    const refOrder = assignReadingOrder(referenceDetections);
    const predOrder = assignReadingOrder(predictedDetections);
    
    // Find optimal matched pairs between ref and pred using IoU >= 0.30
    const matchedPairs = matches.filter(m => m.iou >= 0.30);
    const nMatched = matchedPairs.length;
    
    let rokt = NaN;
    let roa = NaN;
    
    if (nMatched >= 2) {
      // Find ranks in common matched list
      const refRanks = matchedPairs.map(m => {
        const found = refOrder.find(o => o.originalIdx === m.i);
        return found ? found.rank : 0;
      });
      
      const predRanks = matchedPairs.map(m => {
        const found = predOrder.find(o => o.originalIdx === m.j);
        return found ? found.rank : 0;
      });
      
      let concordant = 0;
      let discordant = 0;
      for (let x = 0; x < nMatched; x++) {
        for (let y = x + 1; y < nMatched; y++) {
          const refDiff = refRanks[x] - refRanks[y];
          const predDiff = predRanks[x] - predRanks[y];
          if (refDiff * predDiff > 0) concordant++;
          else if (refDiff * predDiff < 0) discordant++;
        }
      }
      
      const denom = (nMatched * (nMatched - 1)) / 2;
      rokt = denom > 0 ? (concordant - discordant) / denom : 0;
      
      // Sort by ref ranks to compute ROA permutation order
      const combined = matchedPairs.map((m, idx) => ({
        refR: refRanks[idx],
        predR: predRanks[idx]
      })).sort((a, b) => a.refR - b.refR);
      
      let inversions = 0;
      for (let x = 0; x < nMatched - 1; x++) {
        if (combined[x].predR > combined[x + 1].predR) {
          inversions++;
        }
      }
      roa = 1.0 - inversions / (nMatched - 1);
    }

    // Class mismatch list
    const mismatches: { refType: string; predType: string; count: number }[] = [];
    const mismatchCount: Record<string, number> = {};
    matchesAt50.forEach(m => {
      const rType = referenceDetections[m.i].type;
      const pType = predictedDetections[m.j].type;
      if (rType !== pType) {
        const key = `${rType} ➔ ${pType}`;
        mismatchCount[key] = (mismatchCount[key] || 0) + 1;
      }
    });
    Object.entries(mismatchCount).forEach(([key, val]) => {
      const [refType, predType] = key.split(' ➔ ');
      mismatches.push({ refType, predType, count: val });
    });

    return {
      precision,
      recall,
      f1,
      meanIoU,
      classAccuracy,
      coverageVal,
      overlapVal,
      trespassVal,
      excessVal,
      ledErrors,
      rokt,
      roa,
      mismatches,
      nMatched
    };
  }, [referenceDetections, predictedDetections, pageWidth, pageHeight]);

  const getStatusBadge = (val: number, good: boolean) => {
    if (good) {
      return val > 0.7 
        ? <span className="bg-primary/20 text-primary border border-primary/30 px-1.5 py-0.5 rounded text-[10px] uppercase font-bold">Good</span>
        : <span className="bg-warning/20 text-warning border border-warning/30 px-1.5 py-0.5 rounded text-[10px] uppercase font-bold">Low</span>;
    } else {
      return val < 0.15 
        ? <span className="bg-primary/20 text-primary border border-primary/30 px-1.5 py-0.5 rounded text-[10px] uppercase font-bold">Good</span>
        : <span className="bg-error/20 text-error border border-error/30 px-1.5 py-0.5 rounded text-[10px] uppercase font-bold">High</span>;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-surface border border-border w-full max-w-4xl rounded-xs flex flex-col max-h-[90vh] shadow-2xl">
        {/* Header */}
        <header className="p-4 border-b border-border bg-surface-container-high flex justify-between items-center shrink-0">
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5 text-primary animate-pulse" />
            <div>
              <h2 className="font-mono text-sm font-semibold text-primary uppercase">Real-Time Layout Alignment Diagnostics</h2>
              <p className="font-mono text-[9px] text-text-muted uppercase tracking-widest mt-0.5">
                Current Ground Truth vs {modelName}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="text-text-muted hover:text-on-surface cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          
          {/* Layer 1 & 2 Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Layer 1: Geometric & Categorical */}
            <div className="bg-surface-container-low border border-border p-5 flex flex-col justify-between">
              <h3 className="font-mono text-xs text-primary font-bold uppercase tracking-wider border-b border-border pb-2 mb-4">
                Layer 1: Geometric Bipartite Agreement
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-background border border-border p-3">
                  <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider block">F1 Match Score</span>
                  <span className="font-mono text-xl font-bold text-primary mt-1 block">{(metrics.f1 * 100).toFixed(1)}%</span>
                </div>
                <div className="bg-background border border-border p-3">
                  <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider block">Class Agreement</span>
                  <span className="font-mono text-xl font-bold text-primary mt-1 block">{(metrics.classAccuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="bg-background border border-border p-3">
                  <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider block">Precision / Recall</span>
                  <span className="font-mono text-xs font-semibold text-on-surface mt-1 block">
                    P: {(metrics.precision * 100).toFixed(0)}% / R: {(metrics.recall * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="bg-background border border-border p-3">
                  <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider block">Mean overlap IoU</span>
                  <span className="font-mono text-xs font-semibold text-on-surface mt-1 block">{(metrics.meanIoU * 100).toFixed(1)}% IoU</span>
                </div>
              </div>
            </div>

            {/* Layer 2: pyCOTe Spatial Quality */}
            <div className="bg-surface-container-low border border-border p-5 flex flex-col justify-between">
              <h3 className="font-mono text-xs text-primary font-bold uppercase tracking-wider border-b border-border pb-2 mb-4">
                Layer 2: COTe Spatial Density (masks)
              </h3>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center font-mono text-xs">
                  <span className="text-text-muted">COVERAGE (Target &gt; 0.70)</span>
                  <div className="flex items-center gap-2">
                    <span className="text-on-surface font-semibold">{metrics.coverageVal.toFixed(3)}</span>
                    {getStatusBadge(metrics.coverageVal, true)}
                  </div>
                </div>
                <div className="flex justify-between items-center font-mono text-xs">
                  <span className="text-text-muted">TRESPASS (Bleed &lt; 0.15)</span>
                  <div className="flex items-center gap-2">
                    <span className="text-on-surface font-semibold">{metrics.trespassVal.toFixed(3)}</span>
                    {getStatusBadge(metrics.trespassVal, false)}
                  </div>
                </div>
                <div className="flex justify-between items-center font-mono text-xs">
                  <span className="text-text-muted">OVERLAP (Duplicates &lt; 0.15)</span>
                  <div className="flex items-center gap-2">
                    <span className="text-on-surface font-semibold">{metrics.overlapVal.toFixed(3)}</span>
                    {getStatusBadge(metrics.overlapVal, false)}
                  </div>
                </div>
                <div className="flex justify-between items-center font-mono text-xs">
                  <span className="text-text-muted">EXCESS (Background &lt; 0.15)</span>
                  <div className="flex items-center gap-2">
                    <span className="text-on-surface font-semibold">{metrics.excessVal.toFixed(3)}</span>
                    {getStatusBadge(metrics.excessVal, false)}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Layer 3: LED Structural Errors */}
          <div className="bg-surface-container-low border border-border p-5">
            <h3 className="font-mono text-xs text-primary font-bold uppercase tracking-wider border-b border-border pb-2 mb-4">
              Layer 3: LED Structural Errors Classification
            </h3>
            
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {Object.entries(metrics.ledErrors as Record<string, number>).map(([errName, count]) => (
                <div key={errName} className="bg-background border border-border p-3 flex justify-between items-center font-mono">
                  <div>
                    <span className="text-[9px] text-text-muted uppercase tracking-wider block">{errName}</span>
                    <span className={`text-xl font-bold mt-1 block ${count > 0 ? 'text-warning' : 'text-text-muted'}`}>
                      {count}
                    </span>
                  </div>
                  {count > 0 && <AlertTriangle className="w-4 h-4 text-warning" />}
                </div>
              ))}
            </div>
            
            {metrics.mismatches.length > 0 && (
              <div className="mt-4 bg-background border border-border p-3 font-mono text-[11px] space-y-1 max-h-24 overflow-y-auto">
                <span className="text-text-muted block uppercase tracking-wider font-semibold mb-1">Mismatched Label Pairs (IoU &gt;= 0.50):</span>
                {metrics.mismatches.map((m, idx) => (
                  <div key={idx} className="flex justify-between items-center">
                    <span>
                      <span className="text-error font-semibold">{m.refType}</span>
                      <span className="text-text-muted px-2">➔</span>
                      <span className="text-primary font-semibold">{m.predType}</span>
                    </span>
                    <span className="text-warning-light font-bold">×{m.count}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Layer 4: Reading Order */}
          <div className="bg-surface-container-low border border-border p-5">
            <h3 className="font-mono text-xs text-primary font-bold uppercase tracking-wider border-b border-border pb-2 mb-4">
              Layer 4: Reading Order Agreement
            </h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-center">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-background border border-border p-3">
                  <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider block">ROKT (Kendall's Tau)</span>
                  <span className="font-mono text-xl font-bold text-primary mt-1 block">
                    {isNaN(metrics.rokt) ? 'N/A' : metrics.rokt.toFixed(3)}
                  </span>
                </div>
                <div className="bg-background border border-border p-3">
                  <span className="text-[9px] font-mono text-text-muted uppercase tracking-wider block">ROA Accuracy</span>
                  <span className="font-mono text-xl font-bold text-primary mt-1 block">
                    {isNaN(metrics.roa) ? 'N/A' : (metrics.roa * 100).toFixed(0) + '%'}
                  </span>
                </div>
              </div>
              <div className="font-mono text-xs text-text-muted leading-relaxed">
                <p>
                  Reading Order Kendall's Tau (ROKT) measures the sequence rank correlation of matched elements on the page. 
                  A ROKT &gt; 0.7 indicates strong agreement on structural reading flow.
                </p>
                <p className="mt-2 text-[11px] italic">
                  Matched segments evaluated: {metrics.nMatched} / ref: {referenceDetections.length} / pred: {predictedDetections.length}
                </p>
              </div>
            </div>
          </div>

        </div>

        {/* Footer */}
        <footer className="p-4 border-t border-border bg-surface-container-low flex justify-end shrink-0">
          <button 
            onClick={onClose}
            className="bg-primary text-on-primary font-mono text-xs font-bold uppercase tracking-wider px-5 py-2 cursor-pointer hover:brightness-110 active:scale-95 transition-all"
          >
            Acknowledge Diagnostics
          </button>
        </footer>
      </div>
    </div>
  );
};
