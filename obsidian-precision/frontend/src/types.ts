/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type ActiveView = 'landing' | 'ingestion' | 'layout' | 'ocr' | 'table' | 'vision';

export interface RunItem {
  id: string;
  runId: string;
  fileName: string;
  status: 'running' | 'done' | 'failed';
  progress: number;
  duration: string;
  elapsed?: string;
  nodesCount?: number;
  message?: string;
  timestamp: string;
  layoutAlgo?: string;
  ocrAlgo?: string;
  tableAlgo?: string;
  figureAlgo?: string;
  jobId?: string;
}

export interface OCRBox {
  id: string;
  index: string;
  label: string;
  confidence: number;
  pos: string;
  text: string;
  top: string; // percentage for positioning
  left: string; // percentage for positioning
  width: string; // percentage
  height: string; // percentage
  entityType?: string;
  lang?: string;
  isCustom?: boolean;
}

export interface TableRow {
  indexId: string;
  fiscalPeriod: string;
  metricStream: string;
  projectedRev: string;
  actualYield: string;
  varianceDelta: string;
  isMerged?: boolean;
  mergedText?: string;
  isWarning?: boolean;
  warningText?: string;
  span?: number;
}

export interface ExtractionStats {
  confidence: string;
  structureLevels: number;
  manualFlags: number;
  latencyMs: number;
}
