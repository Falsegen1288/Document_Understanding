/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { OCRBox, TableRow, RunItem } from './types';

export const USER_PROFILES = {
  vision: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDzNjN-1fBtUr23ACo0koEeDJamha4s5P5dhFptiMvgneJk7MNtugnb5ktTQwWCIUBgaAJ66iuZO0gKFImrDNJygf5hjwwIQfEb0-iSTIhADeYc17sWOo7nSpxpvN-F8DYrsLgi5Gp7-0b1kI5NYtKbOJVtSVKo-X3-ouT5T1zOfZDJbEjUQ2I48nXmpuJzYea4qrZlUfIeW4g8M1EuKiuDBuKJzWUbbvGqr_Qg95qGbqhyDMr1LVEoPxOKfLp2-aa2PyTPy15P3CQ',
  table: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCXbfytyoQUcTq4i8rr_0YJSzF0eYE67N0i0getCgRHVT0kudjMygE1v2W728yGV0wc_uBEwmdd1Y4AHHzV5r03VCVwIHDnQI9fizpbOQpaSch6hecfoNqRu4s9m-ArRu8Tlkv2WSUU3cRSgBmNGKPiqmC3pbp74_0NHofsuNKsyoQk__oAlrJmk4PcxhgieUwtfiJnVNYxyeBxActEKloUs0Dfw7r5i3FrQ9AkAIHZeMynMZVlcMnj4S6c9jrIUuADHE-SjIsm8ik',
  ocr: 'https://lh3.googleusercontent.com/aida-public/AB6AXuC6aWb7Ks22faZbem3MghDR8teSGG61I76OK_Z6RJAysJdDfkQxNYPBS_qDCYnK2f_jBEEEfw4rAvrSHerfS6XxM4D-9avgqh4dfzVHAuPbXVr0cVCPCpxbH-7iQLe7Syi8r-3xUg01vLVnGoRhsu3vNs0Jty9v8WEjJreKrIFheScGINExtTeSIoPI1_QO2WSf6Bl3zHzgQOwBB2YRuyDgasF8g7T3JWiddqfl0Tz5GnGjtxjHnlqp1dDwKznltu30L84jRr14r5M',
  layout: 'https://lh3.googleusercontent.com/aida-public/AB6AXuB0FlHpVBVQj0tyNsLdAB8xuI5ZRu6Jzx3lIOgES3nruZ8MQ_BI3hMnMBFLtLQsyfhvAuu8tT5Pkih63L-72DOvP1iIxHgziJ3j8ZnFACiUrfKuNhKXPnnKy-GGBvryrgvBy7m9O6h7-P-zXiUEDV7DxAjNYo8bhSaBrrK_QjZYQku4-6gGJD8WR33sjmmb-vJbLIuNusutvzGeKxlQ5QQKlFGTWxvwDBo3FDsn9D79h7curGh1_d12vUz3ORiwwuqF3W9OTpn1hbE',
  inspector: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCZ-XSeGtiNvpgd0eHYyLNbhzNIz_3KZb_khOlk3n3ICChGGLhiGsxYgFCIA7rscRF56O1fuwB0PExbipR_Ezn6jaWOpf99phgQcTrNvOSTd6ZcWlSkOj3lb6IVRqNM5e43ial9BAnnalPulKYFNgfrMxfB5zsllKdPw2Nxbu3vdJtfSyuczNSWUBJCqO95kZvOrs4Bu0rW2KVBXWWF3URcKMNz8-IrBNG1saBeWeeL8LGZUMknMGfNvVps4Amtc4lTFqPErSGqGlY',
};

export const INITIAL_RUNS: RunItem[] = [
  {
    id: 'run-1',
    runId: '#8921-X',
    fileName: 'SOURCE_DOC_089.PDF',
    status: 'done',
    progress: 100,
    duration: 'Duration: 5.4s',
    timestamp: '2026-06-02T10:28:11Z'
  },
  {
    id: 'run-2',
    runId: '#8919-A',
    fileName: 'FIN-Q4-AUDIT_v2.pdf',
    status: 'done',
    progress: 100,
    duration: 'Duration: 12.8s',
    nodesCount: 4209,
    message: 'Processed 4,209 nodes in cloud-mesh.',
    timestamp: '2026-06-02T10:15:33Z'
  },
  {
    id: 'run-3',
    runId: '#8918-B',
    fileName: 'LEGACY_SYSTEMS_RAW.TIFF',
    status: 'failed',
    progress: 100,
    duration: 'Terminated @ 8.1s',
    message: 'Timeout on Table Extraction Engine.',
    timestamp: '2026-06-02T10:02:11Z'
  },
  {
    id: 'run-4',
    runId: '#8915-C',
    fileName: 'TX-091_SCHEMATIC.PDF',
    status: 'done',
    progress: 100,
    duration: 'Duration: 1.2m',
    timestamp: '2026-06-02T09:44:00Z'
  }
];

export const INITIAL_TABLE_ROWS: TableRow[] = [
  {
    indexId: '#OX-7712',
    fiscalPeriod: '2024-Q1',
    metricStream: 'Core Infrastructure',
    projectedRev: '$12,440.00',
    actualYield: '$13,205.12',
    varianceDelta: '+6.15%'
  },
  {
    indexId: '#OX-7713',
    fiscalPeriod: '2024-Q1',
    metricStream: 'Unified Operational Expenditure',
    projectedRev: '',
    actualYield: '$8,900.00',
    varianceDelta: '-2.20%',
    isMerged: true,
    mergedText: 'Unified Operational Expenditure',
    span: 2
  },
  {
    indexId: '#OX-7714',
    fiscalPeriod: '2024-Q2',
    metricStream: 'Research & Dev',
    projectedRev: '$45,000.00',
    actualYield: '$44,120.90',
    varianceDelta: '-0.88%'
  },
  {
    indexId: '#OX-7715',
    fiscalPeriod: '2024-Q2',
    metricStream: 'Cloud Provisioning',
    projectedRev: '$102,400.00',
    actualYield: '$108,900.00',
    varianceDelta: '+5.21%'
  },
  {
    indexId: '#OX-7716',
    fiscalPeriod: '2024-Q3',
    metricStream: 'Legacy Systems',
    projectedRev: '',
    actualYield: '',
    varianceDelta: '',
    isWarning: true,
    warningText: 'Data Fragment: Manual alignment required for Reconciliation columns.',
    span: 4
  },
  {
    indexId: '#OX-7717',
    fiscalPeriod: '2024-Q3',
    metricStream: 'Marketing Acquisition',
    projectedRev: '$5,000.00',
    actualYield: '$4,980.00',
    varianceDelta: '-0.40%'
  },
  {
    indexId: '#OX-7718',
    fiscalPeriod: '2024-Q4',
    metricStream: 'Security Licensing',
    projectedRev: '$22,100.00',
    actualYield: '$24,600.50',
    varianceDelta: '+11.31%'
  },
  {
    indexId: '#OX-7719',
    fiscalPeriod: '2024-Q4',
    metricStream: 'Server Hardware',
    projectedRev: '$67,000.00',
    actualYield: '$61,000.00',
    varianceDelta: '-8.96%'
  }
];

export const INITIAL_OCR_BOXES: OCRBox[] = [
  {
    id: 'box-01',
    index: '01',
    label: 'TECHNICAL_SPEC_HEADER',
    confidence: 98.4,
    pos: '142, 88',
    text: 'TECHNICAL_SPECIFICATION_REV_B',
    top: '10%',
    left: '15%',
    width: '30%',
    height: '5%',
    entityType: 'HEADER',
    lang: 'EN_US'
  },
  {
    id: 'box-02',
    index: '02',
    label: 'DOCUMENT_TYPE',
    confidence: 99.1,
    pos: '142, 112',
    text: 'ENGINEERING_BLUEPRINT',
    top: '18%',
    left: '15%',
    width: '50%',
    height: '3%',
    entityType: 'META_DOC_TYPE',
    lang: 'EN_US'
  },
  {
    id: 'box-03',
    index: '03',
    label: 'TEXT_BLOCK_04',
    confidence: 94.2,
    pos: '240, 210',
    text: 'ALL DIMENSIONS IN MM UNLESS OTHERWISE SPECIFIED. TOLERANCES: +/- 0.05MM FOR ALL MACHINED SURFACES.',
    top: '25%',
    left: '15%',
    width: '70%',
    height: '20%',
    entityType: 'MEASURE_SPEC',
    lang: 'EN_US'
  },
  {
    id: 'box-04',
    index: '04',
    label: 'LOW_CONF_BLOCK',
    confidence: 72.8,
    pos: '142, 340',
    text: '[UNREADABLE_GLYPH] - POTENTIAL SCAN ARTIFACT DETECTED',
    top: '50%',
    left: '15%',
    width: '65%',
    height: '10%',
    entityType: 'SCAN_ARTIFACT',
    lang: 'EN_US'
  },
  {
    id: 'box-05',
    index: '05',
    label: 'MATERIAL_GRADE',
    confidence: 97.6,
    pos: '142, 380',
    text: 'MATERIAL: GRADE 5 TITANIUM ALLOY (TI-6AL-4V)',
    top: '65%',
    left: '15%',
    width: '70%',
    height: '25%',
    entityType: 'MATERIAL_SPEC',
    lang: 'EN_US'
  }
];

export const RAW_MARKDOWN_CONTENT = `| INDEX_ID | FISCAL_PERIOD | METRIC_STREAM | PROJECTED_REV | ACTUAL_YIELD | VARIANCE_DELTA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| #OX-7712 | 2024-Q1 | Core Infrastructure | $12,440.00 | $13,205.12 | +6.15% |
| #OX-7713 | 2024-Q1 | **Unified Operational Expenditure** || $8,900.00 | -2.20% |
| #OX-7714 | 2024-Q2 | Research & Dev | $45,000.00 | $44,120.90 | -0.88% |
| #OX-7715 | 2024-Q2 | Cloud Provisioning | $102,400.00 | $108,900.00 | +5.21% |
| #OX-7716 | 2024-Q3 | Legacy Systems | *Fragment Found* |||
| #OX-7717 | 2024-Q3 | Marketing Acquisition | $5,000.00 | $4,980.00 | -0.40% |
| #OX-7718 | 2024-Q4 | Security Licensing | $22,100.00 | $24,600.50 | +11.31% |
| #OX-7719 | 2024-Q4 | Server Hardware | $67,000.00 | $61,000.00 | -8.96% |

---
**Metadata**
- Engine: Multimodal_v4.2.0
- Extraction_Confidence: 0.9942
- Processing_Timestamp: 2026-06-02T10:32:11Z`;
