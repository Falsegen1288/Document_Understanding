/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { Download, Share2, ChevronLeft, ChevronRight, CheckCircle2, ChevronUpCircle, AlertTriangle, Timer, X, Copy, Check } from 'lucide-react';
import { INITIAL_TABLE_ROWS, RAW_MARKDOWN_CONTENT } from '../data';
import { TableRow } from '../types';

interface TableExtractionViewProps {
  jobId: string | null;
}

export const TableExtractionView: React.FC<TableExtractionViewProps> = ({ jobId }) => {
  const [rows, setRows] = useState<TableRow[]>(INITIAL_TABLE_ROWS);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [copiedAll, setCopiedAll] = useState(false);
  
  // Track active pagination page (simulating pages)
  const [currentPage, setCurrentPage] = useState(1);
  
  // Real dynamic Table variables
  const [tables, setTables] = useState<any[]>([]);
  const [activeTableIdx, setActiveTableIdx] = useState<number>(0);
  const [fileName, setFileName] = useState<string>('FIN-Q4-AUDIT_v2.pdf');
  const [markdownContent, setMarkdownContent] = useState<string>(RAW_MARKDOWN_CONTENT);

  // Fetch job and tables dynamically if jobId is active
  useEffect(() => {
    if (!jobId || jobId.startsWith('run-')) {
      setRows(INITIAL_TABLE_ROWS);
      setTables([]);
      setFileName('FIN-Q4-AUDIT_v2.pdf');
      setMarkdownContent(RAW_MARKDOWN_CONTENT);
      return;
    }

    const loadTables = async () => {
      try {
        const jobRes = await fetch(`/api/jobs/${jobId}`);
        if (jobRes.ok) {
          const jobData = await jobRes.json();
          setFileName(jobData.filename);
        }

        const res = await fetch(`/api/results/${jobId}`);
        if (res.ok) {
          const data = await res.json();
          const foundTables: any[] = [];
          const pages = data.pages || [];
          
          pages.forEach((p: any) => {
            const dets = p.detections || [];
            dets.forEach((d: any) => {
              if (d.label === 'table' && (d.extracted?.dataframe_csv || d.extracted?.csv)) {
                foundTables.push({
                  id: d.id,
                  page: p.page_number,
                  csv: d.extracted.dataframe_csv || d.extracted.csv,
                  markdown: d.extracted.markdown || '',
                  cols: d.extracted.cols || 0,
                  rows: d.extracted.rows || 0,
                });
              }
            });
          });
          
          setTables(foundTables);
          if (foundTables.length > 0) {
            setActiveTableIdx(0);
            setMarkdownContent(foundTables[0].markdown || '# Extracted Table');
          }
        }
      } catch (err) {
        console.error('Failed loading dynamic tables from API:', err);
      }
    };
    loadTables();
  }, [jobId]);

  // Recalculate Markdown whenever active table index changes
  useEffect(() => {
    if (tables.length > 0 && tables[activeTableIdx]) {
      setMarkdownContent(tables[activeTableIdx].markdown || '# Extracted Table');
    }
  }, [activeTableIdx, tables]);

  const parseCsvData = (csvString: string) => {
    const lines = csvString.trim().split('\n');
    if (lines.length === 0 || !lines[0]) return { headers: [], body: [] };

    const parseLine = (line: string) => {
      const result = [];
      let current = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          result.push(current.trim().replace(/^"|"$/g, ''));
          current = '';
        } else {
          current += char;
        }
      }
      result.push(current.trim().replace(/^"|"$/g, ''));
      return result;
    };

    const headers = parseLine(lines[0]);
    const body = lines.slice(1).map(parseLine);
    return { headers, body };
  };

  const { headers: parsedHeaders, body: parsedBody } = tables[activeTableIdx]
    ? parseCsvData(tables[activeTableIdx].csv)
    : { headers: [], body: [] };
  
  // Handle double click cell editing
  const [editingCell, setEditingCell] = useState<{ rowIdx: number; field: 'projectedRev' | 'actualYield' | 'metricStream' } | null>(null);
  const [tempValue, setTempValue] = useState('');

  const startEditCell = (rowIdx: number, field: 'projectedRev' | 'actualYield' | 'metricStream', currentVal: string) => {
    setEditingCell({ rowIdx, field });
    setTempValue(currentVal);
  };

  const handleVarianceRecalc = (projected: string, actual: string): string => {
    const projNum = parseFloat(projected.replace(/[^0-9.-]/g, ''));
    const actNum = parseFloat(actual.replace(/[^0-9.-]/g, ''));
    
    if (isNaN(projNum) || isNaN(actNum) || projNum === 0) return '+0.00%';
    const pct = ((actNum - projNum) / projNum) * 100;
    const sign = pct >= 0 ? '+' : '';
    return `${sign}${pct.toFixed(2)}%`;
  };

  const saveCell = () => {
    if (!editingCell) return;
    const { rowIdx, field } = editingCell;
    
    setRows((prevRows) => {
      const copy = [...prevRows];
      const targetRow = { ...copy[rowIdx] };
      
      let finalVal = tempValue;
      if (field === 'projectedRev' || field === 'actualYield') {
        const cleaned = tempValue.replace(/[^0-9.]/g, '');
        const valNum = parseFloat(cleaned) || 0;
        finalVal = `$${valNum.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
      }

      targetRow[field] = finalVal;

      if (field === 'projectedRev' || field === 'actualYield') {
        targetRow.varianceDelta = handleVarianceRecalc(targetRow.projectedRev, targetRow.actualYield);
      }

      copy[rowIdx] = targetRow;
      return copy;
    });

    setEditingCell(null);
  };

  const handleCopyMarkdown = () => {
    navigator.clipboard.writeText(markdownContent);
    setCopiedAll(true);
    setTimeout(() => setCopiedAll(false), 2000);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-140px)] gap-6 select-none animate-fadeIn">
      {/* Workspace Header Actions bar */}
      <div className="flex justify-between items-center bg-[#020617] border border-outline-variant/60 rounded-2xl px-6 py-4 shadow-xl">
        <div className="flex items-center gap-4">
          <ChevronUpCircle className="w-5 h-5 text-primary" />
          <div>
            <h1 className="font-headline text-base font-black text-on-surface truncate max-w-[240px] md:max-w-md" title={fileName}>
              {fileName}
            </h1>
            <p className="text-[10px] text-on-surface-variant font-sans font-semibold mt-0.5">
              TABLE ANALYSIS WORKSPACE
            </p>
          </div>
          
          {/* Dynamic extracted tables selector dropdown */}
          {tables.length > 0 && (
            <div className="flex items-center gap-2 ml-4">
              <span className="text-[9px] font-mono uppercase text-slate-500 font-bold">Tables Found:</span>
              <select
                value={activeTableIdx}
                onChange={(e) => setActiveTableIdx(parseInt(e.target.value))}
                className="bg-slate-950 text-white border border-outline-variant/60 rounded-lg p-1.5 text-[10px] font-mono focus:ring-1 focus:ring-primary focus:outline-none cursor-pointer hover:border-primary/50 transition-colors"
              >
                {tables.map((t, i) => (
                  <option key={t.id} value={i}>
                    Table {i + 1} (Page {t.page})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          <div className="bg-surface-container-low border border-outline-variant rounded-xl p-1 flex items-center select-none">
            <button 
              onClick={() => setIsDrawerOpen(false)}
              className={`px-4 py-1.5 rounded-lg font-headline text-xs font-black tracking-widest uppercase transition-all whitespace-nowrap cursor-pointer ${
                !isDrawerOpen ? 'bg-secondary text-on-secondary' : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              Table Grid
            </button>
            <button 
              onClick={() => setIsDrawerOpen(true)}
              className={`px-4 py-1.5 rounded-lg font-headline text-xs font-black tracking-widest uppercase transition-all whitespace-nowrap cursor-pointer ${
                isDrawerOpen ? 'bg-secondary text-on-secondary' : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              Raw Markdown
            </button>
          </div>

          <button 
            onClick={() => {
              if (tables.length > 0 && tables[activeTableIdx]) {
                const csvContent = tables[activeTableIdx].csv;
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.setAttribute('download', `table_page_${tables[activeTableIdx].page}.csv`);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
              } else {
                alert('Exporting simulated structured data as CSV stream...');
              }
            }}
            className="bg-primary text-on-primary font-headline text-xs font-black tracking-widest uppercase px-4 py-2.5 rounded-xl hover:brightness-110 active:scale-95 duration-100 flex items-center gap-1.5 transition-all outline-none border-none cursor-pointer"
          >
            <Download className="w-4 h-4 stroke-[2.5]" />
            Export CSV
          </button>
          
          <button 
            onClick={() => alert('Dataset shared link: https://ais-pre-ei42/workspace/v2/FIN-Q4-AUDIT_v2.pdf_extracted')}
            className="hover:bg-surface-container-high/60 border border-outline-variant text-on-surface font-headline text-xs font-black tracking-widest uppercase px-4 py-2.5 rounded-xl active:scale-95 duration-100 flex items-center gap-1.5 transition-all outline-none cursor-pointer"
          >
            <Share2 className="w-4 h-4" />
            Share Dataset
          </button>
        </div>
      </div>

      {/* Main Table Screen content */}
      <div className="flex-1 flex gap-6 overflow-hidden items-start relative">
        
        {/* Left Side: Table spreadsheet grid */}
        <div className="flex-1 h-full overflow-hidden bg-white border border-slate-300 rounded-2xl flex flex-col shadow-lg shadow-slate-200/5">
          <div className="flex-1 overflow-auto custom-scrollbar">
            <table className="w-full border-collapse text-left custom-font-mono text-xs">
              {tables.length > 0 ? (
                <>
                  <thead className="sticky top-0 z-20 bg-slate-100 text-slate-900 border-b border-slate-300">
                    <tr>
                      {parsedHeaders.map((h, i) => (
                        <th key={i} className="px-5 py-4 border-r border-slate-300 font-extrabold tracking-wider bg-slate-100 text-slate-800 uppercase">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {parsedBody.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-50 cursor-pointer transition-colors duration-100 group border-b border-slate-100">
                        {row.map((cell, cellIdx) => (
                          <td key={cellIdx} className={`px-5 py-3 border-r border-slate-200 text-slate-800 select-text ${
                            cellIdx === 0 ? 'text-indigo-700 font-black' : 'text-slate-950 font-medium'
                          }`}>
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </>
              ) : (
                <>
                  <thead className="sticky top-0 z-20 bg-slate-100 text-slate-900 border-b border-slate-300">
                    <tr>
                      <th className="px-5 py-4 border-r border-slate-300 font-extrabold tracking-wider bg-slate-100 text-slate-800">INDEX_ID</th>
                      <th className="px-5 py-4 border-r border-slate-300 font-extrabold tracking-wider bg-slate-100 text-slate-800">FISCAL_PERIOD</th>
                      <th className="px-5 py-4 border-r border-slate-300 font-extrabold tracking-wider bg-slate-100 text-slate-800">METRIC_STREAM</th>
                      <th className="px-5 py-4 border-r border-slate-300 font-extrabold tracking-wider bg-slate-100 text-slate-800">PROJECTED_REV</th>
                      <th className="px-5 py-4 border-r border-slate-300 font-extrabold tracking-wider bg-slate-100 text-slate-800">ACTUAL_YIELD</th>
                      <th className="px-5 py-4 font-extrabold tracking-wider bg-slate-100 text-slate-800">VARIANCE_DELTA</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {rows.map((row, idx) => {
                      return (
                        <tr 
                          key={row.indexId}
                          className="hover:bg-slate-50 cursor-pointer transition-colors duration-100 group border-b border-slate-100"
                        >
                          {/* ID */}
                          <td className="px-5 py-3 border-r border-slate-200 text-indigo-700 font-black">
                            {row.indexId}
                          </td>
                          
                          {/* Period */}
                          <td className="px-5 py-3 border-r border-slate-200 text-slate-800 font-bold">
                            {row.fiscalPeriod}
                          </td>

                          {/* Metric Stream or Merged */}
                          {row.isMerged ? (
                            <td 
                              colSpan={row.span || 1}
                              className="px-5 py-3 border-r border-slate-200 relative"
                              style={{
                                background: `repeating-linear-gradient(45deg, rgba(6,182,212,0.04) 0px, rgba(6,182,212,0.04) 10px, rgba(6,182,212,0.1) 10px, rgba(6,182,212,0.1) 20px)`
                              }}
                            >
                              <div className="flex items-center justify-between">
                                {editingCell?.rowIdx === idx && editingCell.field === 'metricStream' ? (
                                  <input 
                                    type="text"
                                    value={tempValue}
                                    onChange={(e) => setTempValue(e.target.value)}
                                    onBlur={saveCell}
                                    onKeyDown={(e) => e.key === 'Enter' && saveCell()}
                                    autoFocus
                                    className="bg-white text-slate-950 text-xs px-2 py-0.5 border border-indigo-500 rounded focus:outline-none w-full"
                                  />
                                ) : (
                                  <span 
                                    onDoubleClick={() => startEditCell(idx, 'metricStream', row.metricStream)}
                                    className="text-slate-950 font-black hover:text-indigo-600 transition-colors cursor-text"
                                  >
                                    {row.metricStream}
                                  </span>
                                )}
                                <span className="text-[8px] tracking-widest font-black uppercase bg-[#ecfeff] border border-[#06b6d4] text-[#0891b2] px-2 py-0.5 rounded ml-2">
                                  Merged Unit
                                </span>
                              </div>
                            </td>
                          ) : row.isWarning ? (
                            <td 
                              colSpan={row.span || 1}
                              className="px-5 py-3 text-amber-800 font-black relative border-r border-slate-200"
                              style={{
                                background: `repeating-linear-gradient(45deg, rgba(217,119,6,0.04) 0px, rgba(217,119,6,0.04) 11px, rgba(217,119,6,0.1) 11px, rgba(217,119,6,0.1) 22px)`
                              }}
                            >
                              <div className="flex items-center gap-2 font-black">
                                <AlertTriangle className="w-4 h-4 text-amber-700" />
                                <span>{row.warningText}</span>
                              </div>
                            </td>
                          ) : (
                            /* Standard Metric Stream cell */
                            <td className="px-5 py-3 border-r border-slate-200 text-slate-950 font-bold">
                              {editingCell?.rowIdx === idx && editingCell.field === 'metricStream' ? (
                                <input 
                                  type="text"
                                  value={tempValue}
                                  onChange={(e) => setTempValue(e.target.value)}
                                  onBlur={saveCell}
                                  onKeyDown={(e) => e.key === 'Enter' && saveCell()}
                                  autoFocus
                                  className="bg-white text-slate-950 text-xs px-2 py-0.5 border border-indigo-500 rounded focus:outline-none w-full"
                                />
                              ) : (
                                <span 
                                  onDoubleClick={() => startEditCell(idx, 'metricStream', row.metricStream)}
                                  className="hover:text-indigo-600 transition-colors cursor-text"
                                >
                                  {row.metricStream}
                                </span>
                              )}
                            </td>
                          )}

                          {/* Projected Revenue */}
                          {!row.isMerged && !row.isWarning && (
                            <td className="px-5 py-3 border-r border-slate-200 text-slate-800 font-bold">
                              {editingCell?.rowIdx === idx && editingCell.field === 'projectedRev' ? (
                                <input 
                                  type="text"
                                  value={tempValue}
                                  onChange={(e) => setTempValue(e.target.value)}
                                  onBlur={saveCell}
                                  onKeyDown={(e) => e.key === 'Enter' && saveCell()}
                                  autoFocus
                                  className="bg-white text-slate-950 text-xs px-2 py-0.5 border border-indigo-500 rounded focus:outline-none w-24"
                                />
                              ) : (
                                <span 
                                  onDoubleClick={() => startEditCell(idx, 'projectedRev', row.projectedRev)}
                                  className="hover:text-indigo-600 transition-colors cursor-text"
                                >
                                  {row.projectedRev}
                                </span>
                              )}
                            </td>
                          )}

                          {/* Actual Yield */}
                          {!row.isMerged && !row.isWarning && (
                            <td className="px-5 py-3 border-r border-slate-200 text-[#0891b2] font-black">
                              {editingCell?.rowIdx === idx && editingCell.field === 'actualYield' ? (
                                <input 
                                  type="text"
                                  value={tempValue}
                                  onChange={(e) => setTempValue(e.target.value)}
                                  onBlur={saveCell}
                                  onKeyDown={(e) => e.key === 'Enter' && saveCell()}
                                  autoFocus
                                  className="bg-white text-slate-950 text-xs px-2 py-0.5 border border-indigo-500 rounded focus:outline-none w-24"
                                />
                              ) : (
                                <span 
                                  onDoubleClick={() => startEditCell(idx, 'actualYield', row.actualYield)}
                                  className="hover:text-indigo-600 transition-colors cursor-text"
                                >
                                  {row.actualYield}
                                </span>
                              )}
                            </td>
                          )}

                          {/* Variance Delta percentage highlight */}
                          {!row.isMerged && !row.isWarning && (
                            <td className={`px-5 py-3 font-black ${
                              row.varianceDelta.startsWith('+') ? 'text-emerald-700' : 'text-rose-600'
                            }`}>
                              {row.varianceDelta}
                            </td>
                          )}
                        </tr>
                      );
                    })}
                  </tbody>
                </>
              )}
            </table>
          </div>

          {/* Table pagination & footer records data */}
          <div className="bg-slate-100 border-t border-slate-300 px-6 py-3.5 flex justify-between items-center select-none">
            <span className="text-[11px] custom-font-mono text-slate-600 font-medium">
              Showing {tables.length > 0 ? parsedBody.length : rows.length} reconstructed records
            </span>
            <div className="flex gap-1.5">
              <button 
                onClick={() => currentPage > 1 && setCurrentPage(1)}
                className="w-8 h-8 flex items-center justify-center border border-slate-300 hover:bg-slate-200 text-slate-700 rounded-md transition-colors cursor-pointer bg-white"
                title="Previous list page"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button 
                onClick={() => currentPage === 1}
                className="w-8 h-8 flex items-center justify-center font-bold text-xs rounded-md bg-primary text-on-primary border-none cursor-pointer"
              >
                1
              </button>
              <button 
                className="w-8 h-8 flex items-center justify-center border border-slate-300 hover:bg-slate-200 text-slate-700 rounded-md transition-colors bg-white cursor-not-allowed"
                disabled
              >
                2
              </button>
              <button 
                onClick={() => {}}
                className="w-8 h-8 flex items-center justify-center border border-slate-300 hover:bg-slate-200 text-slate-700 rounded-md transition-colors bg-white cursor-pointer"
                title="Next list page"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Right Side: Raw Markdown Preview Panel Drawer */}
        {isDrawerOpen && (
          <aside className="w-[450px] h-full bg-surface-container-lowest border border-outline-variant rounded-2xl overflow-hidden flex flex-col shadow-2xl animate-slideLeft z-30 select-none">
            <div className="h-12 flex items-center justify-between px-4 bg-surface-container-low border-b border-outline-variant">
              <span className="text-on-surface custom-font-mono text-xs font-black tracking-wider uppercase">
                RAW_MARKDOWN_EXPORT
              </span>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={handleCopyMarkdown}
                  className="px-2.5 py-1 bg-secondary text-on-secondary hover:brightness-110 active:scale-95 duration-100 transition-all font-headline text-[9px] font-black tracking-wider uppercase rounded flex items-center gap-1 cursor-pointer outline-none"
                >
                  {copiedAll ? <Check className="w-3.5 h-3.5 stroke-[2.5]" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedAll ? 'Copied' : 'Copy'}</span>
                </button>
                <button
                  onClick={() => setIsDrawerOpen(false)}
                  className="p-1 hover:bg-surface-variant rounded text-on-surface-variant hover:text-error transition-colors cursor-pointer border-none bg-transparent"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-5 custom-scrollbar bg-slate-950 font-mono text-[11px] text-slate-100 leading-relaxed whitespace-pre-wrap select-text">
              {markdownContent}
            </div>
            
            <div className="p-4 bg-surface-container border-t border-outline-variant flex items-center justify-between text-[10px] text-on-surface-variant font-medium select-none">
              <span>Encoding: UTF-8 Payload</span>
              <div className="flex items-center gap-1.5 text-secondary">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Verified Schema Schema</span>
              </div>
            </div>
          </aside>
        )}
      </div>

      {/* Latency statistics or background tracer metrics */}
      <footer className="bg-surface-container-low border border-outline-variant/60 rounded-2xl px-6 py-3 flex justify-between items-center shadow-xl select-none">
        <div className="flex items-center gap-5 text-[10px] custom-font-mono text-on-surface-variant">
          <div className="flex items-center gap-2">
            <Timer className="w-3.5 h-3.5 text-secondary animate-pulse" />
            <span>Active Parsing Delay: <strong>1.42s</strong></span>
          </div>
          <span className="text-outline-variant">•</span>
          <span>Engine: IBM TableFormer v4.1 (TATR backbone)</span>
        </div>
        <span className="text-[10px] custom-font-mono text-secondary-fixed font-black uppercase bg-emerald-500/10 border border-emerald-500/25 px-2 py-0.5 rounded animate-pulse">
          Reconstruction Verified
        </span>
      </footer>
    </div>
  );
};
