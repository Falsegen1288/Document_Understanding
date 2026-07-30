"""
IBM Docling + TableFormer Table Extractor
------------------------------------------
Leverages IBM's Docling document parsing engine combined with TableFormer to reconstruct grids.
"""

import os
import pandas as pd

def extract_tables(pdf_path: str, pages: list[int] | None = None) -> list[dict]:
    """
    Extract table components using IBM Docling.
    
    Args:
        pdf_path: Path to the input PDF file.
        pages: Optional list of 1-indexed page indices.
        
    Returns:
        List of dicts representing parsed tables.
    """
    if not os.path.exists(pdf_path):
        return []
        
    try:
        from docling.document_converter import DocumentConverter
        import fitz
        
        # Check page count to manage memory and prevent std::bad_alloc
        doc_fitz = fitz.open(pdf_path)
        page_count = len(doc_fitz)
        
        # Check if the document has native digital text (to completely bypass OCR for a 10x speedup!)
        has_digital_text = False
        for page in doc_fitz:
            if page.get_text("text").strip():
                has_digital_text = True
                break
        doc_fitz.close()
        
        # Build optimized pipeline options
        pipeline_options = None
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True
            
            # Disable features not needed for table grid extraction to maximize speed
            pipeline_options.do_ocr = not has_digital_text
            pipeline_options.do_code_enrichment = False
            pipeline_options.do_formula_enrichment = False
            pipeline_options.do_picture_classification = False
            pipeline_options.do_picture_description = False
            
            try:
                from docling.datamodel.pipeline_options import TableFormerMode
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            except Exception:
                pass
        except Exception as pe:
            print(f"[WARNING] Docling advanced pipeline options failed to configure: {pe}")
            
        # Initialize converter with optimized options if successful
        if pipeline_options is not None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.document_converter import PdfFormatOption
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
            except Exception as ce:
                print(f"[WARNING] Docling PdfFormatOption failed: {ce}. Using default converter.")
                converter = DocumentConverter()
        else:
            converter = DocumentConverter()
            
        parsed_tables = []
        table_idx = 1
        
        # If specific pages are requested, process only those pages one-by-one to save massive time
        if pages is not None and len(pages) > 0:
            print(f"[INFO] Restricting Docling parser to specific page(s): {pages}")
            for p in pages:
                if p < 1 or p > page_count:
                    continue
                try:
                    page_range = (p, p)
                    result = converter.convert(pdf_path, page_range=page_range)
                    doc = result.document
                    for item in doc.tables:
                        prov = getattr(item, "prov", [])
                        page_no = prov[0].page_no if prov else p
                        
                        try:
                            markdown = item.export_to_markdown(doc=doc) if hasattr(item, "export_to_markdown") else (item.to_markdown() if hasattr(item, "to_markdown") else "")
                        except Exception:
                            markdown = item.export_to_markdown() if hasattr(item, "export_to_markdown") else (item.to_markdown() if hasattr(item, "to_markdown") else "")
                            
                        try:
                            df = item.export_to_dataframe(doc=doc) if hasattr(item, "export_to_dataframe") else (item.to_dataframe() if hasattr(item, "to_dataframe") else pd.DataFrame())
                        except Exception:
                            df = item.export_to_dataframe() if hasattr(item, "export_to_dataframe") else (item.to_dataframe() if hasattr(item, "to_dataframe") else pd.DataFrame())
                        
                        parsed_tables.append({
                            "table_index": table_idx,
                            "page": page_no,
                            "data": df.values.tolist(),
                            "dataframe": df,
                            "markdown": markdown,
                            "engine": "Docling+TableFormer"
                        })
                        table_idx += 1
                except Exception as page_err:
                    print(f"[ERROR] Docling parsing page {p} failed: {page_err}")
        else:
            # Fallback to general range conversion
            page_range = None
            if page_count > 5:
                print(f"[INFO] Large document detected ({page_count} pages). Limiting Docling table parsing to first 5 pages to manage system memory.")
                page_range = (1, 5)
                
            result = converter.convert(pdf_path, page_range=page_range)
            doc = result.document
            
            for item in doc.tables:
                prov = getattr(item, "prov", [])
                page_no = prov[0].page_no if prov else 1
                
                try:
                    markdown = item.export_to_markdown(doc=doc) if hasattr(item, "export_to_markdown") else (item.to_markdown() if hasattr(item, "to_markdown") else "")
                except Exception:
                    markdown = item.export_to_markdown() if hasattr(item, "export_to_markdown") else (item.to_markdown() if hasattr(item, "to_markdown") else "")
                    
                try:
                    df = item.export_to_dataframe(doc=doc) if hasattr(item, "export_to_dataframe") else (item.to_dataframe() if hasattr(item, "to_dataframe") else pd.DataFrame())
                except Exception:
                    df = item.export_to_dataframe() if hasattr(item, "export_to_dataframe") else (item.to_dataframe() if hasattr(item, "to_dataframe") else pd.DataFrame())
                
                parsed_tables.append({
                    "table_index": table_idx,
                    "page": page_no,
                    "data": df.values.tolist(),
                    "dataframe": df,
                    "markdown": markdown,
                    "engine": "Docling+TableFormer"
                })
                table_idx += 1
                
        return parsed_tables
    except ImportError:
        print("[ERROR] docling is not installed. Please install docling.")
        return []
    except Exception as e:
        print(f"[ERROR] Docling table extraction failed: {e}")
        return []
