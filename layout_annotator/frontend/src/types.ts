/**
 * DocLayNet defined classes
 */
export type DocLayClass =
  | 'title'
  | 'section_header'
  | 'text'
  | 'list_item'
  | 'table'
  | 'picture'
  | 'caption'
  | 'footnote'
  | 'formula'
  | 'page_header'
  | 'page_footer';

export interface Detection {
  type: DocLayClass;
  bbox: [number, number, number, number]; // [x0, y0, x1, y1] original pixel space (e.g. 1275x1650)
  model: string; // 'DocLayoutYOLO', 'Nemotron-Parse-v1.1', 'ADE-DPT2', or 'human'
  ade_raw_type?: string;
  id?: string; // unique identifier
}

export interface ModelData {
  name: string;
  detections: Detection[];
}

export interface PageData {
  page: number;
  displayName: string;
  image_path: string;
  image_size: [number, number]; // [width, height] e.g. [1275, 1650]
  model_a: ModelData; // DocLayoutYOLO (DL)
  model_b: ModelData; // Nemotron-Parse-v1.1 (NM)
  model_c: ModelData; // ADE-DPT2 (ADE)
  ground_truth: Detection[] | null;
}

export const CLASS_COLORS: Record<DocLayClass, string> = {
  title: '#e63946',
  section_header: '#f4623a',
  text: '#4a7fa5',
  list_item: '#2a9d8f',
  table: '#e9c46a',
  picture: '#f4a261',
  caption: '#8ecae6',
  footnote: '#a8dadc',
  formula: '#6d6875',
  page_header: '#b5838d',
  page_footer: '#e5989b',
};

export const CLASS_LABELS: Record<DocLayClass, string> = {
  title: 'Title',
  section_header: 'Section-header',
  text: 'Text',
  list_item: 'List-item',
  table: 'Table',
  picture: 'Picture',
  caption: 'Caption',
  footnote: 'Footnote',
  formula: 'Formula',
  page_header: 'Page-header',
  page_footer: 'Page-footer',
};
