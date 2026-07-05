# Project: Medical Document Layout Annotator Refactoring

## Architecture
The application is a React + Vite TypeScript SPA that manages layout annotation on medical document page images. 

### Data Flow
1. **Global State**: Managed by Zustand (`src/store/annotationStore.ts`). Contains the document list, active pages, selected layer tab (`ADE` | `DL` | `NM` | `GT`), and active workspace detections (`workingDetections`).
2. **UI Layer**: React components subscribe directly to Zustand state slices.
3. **Canvas Drawing**: Done via HTML5 Canvas context in `AnnotationCanvas.tsx`. Coordinate conversions map mouse event space coordinates to image pixel coordinate spaces.

### Code Layout
- `src/components/AnnotationCanvas.tsx`: Core canvas drawing, pan/zoom handling, and drawing mode interaction.
- `src/store/annotationStore.ts`: Global state store.
- `src/pages/Annotate.tsx`: Layout workspace wrapper.
- `src/context/AnnotationContext.tsx`: Deprecated context layer (to be removed).

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Test Infrastructure Setup | Set up testing library (Vitest, React Testing Library, canvas-mock) to verify components | None | PLANNED |
| M2 | State Selection Refactor | Remove AnnotationContext.tsx, migrate components to use Zustand selectors directly | M1 | IN_PROGRESS (3de2474c-410b-474a-8905-a4e104747702) |
| M3 | Canvas Flickering Fix | Move zoom, offset, drag coords to refs. Resize canvas in useLayoutEffect/ResizeObserver without resetting buffer. Redraw only on events. Remove blocking render logs. | M2 | PLANNED |
| M4 | Scroll Zoom Calibration | Math.sign(e.deltaY) direction mapping, 12% fixed step, throttle scroll inputs (max 60fps), center on cursor | M3 | PLANNED |
| M5 | Zoom Toolbar & Page Reset | Add floating magnifier controls (Zoom In, Zoom Out, Reset 1:1), percentage display, reset zoom/pan on page change | M4 | PLANNED |
| M6 | Integration & Verification | Run all tests, perform coverage audit, and verify UX stability | M5 | PLANNED |

## Interface Contracts

### Zustand Store (`src/store/annotationStore.ts`)
- `useAnnotationStore`: React selector hook to access the store.
- `changePage(pageIndex: number)`: Updates page index and triggers default layout reset.

### AnnotationCanvas API (`src/components/AnnotationCanvas.tsx`)
- Props:
  - `detections: Detection[]` (the annotations to draw)
  - `imagePath: string` (the source document image to display)
- Internals:
  - Image load ref: `imageRef = useRef<HTMLImageElement | null>(null)`
  - Transient view states: `transformRef = useRef({ scale: 1, x: 0, y: 0 })`
  - Floating controls: bottom-right or top-right overlay.
