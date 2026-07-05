# Original User Request

## Initial Request — 2026-06-10T13:28:28+05:30

Refactor the React annotation canvas to resolve image flickering/vibration, fix hypersensitive scroll-wheel zoom, and add explicit zoom magnifier controls.

Working directory: `c:\Users\user\Downloads\medical-document-layout-annotator`
Integrity mode: development

## Requirements

### R1. Eliminate Canvas Flickering and Vibration
- Separate image loading from canvas drawing using a dedicated `useEffect` (or `onload` listener) with a `useRef` for the loaded image.
- Move transient view states (zoom, pan offset, drag coordinates) to `useRef` objects to prevent React state update feedback loops.
- Ensure event listeners (`mousemove`, `mousedown`, `mouseup`, `wheel`) are cleaned up properly on unmount.
- Remove any continuous `requestAnimationFrame` render loops and redraw the canvas only on events.

### R2. Calibrate Scroll-Wheel Zoom Sensitivity
- Use `Math.sign(e.deltaY)` to determine zoom direction (+1/-1) and apply a fixed, controlled zoom step (e.g. 12% per notch).
- Implement throttling/debouncing to restrict high-frequency scroll inputs (e.g. max 60fps).
- Center the zoom operation on the current cursor position.

### R3. Add Floating Zoom Controls Toolbar
- Add a floating toolbar at the bottom-right or top-right of the canvas container.
- Include explicit magnifier buttons: Zoom In (🔍+), Zoom Out (🔍-), and Reset (1:1).
- Include a percentage display showing the current zoom level (e.g., "120%").
- Selecting the Reset button must return the zoom factor to 1.0 and reset the pan offsets to (0, 0).
- Switching pages must clear pan/zoom coordinates back to default.

## Acceptance Criteria

### UI and Performance
- [ ] Canvas rendering is stable (no flashing, flickering, or vibration when loading pages or panning/zooming).
- [ ] Bounding boxes remain perfectly aligned with the document image content at all zoom levels.
- [ ] Zoom buttons modify the view by a standard factor (e.g. 15%) and the percentage display matches.
- [ ] The Reset button successfully resets coordinates to zoom=1.0 and offset=(0, 0).
- [ ] Scroll zoom centers on the cursor position without hypersensitivity or scaling drift.
