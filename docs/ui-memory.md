# UI Memory

## Preview Layout

- The main workflow is preview-first: keep left and right camera images as the dominant area.
- Keep the `Left Camera` and `Right Camera` title row compact. The preview grid in `stereo_aruco_gui/app/main_window.py` uses zero margins, small vertical spacing, and gives the image row the stretch.
- Do not reduce the bottom console/status log height to solve preview spacing. The console should stay readable.

## Camera Stats Display

- Per-camera detailed stats belong inside the corresponding preview image, in the lower-right overlay.
- The left preview overlay shows left camera details; the right preview overlay shows right camera details.
- Overlay content is:

```text
idx <camera index>
<fps> fps
frames <frame count>
age <seconds>s
```

- The bottom `Camera`/stats row should stay short and only show the overall FPS summary, for example:

```text
Camera: L 29.8 fps | R 28.2 fps
```

- In single-camera mode or when the right camera is not open, clear the right preview overlay instead of leaving old stats visible.

## Overlay Style

- The preview stats overlay is implemented in `stereo_aruco_gui/app/image_view.py` as `ImageView.overlay_label`.
- Current preferred overlay style:

```css
background: rgba(245, 245, 245, 120);
color: #24292f;
border: 0;
border-radius: 3px;
padding: 4px 6px;
```

- Keep the overlay semi-transparent and lightweight so it does not hide calibration targets or ArUco markers.
