# Configuration

AutoTomeQC relies on a YAML configuration file to define directories, post-processing rules, and paths to AI model weights. By default, it uses the configuration defined in `src/autotomeqc/config/yolo-config.yaml`.

## Example Configuration

```yaml
qc:
  output_dir: "example/output"
  save_segmented_images: true
  save_input_images: true
  yolo:
    weights_path: "weights/seg_fast_yolo26_640.pt"
    conf_thresh: 0.5
    img_size: 640
    img_dim:
    max_det: 10
  yolo_post_processing:
    out_dim:
    loop_bbox_margin: 30
    allow_no_loop: true
    overlap_threshold: 0.5
```

### Key Parameters

- **`output_dir`**: The directory where JSON QC reports and debug images are saved.
- **`save_segmented_images`**: If `true`, the pipeline saves the standardized BBox crop of the detected section to disk.
- **`allow_no_loop`**: If `true`, the pipeline will process sections normally even if the global context loop is missing.
- **`overlap_threshold`**: The minimum intersection-over-area required to consider a section "inside" the loop.
