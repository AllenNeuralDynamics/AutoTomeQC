# Usage Guide

## Getting Started

To get started, install the required dependencies using `uv`:

```bash
uv sync
```

## 1. Interactive Mode (CLI)

Use this mode to quickly test your saved section images. It launches the service and waits for you to paste file paths for analysis.

Run the service:
```bash
uv run autotomeqc
```

Once the service says `Ready >`, you can:
- Paste a file path: `example/input_images/img0.jpg`
- Exit: Type `exit`, `stop`, or press `Ctrl+C`.

## 2. Python Library

You can import `AutoTomeService` directly into your own Python code to integrate QC into your acquisition loops.

```python
from autotomeqc.core.autotome_service import AutoTomeService

# Initialize Service (Loads YOLO & Classification Models)
service = AutoTomeService()
service.start()

# Method A: Process by File Path
future_a = service.process(img_path="data/sample_01.jpg")
result = future_a.result()  # Wait for the result
print(f"QC Status: {result['qc_summary']}")

# Method B: Process by Raw Frame (e.g., from Camera)
future_b = service.process(frame=frame)
result = future_b.result()  # Wait for the result
print(f"QC Status: {result['qc_summary']}")

service.stop()
```

## Output Format

A full JSON report is also saved to disk.
- **Directory:** Defined in `src/autotomeqc/config/yolo-config.yaml` (`output_dir` setting).
- **Files:** `{filename}_qc.json`

**Example JSON Report:**
```json
{
    "filename": "img8",
    "timestamp": "2026-04-23 20:49:57",
    "qc_summary": "FAIL",
    "fail_reason": "Section failed QC criteria",
    "processing_time_sec": 0.5774,
    "sections": [
        {
            "qc_result": "FAIL",
            "segmentation_conf": 0.96,
            "area_in_pixels": 24504,
            "overlap_ratio": 1.0,
            "criteria": {
                "coverage": {
                    "pass_status": true,
                    "label": "full_section",
                    "conf": 0.9958
                },
                "knife_mark": {
                    "pass_status": false,
                    "label": "knifemark_shredding",
                    "conf": 0.9992,
                    "reason": "Defect Detected: knifemark_shredding"
                },
                "thickness_consistency": {
                    "pass_status": true,
                    "label": "Consistent",
                    "conf": 0.8967
                },
                "thickness": {
                    "pass_status": true,
                    "label": "60",
                    "conf": 0.5589
                },
                "shape": {
                    "pass_status": true,
                    "label": "Diamond",
                    "metric": 5,
                    "message": "Detected Diamond (vertices=5)"
                }
            }
        }
    ]
}
```

**Example Terminal Output:**
```text
Ready > example/input_images/img1.jpg
autotomeqc.interface.cli - INFO - Processing: img8
autotomeqc.interface.cli - INFO - Status:     FAIL
autotomeqc.interface.cli - INFO - Reason:     Section failed QC criteria
autotomeqc.interface.cli - INFO -  -> Section 0: FAIL | Area: 24504px
autotomeqc.interface.cli - INFO -     ✅ coverage: full_section
autotomeqc.interface.cli - INFO -     ❌ knife_mark: knifemark_shredding
autotomeqc.interface.cli - INFO -     ✅ thickness_consistency: Consistent
autotomeqc.interface.cli - INFO -     ✅ thickness: 60
autotomeqc.interface.cli - INFO -     ✅ shape: Diamond
```
