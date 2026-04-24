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

**Example Terminal Output:**
```text
Ready > example/input_images/img1.jpg
Processing: img1
Status:     FAIL
Reason:     Section failed QC criteria
 -> Section 0: FAIL | Area: 61464px
    ✅ coverage: full_section
    ❌ knife_mark: knifemark_shredding
    ✅ thickness_consistency: Consistent
    ✅ thickness: 80
    ✅ shape: Hexagon (vertices: 6)
```
