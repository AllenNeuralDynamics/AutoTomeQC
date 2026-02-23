# AutoTomeQC
Sectioning Quality Control
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

##  Getting Started
- [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
```bash
uv sync
uv run python -m autotomeqc
```

## Usage 1. Interactive Mode (CLI)

Use this mode to quickly test your saved section images. It launches the service and waits for you to paste file paths for analysis.

Run the service:
```bash
uv run python -m autotomeqc
```

Once the service says `Ready >`, you can:

Paste a file path: `example/input_images/img0.jpg`

Exit: Type `exit`, `stop`, or press `Ctrl+C`.


## Usage 2. Python Library 

You can import `AutoTomeService` directly into your own Python code to integrate QC into your acquisition loops.

See the example codes  `example/example_import.py`

```bash
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
result = future_a.result()  # Wait for the result
print(f"QC Status: {result['qc_summary']}")

service.stop()
```

## Output

A full JSON report are also saved to disk.

Directory: The location is defined in `src/autotomeqc/config/yolo-config.yaml` (see the output_dir setting).

Files: {filename}_qc.json


And, you can view the results directly in the terminal:

**Example Output:**
```bash
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

----------------------------------------
Ready > example/input_images/img2.jpg
Processing: img2
Status:     PASS
Reason:     N/A
 -> Section 0: PASS | Area: 58210px
    ✅ coverage: full_section
    ✅ knife_mark: none
    ✅ thickness_consistency: Consistent
    ✅ thickness: 70
    ✅ shape: Hexagon (vertices: 6)
```


## Tools

### Package/Project Management 

This project utilizes [uv](https://docs.astral.sh/uv/) to handle installing dependencies as well as setting up environments for this project. It replaces tool like pip, poetry, virtualenv, and conda. 

This project also uses [tox](https://tox.wiki/en/latest/index.html) for orchestrating multiple testing environments that mimics the github actions CI/CD so that you can test the workflows locally on your machine before pushing changes. 

### Code Quality Check

The following are tools used to ensure code quality in this project. 

- Unit Testing

```bash
uv run pytest tests
```

- Linting

```bash
uv run ruff check
```

- Type Check

```bash
uv run mypy src
```

## Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o docs/source/ src
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html docs/source/ docs/build/html
```
