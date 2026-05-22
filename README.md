# <img src="web/src/web/static/favicon.ico" width="30" height="30" alt="AutoTomeQC Icon"> AutoTomeQC Instrument Workspace

This repository is a monorepo workspace managed by [uv](https://github.com/astral-sh/uv). It contains the core quality control engine for tissue sectioning and its associated web-based orchestration interface.

## Workspace Overview

The workspace is structured into the following packages:

| Package | Directory | Description |
| :--- | :--- | :--- |
| `autotomeqc` | `/qc` | The core engine, CLI, and FastAPI server for sectioning QC. |
| `autotome-ui` | `/web` | Frontend UI built with `NiceGUI` for interacting with the instrument. |

## Quick Start

### Installation
Ensure `uv` is installed, then sync the entire workspace to create a unified virtual environment:

```bash
uv sync
```


## Running Components
You can execute the entry points for each package directly from the project root.

### Web UI (`autotome-ui`)
Launches the full application, including the backend API server and the NiceGUI frontend.

```bash
# Launch with default settings
uv run autotome-ui
```

**Optional Arguments:**
*   `--ui-port <port>`: Sets the port for the web interface (default: 8080).
*   `--backend-port <port>`: Sets the port for the backend API (default: 8000).
*   `--log-level <level>`: Sets the logging verbosity (e.g., `DEBUG`, `INFO`, `WARNING`).
*   `--web`: Forces the app to open in a web browser. In this mode, file uploads are limited to 1000 files at a time due to browser and websocket constraints.

**Example:**
```bash
uv run autotome-ui --ui-port 8081 --backend-port 8001 --log-level DEBUG
```

### QC Engine CLI (`autotomeqc`)
Starts the core QC engine in an interactive command-line mode, waiting for file paths to be provided for processing.

```bash
# Launch the interactive CLI
uv run autotomeqc
```

**Optional Arguments:**
*   `--config <path>`: Path to a custom `config.yaml` file.
*   `--log-level <level>`: Sets the logging verbosity.

**Example:**
```bash
uv run autotomeqc --log-level DEBUG --config /path/to/my_config.yaml
```

## Development
Testing
Run tests across the entire workspace or for individual members:

# Run all workspace tests
```bash
uv run pytest
```

Type Checking & Linting

# Linting
```bash
uv run ruff check .
```

# Static Type Checking
```bash
uv run mypy .
```