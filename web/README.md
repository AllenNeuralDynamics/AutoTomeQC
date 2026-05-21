# AutoTomeQC - Web UI

This is the web interface for the AutoTomeQC pipeline, providing a rich, interactive experience for section quality control. It is built using the [NiceGUI](https://nicegui.io/) framework.

## Features

*   **Upload Queue**: Drag-and-drop or select multiple images to create a processing queue.
*   **Batch Processing**: Run the AutoTomeQC analysis on all pending images with a single click.
*   **Interactive Viewer**: Click on any image in the queue to view it, along with its detailed QC results and segmentation masks.
*   **Configuration**: View the backend system configuration directly in the UI.
*   **Export**: Download all processed images and their corresponding JSON result files as a single ZIP archive.

## Getting Started

This web application is designed to be run as part of the `AutoTomeQC` project.

1.  **Install Dependencies**

    Ensure you are in the project's root directory and synchronize the environment using `uv`:
    ```bash
    uv sync
    ```

2.  **Run the Application**

    The `autotome-ui` command launches both the backend server and the web UI.
    ```bash
    uv run autotome-ui
    ```
    You can then access the UI in your browser at the address provided in the terminal (e.g., `http://localhost:8080`).

    To specify different ports or change the logging verbosity, you can use the following arguments:
    ```bash
    # Example: Run on different ports with debug logging
    uv run autotome-ui --backend-port 8001 --ui-port 8081 --log-level DEBUG
    ```

## Development

This project uses `uv` for dependency management and a suite of tools for ensuring code quality.

### Running Tests

The web application has a comprehensive test suite covering backend controllers and UI components using `pytest` and `NiceGUI Testing`. To run all tests for the web module:

```bash
uv run pytest tests
```

### Code Quality

We use `ruff` for linting and `mypy` for static type checking, consistent with the `qc` module.

*   **Linting:**
    ```bash
    uv run ruff check src
    ```

*   **Type Checking:**
    ```bash
    uv run mypy src
    ```