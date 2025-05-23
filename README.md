# Opti-Face [PROVISIONAL]

Ongoing...

## Installation

1. **Install UV Python Package Manager (Windows):**
    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
2. **Clone the repository:**
    ```bash
    git clone https://github.com/Solomon-77/opti-face.git --depth=1
    ```
3. **Navigate to the project directory and run in terminal:**
    ```bash
    uv sync
    ```
4. **Add the models in `src/backend/checkpoints` directory** | [**Download Models Here**](https://drive.google.com/drive/folders/1kYFrHNeyw3TNP0XMigu7uaTgLHG1BuhQ?usp=sharing)

5. **Run the program:**
    ```bash
    uv run app.py
    ```

## Building with PyInstaller

1. **Install PyInstaller:**
    ```
    uv add pyinstaller
    ```

2. **Navigate to project directory and enter the command:**
    ```
    .venv/Scripts/activate
    ```

3. **Run this command:**
    ```
    pyinstaller app.spec
    ```
    or
    ```
    pyinstaller --noconfirm --onedir --windowed --add-data "src/backend/checkpoints;src/backend/checkpoints" --add-data "src/gui/icons;src/gui/icons" --collect-data mediapipe app.py
    ```