# Opti-Face [PROVISIONAL]

Opti-Face, short for "Optimal Face," is a hybrid face recognition system.

## Installation

1. **Install UV Python Package Manager (Windows):**
    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
2. **Clone the repository:**
    ```bash
    git clone -b base https://github.com/Solomon-77/opti-face.git --depth=1 <your-project-name>
    ```
3. **Navigate to the project directory and run in terminal:**
    ```bash
    uv sync
    ```
4. **Add the models in `checkpoints` directory** | [**Download Models Here**](https://drive.google.com/drive/folders/1kYFrHNeyw3TNP0XMigu7uaTgLHG1BuhQ?usp=sharing)

5. **Add an image of your face inside the folder named after you in the `face_database` directory.**

6. **Train your dataset:**
    ```bash
    uv run prepare_embeddings.py
    ```
7. **Run the program:**
    ```bash
    uv run inference.py
    ```