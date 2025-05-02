import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # Adjust base path calculation if your entry script `app.py` is not at the project root
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_writable_path(relative_path):
    """ Get path for writable data, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        # Running as a bundled app (executable)
        # Place writable data next to the executable
        app_path = os.path.dirname(sys.executable)
    else:
        # Running as a script (dev mode)
        app_path = os.path.abspath(".") # Assumes script is run from project root

    writable_dir = os.path.join(app_path, relative_path)
    try:
        os.makedirs(writable_dir, exist_ok=True) # Ensure the directory exists
    except OSError as e:
        print(f"Error creating writable directory {writable_dir}: {e}")
        # Fallback or raise error depending on requirements
        # For simplicity, we'll let it proceed, but logging/db might fail
        pass
    return writable_dir

# Original import remains
from src.gui import main