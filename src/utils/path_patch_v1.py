import os
import sys

def get_resource_path(relative_path):
    """ Get the correct file path for development and .exe mode """
    
    if getattr(sys, 'frozen', False):  # Running as a bundled EXE
        base_path = os.path.join(os.getenv("APPDATA"), "opti-face")
    else:
        base_path = os.path.abspath(".")  # Running as a script

    full_path = os.path.join(base_path, relative_path)

    # Ensure the directory exists
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)

    return full_path
