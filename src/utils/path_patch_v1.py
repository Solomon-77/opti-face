import os  # Detailed: Import the os module to handle file system operations such as path joining and directory creation.
         # Simple: Lets us work with files and folders.
import sys  # Detailed: Import the sys module to access system-specific parameters and functions, which helps determine the running context.
         # Simple: Gives us information about the system.

def get_resource_path(relative_path):
    """ Get the correct file path for development and .exe mode """
    # Detailed: This function returns a complete file path by adapting to the runtime context.
    # It distinguishes between running as a bundled executable (e.g., created with PyInstaller) and running as a script.
    # Simple: It gives you the right file path whether you're running a script or a packaged app.
    
    if getattr(sys, 'frozen', False):  # Detailed: Check if the attribute 'frozen' is set in sys, which indicates that the program is running as a bundled EXE.
                                      # Simple: If the program is bundled into an EXE...
        base_path = os.path.join(os.getenv("APPDATA"), "opti-face")  # Detailed: Use the APPDATA directory (a common location for application data on Windows) appended with "opti-face" as the base path.
                                                                     # Simple: Use a folder named "opti-face" inside APPDATA.
    else:
        base_path = os.path.abspath(".")  # Detailed: If not bundled, use the current directory (where the script is run) as the base path.
                                          # Simple: Use the current folder.
    
    full_path = os.path.join(base_path, relative_path)  # Detailed: Combine the base path with the given relative path to create the full resource path.
                                                       # Simple: Join the folder with your file path.
    
    # Ensure the directory exists
    if not os.path.exists(full_path):  # Detailed: Check if the computed full path does not already exist.
                                      # Simple: If the folder isn’t there...
        os.makedirs(full_path, exist_ok=True)  # Detailed: Create the directory (and any necessary parent directories) so that the full path is available.
                                               # Simple: Make the folder if it doesn't exist.
    
    return full_path  # Detailed: Return the fully constructed path, ensuring that the necessary directory structure is present.
                      # Simple: Give back the complete file path.
