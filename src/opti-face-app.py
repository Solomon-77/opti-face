import tkinter as tk  # Detailed: Import the tkinter module to create graphical user interfaces (GUIs).  
                     # Simple: Lets us create windows and buttons.
from pipelines.face_recognition_pipeline import FaceRecognitionPipeline  # Detailed: Import the FaceRecognitionPipeline class which handles face recognition processing.  
                                                                           # Simple: Get the system that recognizes faces.
from pages.start import MainApp  # Detailed: Import the MainApp class which sets up the main GUI of the application.  
                                 # Simple: Get the main window for the app.

if __name__ == "__main__":  # Detailed: Check if this script is being run directly rather than imported as a module.  
                           # Simple: Run the code below only if this file is the main program.
    pipeline = FaceRecognitionPipeline()  # Detailed: Create an instance of the FaceRecognitionPipeline to manage face recognition tasks.  
                                           # Simple: Start the face recognition system.
    root = tk.Tk()  # Detailed: Initialize the main tkinter window (the root window).  
                    # Simple: Create the main window.
    app = MainApp(root, "Face Recognition App", pipeline)  # Detailed: Instantiate the MainApp with the root window, set the window title to "Face Recognition App", and pass the face recognition pipeline to it.  
                                                           # Simple: Build the app window with a title and the face recognition system.

    def on_closing():  # Detailed: Define a function to handle the event when the user attempts to close the window.  
                       # Simple: Create a function that runs when the app is closed.
        app.cleanup()  # Detailed: Call the cleanup method of the app to release resources and stop any background processes.  
                       # Simple: Stop the app and free resources.
        root.destroy()  # Detailed: Destroy the main tkinter window to exit the application completely.  
                        # Simple: Close the window.

    root.protocol("WM_DELETE_WINDOW", on_closing)  # Detailed: Set the protocol for the window manager to call the on_closing function when the window is closed (i.e., when the 'X' button is clicked).  
                                                   # Simple: Tell the window to run on_closing() when it is closed.
    root.mainloop()  # Detailed: Enter the tkinter main event loop which listens for events and updates the GUI until the window is closed.  
                     # Simple: Keep the window open and responsive.
