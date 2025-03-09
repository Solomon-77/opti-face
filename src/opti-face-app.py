import tkinter as tk
from pipelines.face_recognition_pipeline import FaceRecognitionPipeline
from pages.start import MainApp

if __name__ == "__main__":
    pipeline = FaceRecognitionPipeline()
    root = tk.Tk()
    app = MainApp(root, "Face Recognition App", pipeline)
    
    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()