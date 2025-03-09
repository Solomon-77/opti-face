from tkinter import StringVar, Canvas, ttk, Toplevel
import cv2
from PIL import Image, ImageTk
import sv_ttk
from pages.embeddings_manager import FaceEmbeddingApp

class MainApp:
    def __init__(self, window, window_title, pipeline):
        self.window = window
        self.window.title(window_title)
        self.pipeline = pipeline
        self.setup_ui()
        self.window.resizable(False, False)

    def setup_ui(self):
        sv_ttk.set_theme("dark")
        self.side_panel = ttk.Frame(self.window, width=200)
        self.side_panel.pack(side="left", fill="y")

        self.title_label = ttk.Label(self.side_panel, text="Config", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)
        self.min_accuracy_label = ttk.Label(self.side_panel, text="Minimum Accuracy Threshold:")
        self.min_accuracy_label.pack(pady=5)
        self.min_accuracy_var = StringVar(value=str(self.pipeline.min_accuracy))
        self.min_accuracy_entry = ttk.Entry(self.side_panel, textvariable=self.min_accuracy_var)
        self.min_accuracy_entry.pack(pady=5)

        self.min_recognize_label = ttk.Label(self.side_panel, text="Minimum Recognize Threshold:")
        self.min_recognize_label.pack(pady=5)
        self.min_recognize_var = StringVar(value=str(self.pipeline.min_recognize))
        self.min_recognize_entry = ttk.Entry(self.side_panel, textvariable=self.min_recognize_var)
        self.min_recognize_entry.pack(pady=5)

        self.refresh_button = ttk.Button(self.side_panel, text="Refresh Config", command = lambda: (self.refresh_config()))
        self.refresh_button.pack(pady=10)

        self.manage_embeddings_button = ttk.Button(self.side_panel, text="Manage Embeddings", command=self.open_embedding_manager)
        self.manage_embeddings_button.pack(pady=10)
        
        self.canvas = Canvas(self.window, width=640, height=480)
        self.canvas.pack()

        self.cap = cv2.VideoCapture(0)
        self.update()

    def refresh_config(self):
        """Update the configuration values from the entry fields."""
        try:
            self.pipeline.min_accuracy = float(self.min_accuracy_var.get())
            self.pipeline.min_recognize = float(self.min_recognize_var.get())
            print(f"Config updated: Min Accuracy = {self.pipeline.min_accuracy}, Min Recognize = {self.pipeline.min_recognize}")
        except ValueError:
            print("Invalid input. Please enter numeric values.")

    def update(self):
        """Update the OpenCV display in the tkinter window."""
        ret, frame = self.cap.read()
        if ret:
            frame = self.pipeline.process_frame(frame)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.window.after(10, self.update)

    def cleanup(self):
        """Release resources."""
        self.pipeline.cleanup()
        self.cap.release()

    def open_embedding_manager(self):
        """Open the FaceEmbeddingApp and maintain layering."""
        if not hasattr(self, "embedding_window") or not self.embedding_window.winfo_exists():
            self.embedding_window = Toplevel(self.window)
            FaceEmbeddingApp(self.embedding_window, self.pipeline)
            self.window.lower(self.embedding_window)
            self.embedding_window.lift()
            self.window.bind("<FocusIn>", lambda e: self.lift_embedding_window())

    def lift_embedding_window(self):
        """Safely bring FaceEmbeddingApp to the top only if it exists."""
        if hasattr(self, "embedding_window") and self.embedding_window.winfo_exists():
            self.embedding_window.lift()