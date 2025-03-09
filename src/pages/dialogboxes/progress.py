import os
from tkinter import ttk, Toplevel

class LoadingDialog:
    def __init__(self, parent, total_frames):
        self.parent = parent
        self.total_frames = total_frames
        self.current_frame = 0
        self.current_embedding = 0

        # Create a new top-level window
        self.dialog = Toplevel(parent)
        self.dialog.title("Processing...")
        self.dialog.geometry("300x150")
        self.dialog.resizable(False, False)

        # Ensure this window is always on top of all others
        self.dialog.lift()
        self.dialog.attributes("-topmost", True)
        self.dialog.bind("<FocusIn>", lambda e: self.dialog.lift())
        self.dialog.bind("<Map>", lambda e: self.dialog.lift())

        # Prevent interaction with other windows
        self.dialog.grab_set()

        # Add a label for frame extraction progress
        self.frame_label = ttk.Label(self.dialog, text="Extracting frames: 0/0")
        self.frame_label.pack(pady=10)

        # Add a label for embedding extraction progress
        self.embedding_label = ttk.Label(self.dialog, text="Extracting embeddings: 0/0")
        self.embedding_label.pack(pady=10)

        # Add a progress bar
        self.progress = ttk.Progressbar(self.dialog, orient="horizontal", length=250, mode="determinate")
        self.progress.pack(pady=10)

    def update_frame_progress(self, current_frame):
        """Update the frame extraction progress."""
        self.current_frame = current_frame
        self.frame_label.config(text=f"Extracting frames: {self.current_frame}/{self.total_frames}")
        self.dialog.update_idletasks()

    def update_embedding_progress(self, current_embedding):
        """Update the embedding extraction progress."""
        self.current_embedding = current_embedding
        self.embedding_label.config(text=f"Extracting embeddings: {self.current_embedding}/{self.total_frames}")
        self.progress["value"] = (self.current_embedding / self.total_frames) * 100
        self.dialog.update_idletasks()

    def close(self):
        """Close the loading dialog."""
        self.dialog.grab_release()
        self.dialog.destroy()