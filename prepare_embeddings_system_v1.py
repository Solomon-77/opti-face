import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import cv2
from utils.face_utils import preprocess_image, load_face_recognition_model

class FaceEmbeddingApp:
    def __init__(self, root, pipeline):
        self.root = root
        self.pipeline = pipeline
        self.root.title("Face Embeddings Manager")
        self.face_database_dir = './face_database/'
        self.model, self.device = load_face_recognition_model()

        # Ensure this window is always above the config window``
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.bind("<FocusIn>", lambda e: self.root.lift())
        self.root.bind("<Map>", lambda e: self.root.lift())

        # Create a frame for the file manager view
        self.file_manager_frame = ttk.Frame(root)
        self.file_manager_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview to show .npz files
        self.tree = ttk.Treeview(self.file_manager_frame, columns=("Name"), show="headings")
        self.tree.heading("Name", text="Name")
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Right-click menu for deleting files
        self.right_click_menu = tk.Menu(self.root, tearoff=0)
        self.right_click_menu.add_command(label="Delete", command=self.delete_selected_file)

        # Bind right-click event
        self.tree.bind("<Button-3>", self.show_right_click_menu)

        # Add new person button
        self.add_person_button = ttk.Button(root, text="Add New Person", command=self.open_add_person_window)
        self.add_person_button.pack(pady=10)

        # Load existing .npz files
        self.load_npz_files()

    def load_npz_files(self):
        """Load and display existing .npz files in the directory."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        for file_name in os.listdir(self.face_database_dir):
            if file_name.endswith(".npz"):
                self.tree.insert("", tk.END, values=(file_name,))

    def show_right_click_menu(self, event):
        """Show the right-click menu for deleting files."""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.right_click_menu.post(event.x_root, event.y_root)

    def delete_selected_file(self):
        """Delete the selected .npz file."""
        selected_item = self.tree.selection()
        if selected_item:
            file_name = self.tree.item(selected_item, "values")[0]
            file_path = os.path.join(self.face_database_dir, file_name)
            os.remove(file_path)
            self.load_npz_files()
            messagebox.showinfo("Success", f"Deleted {file_name}")

            # Notify the pipeline to reload embeddings
            self.pipeline.load_embeddings()
            self.pipeline.update_recognized_faces()

    def open_add_person_window(self):
        """Open a new window to add a new person."""
        self.add_person_window = tk.Toplevel(self.root)
        self.add_person_window.title("Add New Person")

        # Ensure this window is always above the "Manage Embeddings" window
        self.add_person_window.lift()
        self.add_person_window.attributes("-topmost", True)
        self.add_person_window.bind("<FocusIn>", lambda e: self.add_person_window.lift())
        self.add_person_window.bind("<Map>", lambda e: self.add_person_window.lift())

        # Label and entry for person's name
        ttk.Label(self.add_person_window, text="Person's Name:").grid(row=0, column=0, padx=10, pady=10)
        self.name_entry = ttk.Entry(self.add_person_window)
        self.name_entry.grid(row=0, column=1, padx=10, pady=10)

        # Button to add video
        self.video_path = None
        ttk.Button(self.add_person_window, text="Add Video", command=self.select_video).grid(row=1, column=0, columnspan=2, pady=10)

        # train button
        ttk.Button(self.add_person_window, text="Train", command=self.train_new_person).grid(row=2, column=0, columnspan=2, pady=10)
    def select_video(self):
        """Select a video file."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if self.video_path:
            messagebox.showinfo("Success", f"Selected video: {os.path.basename(self.video_path)}")

    def train_new_person(self):
        """train the system with the new person's details and start processing the video."""
        person_name = self.name_entry.get()
        if not person_name:
            messagebox.showerror("Error", "Please enter the person's name.")
            return
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video.")
            return

        # Start processing in a separate thread
        threading.Thread(target=self.process_video, args=(person_name, self.video_path)).start()
        self.add_person_window.destroy()

    def process_video(self, person_name, video_path):
        """Process the video to extract frames and create embeddings."""
        person_folder = os.path.join(self.face_database_dir, person_name)
        os.makedirs(person_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Show the loading dialog
        loading_dialog = LoadingDialog(self.root, total_frames)

        # Extract frames from the video
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(person_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

            # Update frame extraction progress
            loading_dialog.update_frame_progress(frame_count)

        cap.release()

        # Create embeddings from the frames
        person_embeddings = []
        embedding_count = 0
        for image_name in os.listdir(person_folder):
            face_tensor = preprocess_image(os.path.join(person_folder, image_name))
            if face_tensor is not None:
                embedding = self.model(face_tensor.to(self.device)).detach().cpu().numpy()
                person_embeddings.append(embedding)
                embedding_count += 1

                # Update embedding extraction progress
                loading_dialog.update_embedding_progress(embedding_count)

        # Save embeddings as .npz file
        if person_embeddings:
            npz_path = os.path.join(self.face_database_dir, f"{person_name}.npz")
            np.savez(npz_path, embeddings=np.array(person_embeddings))
            messagebox.showinfo("Success", f"Saved embeddings for {person_name} to {npz_path}")

        # Clean up
        for image_name in os.listdir(person_folder):
            os.remove(os.path.join(person_folder, image_name))
        os.rmdir(person_folder)

        # Close the loading dialog
        loading_dialog.close()

        # Reload .npz files in the main GUI
        self.root.after(0, self.load_npz_files)

        # Reload embeddings in the pipeline
        self.pipeline.load_embeddings()

class LoadingDialog:
    def __init__(self, parent, total_frames):
        self.parent = parent
        self.total_frames = total_frames
        self.current_frame = 0
        self.current_embedding = 0

        # Create a new top-level window
        self.dialog = tk.Toplevel(parent)
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