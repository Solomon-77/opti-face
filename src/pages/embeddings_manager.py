import os
import numpy as np
from tkinter import ttk, Toplevel, filedialog, messagebox, Menu, Frame, Label, Entry, Button
import threading
from pipelines.training_pipeline import TrainingPipeline
from pages.dialogboxes.progress import LoadingDialog
from pages.records import RecordsWindow
import cv2
import shutil
import glob

class FaceEmbeddingApp:
    def __init__(self, root, pipeline):
        self.root = root
        self.pipeline = pipeline
        self.current_page = 1
        self.rows_per_page = 10  # Number of rows to display per page
        self.root.resizable(False, False)
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Face Embeddings Manager")
        self.file_manager_frame = ttk.Frame(self.root)
        self.file_manager_frame.pack(fill="both", expand=True)

        # Table to display .npz files
        self.tree = ttk.Treeview(self.file_manager_frame, columns=("UID", "Name", "Actions"), show="headings")
        self.tree.heading("UID", text="UID")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Actions", text="Actions")
        self.tree.column("UID", width=150)
        self.tree.column("Name", width=150)
        self.tree.column("Actions", width=100)
        self.tree.pack(fill="both", expand=True)

        # Right-click menu for deletion
        self.right_click_menu = Menu(self.root, tearoff=0)
        self.right_click_menu.add_command(label="Delete", command=self.delete_selected_file)
        self.tree.bind("<Button-3>", self.show_right_click_menu)

        # Pagination controls
        self.helper_frame = Frame(self.root)
        # Create a helper frame to center the pagination_frame
        self.helper_frame = Frame(self.root)
        self.helper_frame.pack(fill="x", pady=10)

        # Place the pagination_frame inside the helper frame and center it
        self.pagination_frame = Frame(self.helper_frame)
        self.pagination_frame.pack(anchor="center") 
        self.first_page_button = Button(self.pagination_frame, text="<<", command=self.go_to_first_page)
        self.first_page_button.pack(side="left", padx=5)

        self.prev_page_button = Button(self.pagination_frame, text="<", command=self.go_to_prev_page)
        self.prev_page_button.pack(side="left", padx=5)

        self.page_search_label = Label(self.pagination_frame, text="Page:")
        self.page_search_label.pack(side="left", padx=5)

        self.page_search_entry = Entry(self.pagination_frame, width=5)
        self.page_search_entry.pack(side="left", padx=5)
        self.page_search_entry.bind("<Return>", self.go_to_page)

        self.next_page_button = Button(self.pagination_frame, text=">", command=self.go_to_next_page)
        self.next_page_button.pack(side="left", padx=5)

        self.last_page_button = Button(self.pagination_frame, text=">>", command=self.go_to_last_page)
        self.last_page_button.pack(side="left", padx=5)

        # Add New Person button
        self.add_person_button = ttk.Button(self.root, text="Add New Person", command=self.open_add_person_window)
        self.add_person_button.pack(pady=10, anchor="center")
        self.tree.bind("<Double-1>", self.on_row_double_click)

        # Load .npz files into the table
        self.load_npz_files()

    def load_npz_files(self):
        """Load and display existing .npz files in the directory."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        files = [f for f in os.listdir(self.pipeline.face_database_dir) if f.endswith(".npz")]
        total_pages = (len(files) + self.rows_per_page - 1) // self.rows_per_page

        # Update pagination controls
        self.page_search_entry.delete(0, "end")
        self.page_search_entry.insert(0, str(self.current_page))

        # Display rows for the current page
        start_index = (self.current_page - 1) * self.rows_per_page
        end_index = start_index + self.rows_per_page
        for file_name in files[start_index:end_index]:
            npz_path = os.path.join(self.pipeline.face_database_dir, file_name)
            data = np.load(npz_path)
            uid = data.get("uid", "N/A")
            name = data.get("person_name", os.path.splitext(file_name)[0])
            self.tree.insert("", "end", values=(uid, name, "Show Records"))

    def show_right_click_menu(self, event):
        """Show the right-click menu for deleting files."""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.right_click_menu.post(event.x_root, event.y_root)


    def delete_selected_file(self):
        """Delete the selected .npz file and its corresponding folder based on UID."""
        selected_item = self.tree.selection()
        if selected_item:
            uid = self.tree.item(selected_item, "values")[0]
            name = self.tree.item(selected_item, "values")[1]
            records_dir = os.path.join(self.pipeline.face_database_dir, "records")
            matching_folders = glob.glob(os.path.join(records_dir, f"*{uid}*"))

            if matching_folders:
                for folder in matching_folders:
                    if os.path.isdir(folder):
                        shutil.rmtree(folder, ignore_errors=True)  # Delete folder
                        self.load_npz_files()
                        messagebox.showinfo("Success", f"Deleted folder {folder}")
            else:
                messagebox.showerror("Error", f"No folder matching UID {uid} found.")

            file_name = f"{name}_{uid}.npz"
            file_path = os.path.join(self.pipeline.face_database_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                self.load_npz_files()
                messagebox.showinfo("Success", f"Deleted {file_name}")
                self.pipeline.load_embeddings()
                self.pipeline.update_recognized_faces()
            else:
                messagebox.showerror("Error", f"File {file_name} not found.")

    def open_add_person_window(self):
        """Open the Add New Person window and keep it above FaceEmbeddingApp (Window 2)."""

        if hasattr(self, "add_person_window") and self.add_person_window is not None and self.add_person_window.winfo_exists():
            self.add_person_window.lift()
            self.add_person_window.attributes("-topmost", True)
            return  

        self.add_person_window = Toplevel(self.root)
        self.add_person_window.title("Add New Person")
        self.add_person_window.geometry("270x150")
        self.add_person_window.resizable(False, False)

        # Create a container frame for better alignment
        container = ttk.Frame(self.add_person_window)
        container.pack(fill="both", expand=True, padx=10, pady=10, anchor="w")  # Left-align everything

        # Name Entry with Placeholder Text
        self.name_entry = ttk.Entry(container, foreground="gray", background="black")
        self.name_entry.pack(fill="x", padx=5, pady=5)  # Expand to fill space
        self.name_entry.insert(0, "Enter person's name...")  # Default hint text
        self.name_entry.bind("<FocusIn>", self.clear_placeholder)
        self.name_entry.bind("<FocusOut>", self.restore_placeholder)

        # Add Video Button + Status Indicator
        video_frame = ttk.Frame(container)
        video_frame.pack(fill="x", pady=5, anchor="w")  # Left-align and maintain spacing

        self.add_video_button = ttk.Button(video_frame, text="Add Video", command=self.select_video)
        self.add_video_button.pack(side="left", padx=5)  # Align button to the left

        # Video status indicator (hidden initially)
        self.video_status_label = ttk.Label(video_frame, text="✅", foreground="green")
        self.video_status_label.pack(side="left", padx=5)
        self.video_status_label.pack_forget()  # Hide initially

        # Train Button (also left-aligned)
        self.train_button = ttk.Button(container, text="Train", command=self.train_new_person)
        self.train_button.pack(fill="x", padx=5, pady=10, anchor="w")  # Full width, left-aligned

        # Keep the window always on top
        self.add_person_window.transient(self.root)  
        self.add_person_window.lift()  
        self.add_person_window.attributes("-topmost", True)  

        self.add_person_window.protocol("WM_DELETE_WINDOW", self.on_close_add_person_window)
        self.root.bind("<FocusIn>", lambda e: self.lift_add_person_window())

    def clear_placeholder(self, event):
        """Clear placeholder text when user clicks inside entry field."""
        if self.name_entry.get() == "Enter person's name...":
            self.name_entry.delete(0, "end")
            self.name_entry.config(foreground="white")  # User input remains white

    def restore_placeholder(self, event):
        """Restore placeholder text when entry field is empty."""
        if not self.name_entry.get():
            self.name_entry.insert(0, "Enter person's name...")
            self.name_entry.config(foreground="gray")


    def lift_add_person_window(self):
        """Safely bring Add Person Window to the top only if it exists."""
        if self.add_person_window is not None and self.add_person_window.winfo_exists():
            self.add_person_window.lift()
            self.add_person_window.attributes("-topmost", True)

    def on_close_add_person_window(self):
        """Handle the closing of Add Person Window."""
        if self.add_person_window is not None:
            self.add_person_window.destroy()
            self.add_person_window = None 

    def select_video(self):
        """Select a video file."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        
        if self.video_path:
            messagebox.showinfo("Success", f"Selected video: {os.path.basename(self.video_path)}")
            self.video_status_label.pack(side="left", padx=5)  # indicator ✅

    def train_new_person(self):
        """Train the system with the new person's details and start processing the video."""
        person_name = self.name_entry.get()
        
        if not person_name:
            messagebox.showerror("Error", "Please enter the person's name.")
            return
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video.")
            return

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.loading_dialog = LoadingDialog(self.root, total_frames)
        training_pipeline = TrainingPipeline()
        threading.Thread(target=self.process_video, args=(training_pipeline, person_name, self.video_path)).start()

        # Close the "Add Person" window after training starts
        self.on_close_add_person_window()
        
    def process_video(self, training_pipeline, person_name, video_path):
        """Process the video to extract frames and create embeddings."""
        npz_path = training_pipeline.process_video(person_name, video_path, self.loading_dialog)
        self.loading_dialog.close()
        if npz_path:
            messagebox.showinfo("Success", f"Saved embeddings for {person_name} to {npz_path}")
        self.load_npz_files()
        self.pipeline.load_embeddings()

    def go_to_first_page(self):
        """Navigate to the first page."""
        self.current_page = 1
        self.load_npz_files()

    def go_to_prev_page(self):
        """Navigate to the previous page."""
        if self.current_page > 1:
            self.current_page -= 1
            self.load_npz_files()

    def go_to_next_page(self):
        """Navigate to the next page."""
        files = [f for f in os.listdir(self.pipeline.face_database_dir) if f.endswith(".npz")]
        total_pages = (len(files) + self.rows_per_page - 1) // self.rows_per_page
        if self.current_page < total_pages:
            self.current_page += 1
            self.load_npz_files()

    def go_to_last_page(self):
        """Navigate to the last page."""
        files = [f for f in os.listdir(self.pipeline.face_database_dir) if f.endswith(".npz")]
        total_pages = (len(files) + self.rows_per_page - 1) // self.rows_per_page
        self.current_page = total_pages
        self.load_npz_files()

    def go_to_page(self, event=None):
        """Navigate to a specific page."""
        try:
            page = int(self.page_search_entry.get())
            files = [f for f in os.listdir(self.pipeline.face_database_dir) if f.endswith(".npz")]
            total_pages = (len(files) + self.rows_per_page - 1) // self.rows_per_page
            if 1 <= page <= total_pages:
                self.current_page = page
                self.load_npz_files()
            else:
                messagebox.showerror("Error", f"Page number must be between 1 and {total_pages}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid page number.")
            
    def show_records(self, uid):
        """Open the Records window for the selected UID."""
        records_dir = os.path.join(self.pipeline.face_database_dir, "records")
        folder_path = os.path.join(records_dir, f"{uid}")
        if os.path.exists(folder_path):
            RecordsWindow(self.root, folder_path)
        else:
            messagebox.showerror("Error", f"No records found for UID: {uid}")
            
    def on_row_double_click(self, event):
        """Handle double-click event on a table row."""
        selected_item = self.tree.selection()
        if selected_item:
            uid = self.tree.item(selected_item, "values")[0]
            self.show_records(uid)
