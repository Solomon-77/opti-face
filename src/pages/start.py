import tkinter as tk
from tkinter import ttk, messagebox, Canvas, StringVar, Menu, filedialog, Frame, Label, Entry, Button
import sv_ttk
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import threading
import shutil
import glob
from pipelines.training_pipeline import TrainingPipeline
from pages.dialogboxes.progress import LoadingDialog
from pages.records import RecordsWindow

class MainApp:
    """
    Main application: shows a login screen first then a main menu with four buttons.
    All functionality is contained within a single window using frame switching.
    """
    def __init__(self, root, window_title, pipeline):
        self.root = root
        self.root.title(window_title)
        self.pipeline = pipeline
        self.logged_in = False
        self.video_path = None
        self.add_person_window = None

        # Set dark theme
        sv_ttk.set_theme("dark")
        self.root.geometry("800x600")
        self.root.configure(bg="#2A2A2A")
        
        # Create container frame for all pages
        self.container = ttk.Frame(self.root)
        self.container.pack(fill="both", expand=True)
        
        # Show login screen first
        self.show_login_screen()

    def show_login_screen(self):
        # Clear container
        for widget in self.container.winfo_children():
            widget.destroy()
            
        login_frame = ttk.Frame(self.container)
        login_frame.pack(fill="both", expand=True)
        
        content_frame = ttk.Frame(login_frame)
        content_frame.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(content_frame, text="Login", font=("Arial", 20, "bold")).pack(pady=10)

        ttk.Label(content_frame, text="Username:").pack(pady=(10, 0))
        self.username_entry = ttk.Entry(content_frame, width=30)
        self.username_entry.pack(pady=(0, 10))

        ttk.Label(content_frame, text="Password:").pack(pady=(10, 0))
        self.password_entry = ttk.Entry(content_frame, show="*", width=30)
        self.password_entry.pack(pady=(0, 10))

        ttk.Button(content_frame, text="Login", command=self.check_credentials).pack(pady=20)

    def check_credentials(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if username == "admin" and password == "admin123":
            self.logged_in = True
            self.show_main_menu()
        else:
            messagebox.showerror("Login Failed", "Incorrect username or password.")

    def show_main_menu(self):
        # Clear container
        for widget in self.container.winfo_children():
            widget.destroy()
            
        main_menu_frame = ttk.Frame(self.container)
        main_menu_frame.pack(fill="both", expand=True)
        
        ttk.Label(main_menu_frame, text="Main Menu", font=("Arial", 16, "bold")).pack(pady=20)

        menu_frame = ttk.Frame(main_menu_frame)
        menu_frame.pack(pady=40)

        ttk.Button(menu_frame, text="Camera", command=self.show_camera_page, width=15).grid(row=0, column=0, padx=20)
        ttk.Button(menu_frame, text="Registration", command=self.show_registration_page, width=15).grid(row=0, column=1, padx=20)
        ttk.Button(menu_frame, text="Settings", command=self.show_settings_page, width=15).grid(row=0, column=2, padx=20)
        ttk.Button(menu_frame, text="User", command=self.show_user_settings, width=15).grid(row=0, column=3, padx=20)

    # ===== CAMERA PAGE =====
    def show_camera_page(self):
    # Clear container
        for widget in self.container.winfo_children():
            widget.destroy()
            
        camera_frame = ttk.Frame(self.container)
        camera_frame.pack(fill="both", expand=True)
            
        # Header with title, back button, and expand button
        header_frame = ttk.Frame(camera_frame)
        header_frame.pack(fill="x", pady=10)
            
        ttk.Button(header_frame, text="← Back", command=self.show_main_menu).pack(side="left", padx=20)
        ttk.Label(header_frame, text="Camera Feed", font=("Arial", 14, "bold")).pack(side="left", padx=20)
        
        # Add an expand button on the UI
        expand_button = ttk.Button(header_frame, text="Expand Fullscreen", command=self.open_fullscreen)
        expand_button.pack(side="right", padx=20)
            
        # Create a canvas to display the video feed
        self.canvas = Canvas(camera_frame, width=640, height=480)
        self.canvas.pack(pady=10)
            
        # Open the camera and update the frame
        self.cap = cv2.VideoCapture(0)
        self.update_frame()
            
        # When leaving this page, release the camera
        self.container.bind("<Destroy>", lambda e: self.release_camera())

    def open_fullscreen(self):
        # Create a new Toplevel window and set it to full screen
        fs_window = tk.Toplevel(self.root)
        fs_window.title("Fullscreen Camera")
        fs_window.attributes('-fullscreen', True)
        fs_window.configure(bg='black')
        
        # Create a canvas for the full-screen window that fills the entire space
        fs_canvas = tk.Canvas(fs_window, bg='black')
        fs_canvas.pack(fill="both", expand=True)
        
        # Create the minimize button
        minimize_button = ttk.Button(fs_window, text="Exit Fullscreen", command=fs_window.destroy)
        minimize_button.place(relx=0.95, rely=0.05, anchor="ne")
        
        # Store the after() id for cancelling auto-hide if needed
        fs_window.hide_button_after_id = None
        # Flag to track if window is being destroyed
        fs_window.is_closing = False

        def hide_minimize_button():
            minimize_button.place_forget()

        def on_mouse_motion(event):
            # Show the button at the top-right when mouse moves
            minimize_button.place(relx=0.95, rely=0.05, anchor="ne")
            # Cancel any previously scheduled hide calls
            if fs_window.hide_button_after_id:
                fs_window.after_cancel(fs_window.hide_button_after_id)
            # Schedule the minimize button to hide after 2 seconds of no mouse movement
            fs_window.hide_button_after_id = fs_window.after(2000, hide_minimize_button)
        
        # Bind mouse motion on the full-screen window
        fs_window.bind("<Motion>", on_mouse_motion)
        # Add Escape key binding to close fullscreen
        fs_window.bind("<Escape>", lambda e: fs_window.destroy())

        # Function to properly handle window closing
        def on_window_close():
            fs_window.is_closing = True
            # Cancel any pending after callbacks
            if hasattr(fs_window, '_job') and fs_window._job:
                fs_window.after_cancel(fs_window._job)
            fs_window.destroy()

        # Override close button behavior
        fs_window.protocol("WM_DELETE_WINDOW", on_window_close)

        def update_fs():
            # Stop updating if window is closing
            if fs_window.is_closing or not fs_window.winfo_exists():
                return
            
            # Clear the canvas
            fs_canvas.delete("all")
            
            # Get the dimensions of the fullscreen window
            screen_width = fs_window.winfo_width()
            screen_height = fs_window.winfo_height()

            # Ensure the window has been rendered and has valid dimensions
            if screen_width <= 1 or screen_height <= 1:
                # Window not ready yet, retry after a short delay
                fs_window._job = fs_window.after(50, update_fs)
                return
                
            if hasattr(self, 'cap') and self.cap.isOpened():
                # Capture a fresh frame for fullscreen
                ret, frame = self.cap.read()
                if ret:
                    # Process the frame through the pipeline
                    processed = self.pipeline.process_frame(frame)
                    if processed is not None:
                        rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PIL Image
                        img = Image.fromarray(rgb_frame)
                        
                        # Calculate scaling to fit screen while maintaining aspect ratio
                        img_width, img_height = img.size
                        aspect_ratio = img_width / img_height
                        
                        if screen_width / screen_height > aspect_ratio:
                            # Screen is wider than video
                            new_height = screen_height
                            new_width = int(new_height * aspect_ratio)
                        else:
                            # Screen is taller than video
                            new_width = screen_width
                            new_height = int(new_width / aspect_ratio)
                        
                        # Resize the image
                        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                        
                        # Create PhotoImage
                        try:
                            photo = ImageTk.PhotoImage(image=img_resized)
                            
                            # Calculate position to center the image
                            x_pos = (screen_width - new_width) // 2
                            y_pos = (screen_height - new_height) // 2
                            
                            # Draw on canvas
                            fs_canvas.create_image(x_pos, y_pos, image=photo, anchor="nw")
                            
                            # Keep reference to prevent garbage collection
                            fs_canvas.photo = photo
                        except Exception as e:
                            print(f"Error creating photo: {e}")
            
            # Schedule next update if window still exists
            if not fs_window.is_closing and fs_window.winfo_exists():
                fs_window._job = fs_window.after(30, update_fs)
        
        # Start the update loop
        fs_window._job = fs_window.after(100, update_fs)

    def update_frame(self):
        if not hasattr(self, 'cap') or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Process the frame through your pipeline
            processed = self.pipeline.process_frame(frame)
            if processed is not None:
                rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                # Save the current frame for use in full screen
                self.current_frame = rgb_frame.copy()
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        if hasattr(self, 'canvas') and self.canvas.winfo_exists():
            self.canvas.after(10, self.update_frame)


    def release_camera(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
    
    # ===== SETTINGS PAGE =====
    def show_settings_page(self):
        # Clear container
        for widget in self.container.winfo_children():
            widget.destroy()
            
        settings_frame = ttk.Frame(self.container)
        settings_frame.pack(fill="both", expand=True)
        
        # Header with title and back button
        header_frame = ttk.Frame(settings_frame)
        header_frame.pack(fill="x", pady=10)
        
        ttk.Button(header_frame, text="← Back", command=self.show_main_menu).pack(side="left", padx=20)
        ttk.Label(header_frame, text="Settings", font=("Arial", 14, "bold")).pack(side="left", padx=20)
        
        # Settings content
        main_frame = ttk.Frame(settings_frame, padding=10)
        main_frame.pack(fill="both", expand=True)

        ttk.Label(main_frame, text="Threshold Configuration", font=("Arial", 14, "bold")).pack(pady=5)

        ttk.Label(main_frame, text="Minimum Accuracy Threshold:").pack(pady=(10, 0))
        self.min_accuracy_var = StringVar(value=str(self.pipeline.min_accuracy))
        self.min_accuracy_entry = ttk.Entry(main_frame, textvariable=self.min_accuracy_var)
        self.min_accuracy_entry.pack(pady=(0, 10))

        ttk.Label(main_frame, text="Minimum Recognize Threshold:").pack(pady=(10, 0))
        self.min_recognize_var = StringVar(value=str(self.pipeline.min_recognize))
        self.min_recognize_entry = ttk.Entry(main_frame, textvariable=self.min_recognize_var)
        self.min_recognize_entry.pack(pady=(0, 10))

        ttk.Button(main_frame, text="Refresh Config", command=self.refresh_config).pack(pady=10)

    def refresh_config(self):
        try:
            self.pipeline.min_accuracy = float(self.min_accuracy_var.get())
            self.pipeline.min_recognize = float(self.min_recognize_var.get())
            messagebox.showinfo("Config Updated",
                               f"Min Accuracy = {self.pipeline.min_accuracy}\nMin Recognize = {self.pipeline.min_recognize}")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values.")
    
    # ===== REGISTRATION PAGE =====
    def show_registration_page(self):
        # Clear container
        for widget in self.container.winfo_children():
            widget.destroy()
            
        self.registration_frame = ttk.Frame(self.container)
        self.registration_frame.pack(fill="both", expand=True)
        
        # Header with title and back button
        header_frame = ttk.Frame(self.registration_frame)
        header_frame.pack(fill="x", pady=10)
        
        ttk.Button(header_frame, text="← Back", command=self.show_main_menu).pack(side="left", padx=20)
        ttk.Label(header_frame, text="Face Registration", font=("Arial", 14, "bold")).pack(side="left", padx=20)
        
        # Initialize registration page variables
        self.current_page = 1
        self.rows_per_page = 10
        
        # Table to display .npz files
        self.file_manager_frame = ttk.Frame(self.registration_frame)
        self.file_manager_frame.pack(fill="both", expand=True)
        
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
        self.tree.bind("<Double-1>", self.on_row_double_click)

        # Pagination controls
        self.helper_frame = Frame(self.registration_frame)
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
        self.add_person_button = ttk.Button(self.registration_frame, text="Add New Person", command=self.show_add_person_panel)
        self.add_person_button.pack(pady=10, anchor="center")

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
            
    def on_row_double_click(self, event):
        """Handle double-click event on a table row."""
        selected_item = self.tree.selection()
        if selected_item:
            uid = self.tree.item(selected_item, "values")[0]
            self.show_records(uid)
            
    def show_records(self, uid):
        """Open the Records window for the selected UID."""
        records_dir = os.path.join(self.pipeline.face_database_dir, "records")
        folder_path = os.path.join(records_dir, f"{uid}")
        if os.path.exists(folder_path):
            RecordsWindow(self.root, folder_path)  # This stays as a popup window
        else:
            messagebox.showerror("Error", f"No records found for UID: {uid}")
    
    # ===== ADD PERSON PANEL =====
    def show_add_person_panel(self):
        # Clear container
        for widget in self.container.winfo_children():
            widget.destroy()
            
        add_person_frame = ttk.Frame(self.container)
        add_person_frame.pack(fill="both", expand=True)
        
        # Header with title and back button
        header_frame = ttk.Frame(add_person_frame)
        header_frame.pack(fill="x", pady=10)
        
        ttk.Button(header_frame, text="← Back", command=self.show_registration_page).pack(side="left", padx=20)
        ttk.Label(header_frame, text="Add New Person", font=("Arial", 14, "bold")).pack(side="left", padx=20)
        
        # Add Person content
        content_frame = ttk.Frame(add_person_frame, padding=20)
        content_frame.pack(pady=40)

        # Name Entry with Placeholder Text
        ttk.Label(content_frame, text="Person's Name:").pack(anchor="w", pady=(0, 5))
        self.name_entry = ttk.Entry(content_frame, foreground="gray", width=30)
        self.name_entry.pack(fill="x", pady=(0, 20))
        self.name_entry.insert(0, "Enter person's name...")
        self.name_entry.bind("<FocusIn>", self.clear_placeholder)
        self.name_entry.bind("<FocusOut>", self.restore_placeholder)

        # Add Video Button + Status Indicator
        video_frame = ttk.Frame(content_frame)
        video_frame.pack(fill="x", pady=10, anchor="w")

        ttk.Button(video_frame, text="Select Video", command=self.select_video).pack(side="left")

        # Video status indicator (hidden initially)
        self.video_status_label = ttk.Label(video_frame, text="✅ Video Selected", foreground="green")
        self.video_status_label.pack(side="left", padx=10)
        self.video_status_label.pack_forget()  # Hide initially

        # Train Button
        ttk.Button(content_frame, text="Train Face Recognition", command=self.train_new_person).pack(fill="x", pady=20)

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

    def select_video(self):
        """Select a video file."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        
        if self.video_path:
            messagebox.showinfo("Success", f"Selected video: {os.path.basename(self.video_path)}")
            self.video_status_label.pack(side="left", padx=5)  # Show indicator

    def train_new_person(self):
        """Train the system with the new person's details and start processing the video."""
        person_name = self.name_entry.get()
        if person_name == "Enter person's name...":
            person_name = ""
            
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
        
    def process_video(self, training_pipeline, person_name, video_path):
        """Process the video to extract frames and create embeddings."""
        npz_path = training_pipeline.process_video(person_name, video_path, self.loading_dialog)
        self.loading_dialog.close()
        if npz_path:
            messagebox.showinfo("Success", f"Saved embeddings for {person_name} to {npz_path}")
            self.show_registration_page()  # Go back to registration page after training
        self.pipeline.load_embeddings()
    
    # ===== USER SETTINGS PAGE =====
    def show_user_settings(self):
        # Clear container
        for widget in self.container.winfo_children():
            widget.destroy()
            
        user_frame = ttk.Frame(self.container)
        user_frame.pack(fill="both", expand=True)
        
        # Header with title and back button
        header_frame = ttk.Frame(user_frame)
        header_frame.pack(fill="x", pady=10)
        
        ttk.Button(header_frame, text="← Back", command=self.show_main_menu).pack(side="left", padx=20)
        ttk.Label(header_frame, text="User Settings", font=("Arial", 14, "bold")).pack(side="left", padx=20)
        
        # Placeholder for user settings content
        content_frame = ttk.Frame(user_frame)
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ttk.Label(content_frame, text="User admin settings interface would appear here").pack(pady=50)
        messagebox.showinfo("User Settings", "User admin settings go here (change password, etc.).")