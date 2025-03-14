
import os
import cv2
from tkinter import ttk, Toplevel, Frame, Label, Canvas, messagebox
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime

class RecordsWindow:
    instance = None  # Track the existing instance

    def __init__(self, root, folder_path):
        """Initialize the Records window."""
        if RecordsWindow.instance is not None and RecordsWindow.instance.winfo_exists():
            RecordsWindow.instance.lift()  # Bring existing window to front
            RecordsWindow.instance.attributes("-topmost", True)
            return  # Prevent multiple instances

        RecordsWindow.instance = self  # Set current instance

        self.root = root
        self.folder_path = folder_path
        self.sort_order = "desc"  # Default sort order: newest to oldest
        self.files = []  # Store file data for dynamic rendering
        self.visible_rows = 20  # Number of rows to render at a time
        self.current_offset = 0  # Current scroll offset

        self.setup_ui()
        self.records_window.resizable(False, False)
        self.records_window.attributes("-topmost", True)  # Keep on top
        self.records_window.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle close event

    def setup_ui(self):
        """Set up the UI for the Records window."""
        self.records_window = Toplevel(self.root)
        self.records_window.title("Records")
        self.records_window.geometry("700x400")

        self.main_container = Frame(self.records_window)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        self.table_frame = Frame(self.main_container)
        self.table_frame.pack(side="left", fill="both", expand=True)

        self.preview_frame = Frame(self.main_container, width=300)
        self.preview_frame.pack(side="right", fill="y", expand=False, padx=(20, 0))

        self.preview_label = Label(self.preview_frame, text="Image Preview", font=("Arial", 12))
        self.preview_label.pack(pady=10)

        self.canvas = Canvas(self.preview_frame, width=280, height=280)
        self.canvas.pack(pady=20)

        self.tree = ttk.Treeview(self.table_frame, columns=("Frame Name", "Last Seen"), show="headings")
        self.tree.heading("Frame Name", text="Frame Name")
        self.tree.heading("Last Seen", text="Last Seen", command=self.sort_by_last_seen)
        self.tree.column("Frame Name", width=100)
        self.tree.column("Last Seen", width=200)
        self.tree.pack(fill="both", expand=True)

        self.tree.bind("<MouseWheel>", self.on_mousewheel)
        self.tree.bind("<<TreeviewSelect>>", self.show_image_preview)

        self.load_folder_contents()

    def load_folder_contents(self):
        """Load the contents of the folder into the table."""
        if not os.path.exists(self.folder_path):
            messagebox.showerror("Error", f"Folder {self.folder_path} does not exist.")
            return

        self.files = []
        for file_name in os.listdir(self.folder_path):
            if file_name.endswith((".jpg", ".png")):
                file_path = os.path.join(self.folder_path, file_name)
                last_seen = self.get_last_seen_timestamp(file_path)
                self.files.append((file_name, last_seen))

        self.sort_files()
        self.render_visible_rows()

    def sort_files(self):
        """Sort the files based on the numerical order of the jpg file names."""
        def extract_number(file_name):
            name, ext = os.path.splitext(file_name)
            return int(name) if name.isdigit() else float('inf')  # Handle non-numeric cases

        if self.sort_order == "desc":
            self.files.sort(key=lambda x: extract_number(x[0]), reverse=True)  # Highest number first
        else:
            self.files.sort(key=lambda x: extract_number(x[0]))  # Lowest number first

    def render_visible_rows(self):
        """Render only the visible rows in the table."""
        self.tree.delete(*self.tree.get_children())  # Clear existing rows

        start, end = self.current_offset, min(self.current_offset + self.visible_rows, len(self.files))
        for i in range(start, end):
            file_name, last_seen = self.files[i]
            self.tree.insert("", "end", values=(file_name, last_seen))

    def on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        if event.delta < 0:  # Scroll down
            self.current_offset = min(self.current_offset + 1, len(self.files) - self.visible_rows)
        else:  # Scroll up
            self.current_offset = max(self.current_offset - 1, 0)

        self.render_visible_rows()

    def get_last_seen_timestamp(self, file_path):
        """
        Get the last modified timestamp of a file in human-readable format.

        Args:
            file_path: Path to the file.

        Returns:
            A string representing the last modified timestamp in "YYYY-MM-DD HH:MM:SS" format.
        """
        timestamp = os.path.getmtime(file_path)  # Get last modified time
        formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return formatted_time

    def show_image_preview(self, event=None):
        """Display the selected image in the preview area."""
        selected_item = self.tree.selection()
        if selected_item:
            file_name = self.tree.item(selected_item, "values")[0]
            file_path = os.path.join(self.folder_path, file_name)

            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image.thumbnail((280, 280))

                photo = ImageTk.PhotoImage(image)
                self.canvas.delete("all")  # Clear previous image
                self.canvas.create_image(0, 0, anchor="nw", image=photo)
                self.photo = photo  # Keep a reference
            else:
                messagebox.showerror("Error", f"Unable to load image: {file_name}")

    def sort_by_last_seen(self):
        """Sort the table by the 'Last Seen' column."""
        self.sort_order = "asc" if self.sort_order == "desc" else "desc"
        self.sort_files()
        self.render_visible_rows()

    def on_close(self):
        """Handle closing of the Records window."""
        RecordsWindow.instance = None  # Reset instance reference
        self.records_window.destroy()


# import os
# import cv2
# from tkinter import ttk, Toplevel, Frame, Label, Canvas, messagebox
# from PIL import Image, ImageTk
# import numpy as np

# class RecordsWindow:
#     instance = None  # Track the existing instance

#     def __init__(self, root, folder_path):
#         """
#         Initialize the Records window.

#         Args:
#             root: The parent window.
#             folder_path: Path to the folder containing the images and metadata.
#         """

#         if RecordsWindow.instance is not None and RecordsWindow.instance.winfo_exists():
#             RecordsWindow.instance.lift()  # Bring existing window to front
#             RecordsWindow.instance.attributes("-topmost", True)
#             return  # Prevent multiple instances

#         RecordsWindow.instance = self  # Set current instance

#         self.root = root
#         self.folder_path = folder_path
#         self.sort_order = "desc"  # Default sort order: newest to oldest
#         self.files = []  # Store file data for dynamic rendering
#         self.visible_rows = 20  # Number of rows to render at a time
#         self.current_offset = 0  # Current scroll offset

#         self.setup_ui()
#         self.records_window.resizable(False, False)
#         self.records_window.attributes("-topmost", True)  # Keep on top
#         self.records_window.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle close event

#     def setup_ui(self):
#         """Set up the UI for the Records window."""
#         self.records_window = Toplevel(self.root)
#         self.records_window.title("Records")
#         self.records_window.geometry("700x400")

#         self.main_container = Frame(self.records_window)
#         self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

#         self.table_frame = Frame(self.main_container)
#         self.table_frame.pack(side="left", fill="both", expand=True)

#         self.preview_frame = Frame(self.main_container, width=300)
#         self.preview_frame.pack(side="right", fill="y", expand=False, padx=(20, 0))

#         self.preview_label = Label(self.preview_frame, text="Image Preview", font=("Arial", 12))
#         self.preview_label.pack(pady=10)

#         self.canvas = Canvas(self.preview_frame, width=280, height=280)
#         self.canvas.pack(pady=20)

#         self.tree = ttk.Treeview(self.table_frame, columns=("Frame Name", "Last Seen"), show="headings")
#         self.tree.heading("Frame Name", text="Frame Name")
#         self.tree.heading("Last Seen", text="Last Seen", command=self.sort_by_last_seen)
#         self.tree.column("Frame Name", width=100)
#         self.tree.column("Last Seen", width=200)
#         self.tree.pack(fill="both", expand=True)

#         self.tree.bind("<MouseWheel>", self.on_mousewheel)
#         self.tree.bind("<<TreeviewSelect>>", self.show_image_preview)

#         self.load_folder_contents()

#     def load_folder_contents(self):
#         """Load the contents of the folder into the table."""
#         if not os.path.exists(self.folder_path):
#             messagebox.showerror("Error", f"Folder {self.folder_path} does not exist.")
#             return

#         self.files = []
#         for file_name in os.listdir(self.folder_path):
#             if file_name.endswith((".jpg", ".png")):
#                 file_path = os.path.join(self.folder_path, file_name)
#                 last_seen = self.get_last_seen_timestamp(file_path)
#                 self.files.append((file_name, last_seen))

#         self.sort_files()
#         self.render_visible_rows()

#     def sort_files(self):
#         """Sort the files based on the selected order."""
#         if self.sort_order == "desc":
#             self.files.sort(key=lambda x: x[1], reverse=True)  # Newest first
#         else:
#             self.files.sort(key=lambda x: x[1])  # Oldest first

#     def render_visible_rows(self):
#         """Render only the visible rows in the table."""
#         self.tree.delete(*self.tree.get_children())  # Clear existing rows

#         start, end = self.current_offset, min(self.current_offset + self.visible_rows, len(self.files))
#         for i in range(start, end):
#             file_name, last_seen = self.files[i]
#             self.tree.insert("", "end", values=(file_name, last_seen))

#     def on_mousewheel(self, event):
#         """Handle mousewheel scrolling."""
#         if event.delta < 0:  # Scroll down
#             self.current_offset = min(self.current_offset + 1, len(self.files) - self.visible_rows)
#         else:  # Scroll up
#             self.current_offset = max(self.current_offset - 1, 0)

#         self.render_visible_rows()

#     def get_last_seen_timestamp(self, file_path):
#         """
#         Get the last modified timestamp of a file.

#         Args:
#             file_path: Path to the file.

#         Returns:
#             A string representing the last modified timestamp.
#         """
#         timestamp = os.path.getmtime(file_path)
#         return np.datetime64(int(timestamp * 1000), 'ms').astype(str)

#     def show_image_preview(self, event=None):
#         """Display the selected image in the preview area."""
#         selected_item = self.tree.selection()
#         if selected_item:
#             file_name = self.tree.item(selected_item, "values")[0]
#             file_path = os.path.join(self.folder_path, file_name)

#             image = cv2.imread(file_path)
#             if image is not None:
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = Image.fromarray(image)
#                 image.thumbnail((280, 280))

#                 photo = ImageTk.PhotoImage(image)
#                 self.canvas.delete("all")  # Clear previous image
#                 self.canvas.create_image(0, 0, anchor="nw", image=photo)
#                 self.photo = photo  # Keep a reference
#             else:
#                 messagebox.showerror("Error", f"Unable to load image: {file_name}")

#     def sort_by_last_seen(self):
#         """Sort the table by the 'Last Seen' column."""
#         self.sort_order = "asc" if self.sort_order == "desc" else "desc"
#         self.sort_files()
#         self.render_visible_rows()

#     def on_close(self):
#         """Handle closing of the Records window."""
#         RecordsWindow.instance = None  # Reset instance reference
#         self.records_window.destroy()

# import os
# import cv2
# from tkinter import ttk, Toplevel, Frame, Label, Canvas, Scrollbar, messagebox
# from PIL import Image, ImageTk
# import numpy as np

# class RecordsWindow:
#     def __init__(self, root, folder_path):
#         """
#         Initialize the Records window.

#         Args:
#             root: The parent window.
#             folder_path: Path to the folder containing the images and metadata.
#         """
#         self.root = root
#         self.folder_path = folder_path
#         self.sort_order = "desc"  # Default sort order: newest to oldest
#         self.setup_ui()

#     def setup_ui(self):
#         """Set up the UI for the Records window."""
#         self.records_window = Toplevel(self.root)
#         self.records_window.title("Records")
#         self.records_window.geometry("1000x600")  # Adjust window size

#         # Main container (flex row)
#         self.main_container = Frame(self.records_window)
#         self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

#         # Left side: Table view (flex col)
#         self.table_frame = Frame(self.main_container)
#         self.table_frame.pack(side="left", fill="both", expand=True)

#         # Right side: Image preview (flex col with vertical centering)
#         self.preview_frame = Frame(self.main_container, width=300)
#         self.preview_frame.pack(side="right", fill="y", expand=False, padx=(20, 0))

#         # Add a label for the image preview
#         self.preview_label = Label(self.preview_frame, text="Image Preview", font=("Arial", 12))
#         self.preview_label.pack(pady=10)

#         # Canvas for displaying the image
#         self.canvas = Canvas(self.preview_frame, width=280, height=280)
#         self.canvas.pack(pady=20)

#         # Table to display folder contents
#         self.tree = ttk.Treeview(self.table_frame, columns=("Frame Name", "Last Seen"), show="headings")
#         self.tree.heading("Frame Name", text="Frame Name")
#         self.tree.heading("Last Seen", text="Last Seen", command=self.sort_by_last_seen)  # Make column clickable
#         self.tree.column("Frame Name", width=400)
#         self.tree.column("Last Seen", width=200)
#         self.tree.pack(fill="both", expand=True)

#         # Add scrollbar to the table
#         scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
#         self.tree.configure(yscrollcommand=scrollbar.set)
#         scrollbar.pack(side="right", fill="y")

#         # Bind row selection event to show image preview
#         self.tree.bind("<<TreeviewSelect>>", self.show_image_preview)

#         # Load folder contents into the table
#         self.load_folder_contents()

#     def load_folder_contents(self):
#         """Load the contents of the folder into the table."""
#         if not os.path.exists(self.folder_path):
#             messagebox.showerror("Error", f"Folder {self.folder_path} does not exist.")
#             return

#         # Clear existing rows
#         for item in self.tree.get_children():
#             self.tree.delete(item)

#         # Iterate through files in the folder
#         files = []
#         for file_name in os.listdir(self.folder_path):
#             if file_name.endswith(".jpg") or file_name.endswith(".png"):
#                 file_path = os.path.join(self.folder_path, file_name)
#                 last_seen = self.get_last_seen_timestamp(file_path)
#                 files.append((file_name, last_seen))

#         # Sort files based on the current sort order
#         if self.sort_order == "desc":
#             files.sort(key=lambda x: x[1], reverse=True)  # Newest to oldest
#         else:
#             files.sort(key=lambda x: x[1])  # Oldest to newest

#         # Insert sorted files into the table
#         for file_name, last_seen in files:
#             self.tree.insert("", "end", values=(file_name, last_seen))

#     def get_last_seen_timestamp(self, file_path):
#         """
#         Get the last modified timestamp of a file.

#         Args:
#             file_path: Path to the file.

#         Returns:
#             A string representing the last modified timestamp.
#         """
#         timestamp = os.path.getmtime(file_path)
#         return np.datetime64(int(timestamp * 1000), 'ms').astype(str)

#     def show_image_preview(self, event):
#         """Display the selected image in the preview area."""
#         selected_item = self.tree.selection()
#         if selected_item:
#             file_name = self.tree.item(selected_item, "values")[0]
#             file_path = os.path.join(self.folder_path, file_name)

#             # Load the image using OpenCV
#             image = cv2.imread(file_path)
#             if image is not None:
#                 # Convert the image to RGB format
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = Image.fromarray(image)

#                 # Resize the image to fit the canvas
#                 image.thumbnail((280, 280))
#                 photo = ImageTk.PhotoImage(image)

#                 # Update the canvas with the new image
#                 self.canvas.create_image(0, 0, anchor="nw", image=photo)
#                 self.canvas.image = photo  # Keep a reference to avoid garbage collection
#             else:
#                 messagebox.showerror("Error", f"Unable to load image: {file_name}")

#     def sort_by_last_seen(self):
#         """Sort the table by the 'Last Seen' column."""
#         # Toggle the sort order
#         if self.sort_order == "desc":
#             self.sort_order = "asc"  # Oldest to newest
#         else:
#             self.sort_order = "desc"  # Newest to oldest

#         # Reload the folder contents with the new sort order
#         self.load_folder_contents()