import os  # Detailed: Import the operating system module to work with file paths and directories.
         # Simple: Lets us work with folders and files.
import numpy as np  # Detailed: Import NumPy for numerical operations and handling arrays.
                   # Simple: Helps with math and arrays.
import tkinter as tk  # Detailed: Import the tkinter module for creating graphical user interfaces (GUIs).
                     # Simple: Allows us to make windows and buttons.
from tkinter import filedialog, messagebox, ttk  
# Detailed: Import specific tkinter components: filedialog for file selection dialogs, messagebox for pop-up messages, and ttk for themed widgets.
# Simple: These help with picking files, showing messages, and making cool UI elements.
import threading  # Detailed: Import threading for running tasks concurrently in separate threads.
                  # Simple: Lets the program do many things at the same time.
import cv2  # Detailed: Import OpenCV (cv2) for video and image processing tasks.
          # Simple: Helps with handling videos and pictures.
from utils.face_utils import preprocess_image, load_face_recognition_model  
# Detailed: Import custom functions for preprocessing images and loading a pre-trained face recognition model from the utils.face_utils module.
# Simple: These functions prepare images and load the face model.

class FaceEmbeddingApp:
    def __init__(self, root, pipeline):
        # Detailed: The initializer for the FaceEmbeddingApp class sets up the main GUI window and its components.
        # Simple: This starts our face embedding app and makes the window and buttons.
        self.root = root  # Detailed: Store the main window reference.
                           # Simple: Save the main window.
        self.pipeline = pipeline  # Detailed: Store the pipeline object that will be used to process embeddings later.
                                  # Simple: Save the processing pipeline.
        self.root.title("Face Embeddings Manager")  # Detailed: Set the title of the main window.
                                                     # Simple: Name the window.
        self.face_database_dir = './face_database/'  # Detailed: Define the directory where face images and embeddings are stored.
                                                      # Simple: Set the folder for face data.
        self.model, self.device = load_face_recognition_model()  
        # Detailed: Load the face recognition model and determine the device (CPU/GPU) to use for processing.
        # Simple: Load the face model and figure out which computer part (CPU or GPU) to use.

        # Ensure this window is always above the config window
        self.root.lift()  # Detailed: Bring the window to the front of the window stacking order.
                          # Simple: Make sure the window is on top.
        self.root.attributes("-topmost", True)  # Detailed: Set the window attribute to always be on top.
                                                # Simple: Keep the window above others.
        self.root.bind("<FocusIn>", lambda e: self.root.lift())  # Detailed: Bind an event to raise the window when it gains focus.
                                                                 # Simple: When clicked, bring it to the front.
        self.root.bind("<Map>", lambda e: self.root.lift())  # Detailed: Bind an event to ensure the window remains on top when mapped.
                                                            # Simple: Keep it on top when shown.

        # Create a frame for the file manager view
        self.file_manager_frame = ttk.Frame(root)  
        # Detailed: Create a frame widget inside the main window to hold file management widgets.
        # Simple: Make a container for file stuff.
        self.file_manager_frame.pack(fill=tk.BOTH, expand=True)  
        # Detailed: Pack the frame to fill the window both horizontally and vertically and expand as needed.
        # Simple: Make it grow to fill the window.

        # Treeview to show .npz files
        self.tree = ttk.Treeview(self.file_manager_frame, columns=("Name"), show="headings")
        # Detailed: Create a Treeview widget inside the file manager frame to display .npz files with a single column named "Name".
        # Simple: Create a list view to show our saved files.
        self.tree.heading("Name", text="Name")  # Detailed: Set the heading of the "Name" column to display "Name".
                                               # Simple: Label the column as "Name".
        self.tree.pack(fill=tk.BOTH, expand=True)  
        # Detailed: Pack the Treeview widget so that it fills the frame and expands as necessary.
        # Simple: Make the list view fill its container.

        # Right-click menu for deleting files
        self.right_click_menu = tk.Menu(self.root, tearoff=0)
        # Detailed: Create a popup (context) menu that will be shown on right-click events.
        # Simple: Make a menu that pops up when you right-click.
        self.right_click_menu.add_command(label="Delete", command=self.delete_selected_file)
        # Detailed: Add a "Delete" command to the popup menu that calls the delete_selected_file method.
        # Simple: Add a button in the menu that deletes a file.

        # Bind right-click event
        self.tree.bind("<Button-3>", self.show_right_click_menu)
        # Detailed: Bind the right-click mouse event (Button-3) on the Treeview to trigger the show_right_click_menu method.
        # Simple: When you right-click on the list, show the menu.

        # Add new person button
        self.add_person_button = ttk.Button(root, text="Add New Person", command=self.open_add_person_window)
        # Detailed: Create a button labeled "Add New Person" that, when clicked, opens a new window to add a new person.
        # Simple: Make a button that lets you add someone new.
        self.add_person_button.pack(pady=10)
        # Detailed: Pack the button with vertical padding for spacing.
        # Simple: Place the button with some space around it.

        # Load existing .npz files
        self.load_npz_files()  # Detailed: Call the method to load and display any existing .npz files in the face database directory.
                              # Simple: Show any saved face files.

    def load_npz_files(self):
        """Load and display existing .npz files in the directory."""
        # Detailed: Clear the current items in the Treeview and then iterate over files in the face database directory.
        # Simple: Erase the current file list and add the saved files.
        for item in self.tree.get_children():
            self.tree.delete(item)  # Detailed: Delete each existing item in the Treeview.
                                     # Simple: Remove old items.
        for file_name in os.listdir(self.face_database_dir):
            # Detailed: Loop over each file in the face database directory.
            # Simple: For each file in the folder:
            if file_name.endswith(".npz"):
                # Detailed: Check if the file has a .npz extension.
                # Simple: If it's a saved face file:
                self.tree.insert("", tk.END, values=(file_name,))
                # Detailed: Insert the file name into the Treeview.
                # Simple: Add the file to the list.

    def show_right_click_menu(self, event):
        """Show the right-click menu for deleting files."""
        # Detailed: Determine which row in the Treeview was clicked using the event coordinates.
        # Simple: Figure out which file you right-clicked.
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)  # Detailed: Select the identified row.
                                         # Simple: Highlight the file.
            self.right_click_menu.post(event.x_root, event.y_root)
            # Detailed: Display the right-click menu at the mouse pointer's screen coordinates.
            # Simple: Show the menu where you clicked.

    def delete_selected_file(self):
        """Delete the selected .npz file."""
        # Detailed: Get the selected item from the Treeview and delete the corresponding file from the file system.
        # Simple: Remove the chosen file.
        selected_item = self.tree.selection()
        if selected_item:
            file_name = self.tree.item(selected_item, "values")[0]
            # Detailed: Retrieve the file name from the selected Treeview item.
            # Simple: Get the file's name.
            file_path = os.path.join(self.face_database_dir, file_name)
            # Detailed: Construct the full file path for the selected file.
            # Simple: Create the full path for the file.
            os.remove(file_path)
            # Detailed: Remove the file from the file system.
            # Simple: Delete the file.
            self.load_npz_files()  # Detailed: Reload the Treeview to reflect the deletion.
                                  # Simple: Update the list of files.
            messagebox.showinfo("Success", f"Deleted {file_name}")
            # Detailed: Display a message box confirming that the file was deleted.
            # Simple: Tell the user the file was deleted.

            # Notify the pipeline to reload embeddings
            self.pipeline.load_embeddings()
            # Detailed: Call the pipeline's method to reload the embeddings after deletion.
            # Simple: Tell the system to update its face codes.
            self.pipeline.update_recognized_faces()
            # Detailed: Update the recognized faces in the pipeline based on the new embeddings.
            # Simple: Refresh the face display.

    def open_add_person_window(self):
        """Open a new window to add a new person."""
        # Detailed: Create a new top-level window that will allow the user to add a new person and upload their video.
        # Simple: Open a new window for adding someone.
        self.add_person_window = tk.Toplevel(self.root)
        self.add_person_window.title("Add New Person")
        # Detailed: Set the title of the new window.
        # Simple: Name the new window.

        # Ensure this window is always above the "Manage Embeddings" window
        self.add_person_window.lift()
        # Detailed: Bring the new window to the front.
        # Simple: Make sure it's on top.
        self.add_person_window.attributes("-topmost", True)
        # Detailed: Set the new window to always stay on top.
        # Simple: Keep it above other windows.
        self.add_person_window.bind("<FocusIn>", lambda e: self.add_person_window.lift())
        self.add_person_window.bind("<Map>", lambda e: self.add_person_window.lift())
        # Detailed: Bind events to ensure the window remains on top when it gains focus or is mapped.
        # Simple: Make sure it stays visible when interacted with.

        # Label and entry for person's name
        ttk.Label(self.add_person_window, text="Person's Name:").grid(row=0, column=0, padx=10, pady=10)
        # Detailed: Create a label for the person's name input and place it in the grid layout.
        # Simple: Show text asking for the person's name.
        self.name_entry = ttk.Entry(self.add_person_window)
        # Detailed: Create an entry widget for the user to input the person's name.
        # Simple: Create a box to type the name.
        self.name_entry.grid(row=0, column=1, padx=10, pady=10)
        # Detailed: Place the entry widget next to the label in the grid layout.

        # Button to add video
        self.video_path = None  # Detailed: Initialize the video path variable to store the selected video file path.
                              # Simple: Start with no video selected.
        ttk.Button(self.add_person_window, text="Add Video", command=self.select_video).grid(row=1, column=0, columnspan=2, pady=10)
        # Detailed: Create a button labeled "Add Video" that opens a file dialog to select a video, and place it in the grid layout.
        # Simple: Add a button to pick a video file.

        # Train button
        ttk.Button(self.add_person_window, text="Train", command=self.train_new_person).grid(row=2, column=0, columnspan=2, pady=10)
        # Detailed: Create a "Train" button that triggers the process to train and process the new person's video.
        # Simple: Add a button to start processing the video.

    def select_video(self):
        """Select a video file."""
        # Detailed: Open a file dialog to allow the user to select a video file with an .mp4 extension.
        # Simple: Let the user choose a video file.
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if self.video_path:
            # Detailed: If a video file is selected, show a message box with the file name.
            # Simple: If a video is chosen, tell the user which one.
            messagebox.showinfo("Success", f"Selected video: {os.path.basename(self.video_path)}")

    def train_new_person(self):
        """Train the system with the new person's details and start processing the video."""
        # Detailed: Get the person's name from the input field and validate that both a name and video have been provided.
        # Simple: Check that a name and video are given.
        person_name = self.name_entry.get()
        if not person_name:
            messagebox.showerror("Error", "Please enter the person's name.")
            return
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video.")
            return

        # Start processing in a separate thread
        threading.Thread(target=self.process_video, args=(person_name, self.video_path)).start()
        # Detailed: Start a new thread to process the video so that the GUI remains responsive.
        # Simple: Process the video in the background.
        self.add_person_window.destroy()
        # Detailed: Close the "Add New Person" window once processing starts.
        # Simple: Close the window after starting.

    def process_video(self, person_name, video_path):
        """Process the video to extract frames and create embeddings."""
        # Detailed: Create a folder for the person in the face database directory and ensure it exists.
        # Simple: Make a new folder for the person.
        person_folder = os.path.join(self.face_database_dir, person_name)
        os.makedirs(person_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        # Detailed: Open the video file using OpenCV.
        # Simple: Open the video.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Detailed: Retrieve the total number of frames in the video.
        # Simple: Find out how many frames the video has.

        # Show the loading dialog
        loading_dialog = LoadingDialog(self.root, total_frames)
        # Detailed: Create and display a loading dialog that shows progress during frame and embedding extraction.
        # Simple: Open a window that shows progress.

        # Extract frames from the video
        frame_count = 0
        while True:
            ret, frame = cap.read()
            # Detailed: Read a frame from the video; ret is a boolean indicating success, and frame is the image data.
            # Simple: Read a picture from the video.
            if not ret:
                break  # Detailed: If no frame is returned (end of video), exit the loop.
                       # Simple: Stop if there are no more frames.
            frame_path = os.path.join(person_folder, f"frame_{frame_count}.jpg")
            # Detailed: Construct a file path for the current frame.
            # Simple: Create a name for this picture.
            cv2.imwrite(frame_path, frame)
            # Detailed: Save the current frame as an image file.
            # Simple: Write the picture to the folder.
            frame_count += 1
            # Detailed: Increment the frame counter.
            # Simple: Count this frame.

            # Update frame extraction progress
            loading_dialog.update_frame_progress(frame_count)
            # Detailed: Update the loading dialog to reflect the number of frames processed.
            # Simple: Update the progress window with the new frame count.

        cap.release()
        # Detailed: Release the video file once frame extraction is complete.
        # Simple: Close the video file.

        # Create embeddings from the frames
        person_embeddings = []  # Detailed: Initialize an empty list to store embeddings for the current person.
                               # Simple: Make a list for face codes.
        embedding_count = 0  # Detailed: Initialize a counter for the number of embeddings created.
                           # Simple: Count how many face codes are made.
        for image_name in os.listdir(person_folder):
            # Detailed: Loop through each saved frame image in the person's folder.
            # Simple: For every picture in the folder:
            face_tensor = preprocess_image(os.path.join(person_folder, image_name))
            # Detailed: Preprocess the image to prepare it for embedding extraction.
            # Simple: Prepare the picture for the face model.
            if face_tensor is not None:
                embedding = self.model(face_tensor.to(self.device)).detach().cpu().numpy()
                # Detailed: Pass the preprocessed image through the face recognition model to obtain an embedding,
                # then detach, move it to the CPU, and convert it to a NumPy array.
                # Simple: Get the face code (number list) from the picture.
                person_embeddings.append(embedding)
                # Detailed: Append the computed embedding to the list.
                # Simple: Save this face code.
                embedding_count += 1
                # Detailed: Increment the embedding counter.
                # Simple: Count the face code.

                # Update embedding extraction progress
                loading_dialog.update_embedding_progress(embedding_count)
                # Detailed: Update the loading dialog with the current number of embeddings processed.
                # Simple: Update the progress window with the new face code count.

        # Save embeddings as .npz file
        if person_embeddings:
            # Detailed: If embeddings were generated, save them to a compressed .npz file.
            # Simple: If there are face codes, save them.
            npz_path = os.path.join(self.face_database_dir, f"{person_name}.npz")
            # Detailed: Construct the file path for saving the embeddings.
            # Simple: Create a file name for the face codes.
            np.savez(npz_path, embeddings=np.array(person_embeddings))
            # Detailed: Convert the list of embeddings to a NumPy array and save it using np.savez.
            # Simple: Save the face codes to a file.
            messagebox.showinfo("Success", f"Saved embeddings for {person_name} to {npz_path}")
            # Detailed: Show a confirmation message to the user that the embeddings were saved.
            # Simple: Tell the user the file was saved.

        # Clean up
        for image_name in os.listdir(person_folder):
            os.remove(os.path.join(person_folder, image_name))
            # Detailed: Remove all individual frame images from the person's folder after processing.
            # Simple: Delete all the temporary pictures.
        os.rmdir(person_folder)
        # Detailed: Remove the now-empty folder created for the person.
        # Simple: Delete the empty folder.

        # Close the loading dialog
        loading_dialog.close()
        # Detailed: Close and destroy the loading dialog window.
        # Simple: Close the progress window.

        # Reload .npz files in the main GUI
        self.root.after(0, self.load_npz_files)
        # Detailed: Schedule an immediate update of the file list in the main GUI.
        # Simple: Refresh the file list.

        # Reload embeddings in the pipeline
        self.pipeline.load_embeddings()
        # Detailed: Instruct the pipeline to reload the embeddings, ensuring that the new data is available.
        # Simple: Update the face codes in the system.

class LoadingDialog:
    def __init__(self, parent, total_frames):
        # Detailed: Initialize the LoadingDialog class which creates a pop-up window showing progress of frame and embedding extraction.
        # Simple: Start a progress window that shows how far along we are.
        self.parent = parent  # Detailed: Store the parent window reference.
                             # Simple: Save the main window.
        self.total_frames = total_frames  # Detailed: Store the total number of frames to be processed.
                                         # Simple: Remember the total frame count.
        self.current_frame = 0  # Detailed: Initialize the current frame counter.
                              # Simple: Start counting frames at zero.
        self.current_embedding = 0  # Detailed: Initialize the current embedding counter.
                                  # Simple: Start counting face codes at zero.

        # Create a new top-level window
        self.dialog = tk.Toplevel(parent)
        # Detailed: Create a new top-level window that will serve as the loading dialog.
        # Simple: Open a new window for progress.
        self.dialog.title("Processing...")
        # Detailed: Set the title of the dialog window.
        # Simple: Name the window "Processing..."
        self.dialog.geometry("300x150")
        # Detailed: Set the fixed size of the dialog window.
        # Simple: Make the window 300x150 pixels.
        self.dialog.resizable(False, False)
        # Detailed: Prevent the window from being resized.
        # Simple: Don't let the window change size.

        # Ensure this window is always on top of all others
        self.dialog.lift()
        # Detailed: Bring the dialog window to the front.
        # Simple: Keep it on top.
        self.dialog.attributes("-topmost", True)
        # Detailed: Set the dialog to always remain on top.
        # Simple: Make sure it stays above other windows.
        self.dialog.bind("<FocusIn>", lambda e: self.dialog.lift())
        self.dialog.bind("<Map>", lambda e: self.dialog.lift())
        # Detailed: Bind events to ensure the dialog remains on top when it gains focus or is mapped.
        # Simple: Keep it visible when interacted with.

        # Prevent interaction with other windows
        self.dialog.grab_set()
        # Detailed: Capture all events for the dialog window, preventing user interaction with other windows until this one is closed.
        # Simple: Stop you from clicking outside the progress window.

        # Add a label for frame extraction progress
        self.frame_label = ttk.Label(self.dialog, text="Extracting frames: 0/0")
        # Detailed: Create a label widget in the dialog to display the frame extraction progress.
        # Simple: Show text indicating how many frames have been processed.
        self.frame_label.pack(pady=10)
        # Detailed: Pack the label into the dialog with vertical padding.
        # Simple: Place the label with some space around it.

        # Add a label for embedding extraction progress
        self.embedding_label = ttk.Label(self.dialog, text="Extracting embeddings: 0/0")
        # Detailed: Create another label widget for embedding extraction progress.
        # Simple: Show text indicating how many face codes have been processed.
        self.embedding_label.pack(pady=10)
        # Detailed: Pack the embedding label into the dialog with padding.
        # Simple: Place it with some space.

        # Add a progress bar
        self.progress = ttk.Progressbar(self.dialog, orient="horizontal", length=250, mode="determinate")
        # Detailed: Create a horizontal progress bar widget with a fixed length in determinate mode to show progress percentage.
        # Simple: Add a bar that fills up as processing goes on.
        self.progress.pack(pady=10)
        # Detailed: Pack the progress bar into the dialog with vertical padding.
        # Simple: Place the progress bar with space around it.

    def update_frame_progress(self, current_frame):
        """Update the frame extraction progress."""
        # Detailed: Update the current frame count and refresh the label text to display the updated progress.
        # Simple: Change the frame count and update the text.
        self.current_frame = current_frame
        self.frame_label.config(text=f"Extracting frames: {self.current_frame}/{self.total_frames}")
        self.dialog.update_idletasks()  # Detailed: Force the dialog to process any pending updates to ensure the progress display is current.
                                      # Simple: Make sure the window updates immediately.

    def update_embedding_progress(self, current_embedding):
        """Update the embedding extraction progress."""
        # Detailed: Update the current embedding count and adjust the progress bar value and label to reflect progress.
        # Simple: Change the face code count and update the progress bar.
        self.current_embedding = current_embedding
        self.embedding_label.config(text=f"Extracting embeddings: {self.current_embedding}/{self.total_frames}")
        self.progress["value"] = (self.current_embedding / self.total_frames) * 100
        self.dialog.update_idletasks()  # Detailed: Force the dialog to process any pending updates.
                                      # Simple: Refresh the window so it shows the new progress.

    def close(self):
        """Close the loading dialog."""
        # Detailed: Release the grab and destroy the dialog window to allow interaction with other windows.
        # Simple: Let you click other windows again and close the progress window.
        self.dialog.grab_release()
        self.dialog.destroy()
