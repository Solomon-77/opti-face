import sys
import os
import shutil
import numpy as np
import torch
import cv2 # Added for video processing
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QSizePolicy,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QCursor, QIcon, QPixmap
from src.backend.inference import CameraWidget
# Import the necessary function and utility from backend
from src.backend.prepare_embeddings import generate_and_save_embeddings
from src.backend.utils.face_utils import load_face_recognition_model

# --- Application Setup ---
app = QApplication(sys.argv)

# Change cursor upon hover
cursor_pointer = QCursor(Qt.CursorShape.PointingHandCursor)

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Admin window
class AdminWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Admin Screen")
        self.resize(900, 600)
        self.is_feed_running = False # Track feed state
        self.selected_image_paths = [] # To store paths of images selected for upload
        self.selected_video_path = None # To store path of selected video
        self.face_database_dir = './src/backend/face_database/' # Define database path

        # Load the face recognition model once for training tasks
        # Handle potential errors during model loading
        try:
            # Use the same model path as inference for consistency
            model_path = "src/backend/checkpoints/edgeface_s_gamma_05.pt"
            self.training_model, self.training_device = load_face_recognition_model(model_path=model_path)
            print("Training model loaded successfully.")
        except Exception as e:
            print(f"Error loading training model: {e}")
            QMessageBox.critical(self, "Model Load Error", f"Failed to load the face recognition model for training: {e}")
            # Optionally disable training features if model fails to load
            self.training_model, self.training_device = None, None


        # Admin window layout
        admin_layout = QHBoxLayout(self)
        admin_layout.setContentsMargins(0, 0, 0, 0)
        admin_layout.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        sidebar_layout.setSpacing(10)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        sidebar.setObjectName("sidebar")

        # Dashboard button
        self.dashboard_button = QPushButton("Dashboard")
        self.dashboard_button.setIconSize(QSize(18, 18))
        self.dashboard_button.setCursor(cursor_pointer)
        self.dashboard_button.setProperty("class", "sidebar-buttons")
        self.dashboard_button.setCheckable(True)
        self.dashboard_button.setChecked(True)
        sidebar_layout.addWidget(self.dashboard_button)

        # Train button
        self.train_button = QPushButton("Train")
        self.train_button.setIconSize(QSize(18, 18))
        self.train_button.setCursor(cursor_pointer)
        self.train_button.setProperty("class", "sidebar-buttons")
        self.train_button.setCheckable(True)
        sidebar_layout.addWidget(self.train_button)

        # Settings button
        self.settings_button = QPushButton("Settings")
        self.settings_button.setIconSize(QSize(18, 18))
        self.settings_button.setCursor(cursor_pointer)
        self.settings_button.setProperty("class", "sidebar-buttons")
        self.settings_button.setCheckable(True)
        sidebar_layout.addWidget(self.settings_button)

        # Stacked widget
        self.contentStack = QStackedWidget()

        # --- Dashboard page ---
        dashboard_page = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_page)
        dashboard_layout.setContentsMargins(15, 15, 15, 15)
        dashboard_layout.setSpacing(15)

        # Live camera card
        camera_card = QWidget()
        camera_card_layout = QVBoxLayout(camera_card)
        camera_card_layout.setContentsMargins(15, 15, 15, 15)
        camera_card_layout.setSpacing(15)

        # Top row for label and button
        camera_top_layout = QHBoxLayout()
        camera_label = QLabel("Live Face Recognition Feed")
        camera_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        camera_label.setObjectName("camera-label")

        self.toggle_feed_button = QPushButton("Start Feed")
        self.toggle_feed_button.setCursor(cursor_pointer)
        self.toggle_feed_button.setFixedWidth(100)
        self.toggle_feed_button.clicked.connect(self.toggle_camera_feed)
        self.toggle_feed_button.setStyleSheet("""        
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        camera_top_layout.addWidget(camera_label)
        camera_top_layout.addStretch() # Push button to the right
        camera_top_layout.addWidget(self.toggle_feed_button)

        self.camera_widget = CameraWidget() # Store instance for access
        camera_card_layout.addLayout(camera_top_layout) # Add the top row layout
        camera_card_layout.addWidget(self.camera_widget)
        camera_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")
        # Allow camera card to expand vertically
        camera_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


        threshold_card = QWidget()
        # Use QHBoxLayout for the card itself to arrange items horizontally
        threshold_card_layout = QHBoxLayout(threshold_card)
        threshold_card_layout.setContentsMargins(15, 10, 15, 10) # Adjust margins if needed
        threshold_card_layout.setSpacing(10) # Add spacing between elements

        threshold_label = QLabel("Minimum Face Recognition Threshold:") # Changed label text slightly
        threshold_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) # Align left

        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("0.0-1.0") # Shortened placeholder
        self.threshold_input.setText("0.6")  # Default value
        self.threshold_input.setFixedWidth(80) # Adjust width as needed

        self.apply_threshold = QPushButton("Apply")
        self.apply_threshold.setCursor(cursor_pointer)
        self.apply_threshold.setFixedWidth(80) # Adjust width as needed
        self.apply_threshold.clicked.connect(self.update_threshold)
        # Add specific styling for the apply button
        self.apply_threshold.setStyleSheet("""        
            QPushButton {
                padding: 5px;
                background-color: #4CAF50; /* Green background */
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049; /* Darker green on hover */
            }
        """)

        # Add widgets directly to the horizontal card layout
        threshold_card_layout.addWidget(threshold_label)
        threshold_card_layout.addWidget(self.threshold_input)
        threshold_card_layout.addWidget(self.apply_threshold)
        threshold_card_layout.addStretch() # Push elements to the left

        threshold_card.setStyleSheet("""
            QWidget { /* Apply to the card itself */
                background-color: #2a2b2e;
                border-radius: 6px;
            }
            QLabel { /* Style label within the card */
                 font-size: 13px; /* Adjust as needed */
                 font-weight: 600;
            }
            QLineEdit { /* Style input within the card */
                padding: 5px;
                border: 1px solid #5f6368;
                border-radius: 4px;
                background-color: #3a3b3e; /* Slightly different background */
                color: white;
            }
            QLineEdit:focus {
                 border: 1px solid #8ab4f8;
            }
            /* Button styling is now handled inline above */
        """)
        threshold_card.setMaximumHeight(60) # Adjust max height if needed

        dashboard_layout.addWidget(camera_card)
        dashboard_layout.addWidget(threshold_card)
        # Removed dashboard_layout.addStretch() to allow camera_card to expand

        # --- Train page ---
        train_page = QWidget()
        train_layout = QVBoxLayout(train_page)
        train_layout.setContentsMargins(15, 15, 15, 15)
        train_layout.setSpacing(15)

        # --- Upload Card ---
        upload_card = QWidget()
        upload_card_layout = QVBoxLayout(upload_card)
        upload_card_layout.setContentsMargins(15, 15, 15, 15)
        upload_card_layout.setSpacing(10)
        upload_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")

        upload_label = QLabel("Add New Person to Database")
        upload_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        upload_card_layout.addWidget(upload_label)

        # File selection row
        file_selection_layout = QHBoxLayout()
        self.upload_image_button = QPushButton("Select Images")
        self.upload_image_button.setCursor(cursor_pointer)
        self.upload_image_button.clicked.connect(self.select_images)

        self.upload_video_button = QPushButton("Select Video") # Added video button
        self.upload_video_button.setCursor(cursor_pointer)
        self.upload_video_button.clicked.connect(self.select_video) # Connect signal

        file_selection_layout.addWidget(self.upload_image_button)
        file_selection_layout.addWidget(self.upload_video_button) # Add video button to layout

        # Label to show selected file count or video name
        self.selected_files_label = QLabel("No files selected.")
        self.selected_files_label.setStyleSheet("font-size: 11px; color: #aaa;") # Basic styling
        file_selection_layout.addWidget(self.selected_files_label)
        file_selection_layout.addStretch() # Push label and button

        upload_card_layout.addLayout(file_selection_layout)

        # Name input row
        name_input_layout = QHBoxLayout()
        name_label = QLabel("Person's Name:")
        self.person_name_input = QLineEdit()
        self.person_name_input.setPlaceholderText("Enter name")
        # Add styling as needed
        name_input_layout.addWidget(name_label)
        name_input_layout.addWidget(self.person_name_input)
        upload_card_layout.addLayout(name_input_layout)

        # Add Person button
        self.add_person_button = QPushButton("Add Person and Train") # Changed text
        self.add_person_button.setCursor(cursor_pointer)
        self.add_person_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; font-weight: bold;
                border: none; border-radius: 4px; padding: 8px; margin-top: 10px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.add_person_button.clicked.connect(self.add_person_action) # Connect signal
        upload_card_layout.addWidget(self.add_person_button)

        # Status label for training feedback
        self.train_status_label = QLabel("")
        self.train_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.train_status_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        upload_card_layout.addWidget(self.train_status_label)


        upload_card_layout.addStretch() # Push content upwards
        upload_card.setMinimumHeight(250) # Adjust height as needed

        # --- Training Status Card (Optional - can be merged or removed) ---
        train_info_card = QWidget()
        train_info_card_layout = QVBoxLayout(train_info_card)
        train_info_label = QLabel("Training Status") # Changed label
        train_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Add more widgets here later to show progress/status
        train_info_card_layout.addWidget(train_info_label)
        train_info_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")
        train_info_card.setMaximumHeight(150) # Adjust height

        train_layout.addWidget(upload_card) # Add the upload card
        # train_layout.addWidget(train_info_card) # Keep or remove
        train_layout.addStretch()

        # Settings page
        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_label = QLabel("This is settings.")
        settings_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        settings_layout.addWidget(settings_label)

        # Add pages to stack
        self.contentStack.addWidget(dashboard_page)
        self.contentStack.addWidget(train_page)
        self.contentStack.addWidget(settings_page)

        admin_layout.addWidget(sidebar)
        admin_layout.addWidget(self.contentStack)

        self.dashboard_button.clicked.connect(lambda: self._update_button_states(0, self.dashboard_button))
        self.train_button.clicked.connect(lambda: self._update_button_states(1, self.train_button))
        self.settings_button.clicked.connect(lambda: self._update_button_states(2, self.settings_button))

        # Styling
        self.setStyleSheet(
            """
            #sidebar {
                background-color: #2a2b2e;
            }
            *[class=sidebar-buttons] {
                font-size: 16px;
                padding: 10px;
                border-radius: 6px;
                background-color: None;
                text-align: left;
            }
            *[class=sidebar-buttons]:checked {
                background-color: white;
                color: black;
            }
            *[class=sidebar-buttons]:hover:!checked {
                background-color: #3a3b3e;
            }
            *[class=start-button] {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
            *[class=start-button]:hover {
                background-color: #45a049;
            }
            *[class=stop-button] {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 8px;
            }
            *[class=stop-button]:hover {
                background-color: #da190b;
            }
            #camera-label {
                font-size: 14px;
                font-weight: 600;
            }
            """
        )

        self._update_icon_color(self.dashboard_button)
        self._update_icon_color(self.train_button)
        self._update_icon_color(self.settings_button)

    def select_images(self):
        """Opens a file dialog to select multiple image files."""
        # Define supported image file extensions
        image_extensions = "*.png *.jpg *.jpeg *.bmp *.tiff"
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images for Training",
            "", # Start directory (optional)
            f"Image Files ({image_extensions});;All Files (*)"
        )
        if files:
            self.selected_image_paths = files
            self.selected_video_path = None # Clear video selection if images are selected
            self.selected_files_label.setText(f"{len(files)} image(s) selected.")
            print(f"Selected images: {files}")
        else:
            # If selection is cancelled or empty, reset
            # self.selected_image_paths = [] # Keep previous selection? Or clear? Let's clear.
            # self.selected_files_label.setText("No files selected.")
            print("No images selected.")

    def select_video(self):
        """Opens a file dialog to select a single video file."""
        # Define supported video file extensions
        video_extensions = "*.mp4 *.avi *.mov *.mkv"
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video for Training",
            "", # Start directory (optional)
            f"Video Files ({video_extensions});;All Files (*)"
        )
        if file:
            self.selected_video_path = file
            self.selected_image_paths = [] # Clear image selection if video is selected
            self.selected_files_label.setText(f"Video: {os.path.basename(file)}")
            print(f"Selected video: {file}")
        else:
            print("No video selected.")


    def add_person_action(self):
        """Handles adding a new person, processing video or images, and triggering embedding generation."""
        person_name = self.person_name_input.text().strip()
        use_video = self.selected_video_path is not None
        use_images = bool(self.selected_image_paths)

        # --- Input Validation ---
        if not person_name:
            QMessageBox.warning(self, "Input Error", "Please enter the person's name.")
            self.train_status_label.setText("Error: Person's name is required.")
            self.train_status_label.setStyleSheet("color: #f28b82;") # Error color
            return
        if not use_video and not use_images:
            QMessageBox.warning(self, "Input Error", "Please select images or a video.")
            self.train_status_label.setText("Error: No images or video selected.")
            self.train_status_label.setStyleSheet("color: #f28b82;") # Error color
            return
        # Cannot select both video and images for one person addition
        if use_video and use_images:
             QMessageBox.warning(self, "Input Error", "Please select either images OR a video, not both.")
             self.train_status_label.setText("Error: Select images OR video.")
             self.train_status_label.setStyleSheet("color: #f28b82;")
             # Clear selections to force user re-selection
             self.selected_image_paths = []
             self.selected_video_path = None
             self.selected_files_label.setText("No files selected.")
             return
        if self.training_model is None or self.training_device is None:
             QMessageBox.critical(self, "Error", "Training model not loaded. Cannot add person.")
             self.train_status_label.setText("Error: Training model failed to load.")
             self.train_status_label.setStyleSheet("color: #f28b82;")
             return

        # --- Directory and File Handling ---
        person_folder = os.path.join(self.face_database_dir, person_name)
        extracted_frames_count = 0
        try:
            os.makedirs(person_folder, exist_ok=True) # Create directory if it doesn't exist
            print(f"Ensured directory exists: {person_folder}")

            if use_video:
                # --- Video Frame Extraction ---
                self.train_status_label.setText("Extracting frames from video...")
                self.train_status_label.setStyleSheet("color: #8ab4f8;")
                QApplication.processEvents() # Update UI

                cap = cv2.VideoCapture(self.selected_video_path)
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file: {self.selected_video_path}")

                frame_count = 0
                saved_frame_count = 0
                frame_interval = 5

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break # End of video

                    if frame_count % frame_interval == 0:
                        # Save frame as JPEG
                        frame_filename = f"{person_name}_frame_{saved_frame_count:04d}.jpg"
                        frame_path = os.path.join(person_folder, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        saved_frame_count += 1

                    frame_count += 1
                    # Optional: Update status label periodically during extraction
                    if frame_count % 100 == 0:
                         self.train_status_label.setText(f"Extracting... (Frame {frame_count})")
                         QApplication.processEvents()


                cap.release()
                extracted_frames_count = saved_frame_count
                print(f"Extracted {extracted_frames_count} frames from video.")

                if extracted_frames_count == 0:
                     QMessageBox.warning(self, "Video Error", "Could not extract any frames from the video.")
                     self.train_status_label.setText("Error: Failed to extract frames.")
                     self.train_status_label.setStyleSheet("color: #f28b82;")
                     # Optional cleanup
                     # if not os.listdir(person_folder): os.rmdir(person_folder)
                     return

                self.train_status_label.setText(f"Extracted {extracted_frames_count} frames. Starting training...")
                self.train_status_label.setStyleSheet("color: #8ab4f8;")
                QApplication.processEvents() # Update UI

            elif use_images:
                # --- Copy Image Files ---
                # Copy selected images to the person's folder
                copied_count = 0
                for img_path in self.selected_image_paths:
                    try:
                        # Generate a safe filename (e.g., using basename)
                        dest_filename = os.path.basename(img_path)
                        dest_path = os.path.join(person_folder, dest_filename)
                        # Avoid overwriting existing files with the same name in the folder?
                        # For simplicity, we'll overwrite now. Add checks if needed.
                        shutil.copy(img_path, dest_path)
                        print(f"Copied {img_path} to {dest_path}")
                        copied_count += 1
                    except Exception as copy_err:
                        print(f"Error copying file {img_path}: {copy_err}")
                        # Decide whether to continue or stop on copy error

                if copied_count == 0:
                     QMessageBox.warning(self, "File Error", "Could not copy any of the selected images.")
                     self.train_status_label.setText("Error: Failed to copy images.")
                     self.train_status_label.setStyleSheet("color: #f28b82;")
                     # Clean up potentially empty folder?
                     # if not os.listdir(person_folder): os.rmdir(person_folder) # Optional cleanup
                     return

                self.train_status_label.setText(f"Copied {copied_count} images. Starting training...")
                self.train_status_label.setStyleSheet("color: #8ab4f8;") # Processing color
                QApplication.processEvents() # Update UI

        except IOError as vid_err: # Specific error for video opening
             QMessageBox.critical(self, "Video Error", f"Error opening video file: {vid_err}")
             self.train_status_label.setText(f"Error: {vid_err}")
             self.train_status_label.setStyleSheet("color: #f28b82;")
             return
        except Exception as dir_err:
            QMessageBox.critical(self, "Directory Error", f"Could not create directory for {person_name}: {dir_err}")
            self.train_status_label.setText(f"Error creating directory: {dir_err}")
            self.train_status_label.setStyleSheet("color: #f28b82;")
            return

        # --- Embedding Generation (uses images in person_folder) ---
        try:
            # Call the function from prepare_embeddings
            npz_path = generate_and_save_embeddings(
                person_name=person_name,
                person_folder=person_folder,
                output_dir=self.face_database_dir, # Save .npz in the main db dir
                model=self.training_model,
                device=self.training_device
            )

            if npz_path:
                self.train_status_label.setText(f"Training complete for {person_name}. Embeddings saved.")
                self.train_status_label.setStyleSheet("color: #4CAF50;") # Success color

                # --- Update Live Recognition Pipeline ---
                if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
                    self.camera_widget.pipeline.load_new_person(person_name, npz_path)
                    print(f"Requested pipeline to load new embeddings for {person_name}")
                else:
                     print("Warning: CameraWidget pipeline not available to load new embeddings.")
                     QMessageBox.information(self, "Info", f"Embeddings for {person_name} saved. Restart the application or feed to activate recognition for the new person if the feed wasn't running.")


                # --- Clear Inputs ---
                self.person_name_input.clear()
                self.selected_image_paths = []
                self.selected_video_path = None # Clear video path
                self.selected_files_label.setText("No files selected.")

            else:
                # Handle case where embedding generation failed (e.g., no faces found)
                QMessageBox.warning(self, "Training Warning", f"Could not generate embeddings for {person_name}. Check images/frames and logs.")
                self.train_status_label.setText(f"Warning: No embeddings generated for {person_name}.")
                self.train_status_label.setStyleSheet("color: #fbbc04;") # Warning color


        except Exception as train_err:
            QMessageBox.critical(self, "Training Error", f"An error occurred during embedding generation: {train_err}")
            self.train_status_label.setText(f"Error during training: {train_err}")
            self.train_status_label.setStyleSheet("color: #f28b82;")
            print(f"Training error for {person_name}: {train_err}")


    def toggle_camera_feed(self):
        if self.is_feed_running:
            self.camera_widget.stop_feed()
            self.toggle_feed_button.setText("Start Feed")
            self.toggle_feed_button.setStyleSheet("""        
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-radius: 4px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.is_feed_running = False
        else:
            self.camera_widget.start_feed()
            self.toggle_feed_button.setText("Stop Feed")
            self.toggle_feed_button.setStyleSheet("""        
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    font-weight: bold;
                    border: none;
                    border-radius: 4px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
            """)
            self.is_feed_running = True

    def _update_button_states(self, index, clicked_button):
        # Uncheck all buttons first
        self.dashboard_button.setChecked(False)
        self.train_button.setChecked(False)
        self.settings_button.setChecked(False)

        # Check the clicked button
        clicked_button.setChecked(True)

        # Switch to the correct page in the stack
        self.contentStack.setCurrentIndex(index)

        self._update_icon_color(self.dashboard_button)
        self._update_icon_color(self.train_button)
        self._update_icon_color(self.settings_button)

    def _update_icon_color(self, button):
        icon_base_name = ""
        icon_folder = "src/gui/icons"

        if button == self.dashboard_button:
            icon_base_name = 'dashboard'
        elif button == self.train_button:
            icon_base_name = 'camera'
        elif button == self.settings_button:
            icon_base_name = 'settings'

        if icon_base_name:
            if button.isChecked():
                icon_path = f"{icon_folder}/{icon_base_name}_black.svg"
            else:
                icon_path = f"{icon_folder}/{icon_base_name}_white.svg"

            try:
                icon = QIcon(QPixmap(icon_path))
                if not icon.isNull():
                    button.setIcon(icon)
                else:
                    print(f"Warning: Could not load icon from {icon_path}")
                    button.setIcon(QIcon()) # Set an empty icon or a default one
            except Exception as e:
                print(f"Error loading icon {icon_path}: {e}")
                button.setIcon(QIcon()) # Set an empty icon on error

    def update_threshold(self):
        try:
            threshold = float(self.threshold_input.text())
            if 0 <= threshold <= 1:
                # Ensure camera_widget and pipeline exist before setting threshold
                if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
                    self.camera_widget.set_recognition_threshold(threshold)
                    print(f"Recognition threshold set to: {threshold}")
                else:
                    print("Warning: CameraWidget or pipeline not initialized. Cannot set threshold.")
                    # Optionally store the value and apply it when the pipeline starts
            else:
                QMessageBox.warning(self, "Invalid Input", "Threshold must be between 0.0 and 1.0.")
                print("Threshold must be between 0 and 1")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for the threshold.")
            print("Invalid threshold value")
        except Exception as e:
             print(f"Error setting threshold: {e}")


def login():
    if username.text() == ADMIN_USERNAME and password.text() == ADMIN_PASSWORD:
        global admin_window
        admin_window = AdminWindow()
        admin_window.show()
        window.close()
    else:
        error_label.setText("Invalid username or password.")
        error_label.show()
        password.clear()
        QTimer.singleShot(5000, error_label.hide)

# --- Main Window Setup ---
window = QWidget()
window.setWindowTitle("OptiFace")
window.resize(450, 600)

# Title label
title = QLabel("Face Recognition System")
title.setObjectName("title")
title.setAlignment(Qt.AlignmentFlag.AlignCenter)
title.setWordWrap(True)

# Username input
username = QLineEdit()
username.setProperty("class", "user-pass")
username.setPlaceholderText("Username")

# Password input
password = QLineEdit()
password.setProperty("class", "user-pass")
password.setPlaceholderText("Password")
password.setEchoMode(QLineEdit.EchoMode.Password)
password.returnPressed.connect(login)

# Error label
error_label = QLabel("")
error_label.setObjectName("error-label")
error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
error_label.hide()

# Login button
login_button = QPushButton("Login")
login_button.setObjectName("login-button")
login_button.setCursor(cursor_pointer)
login_button.clicked.connect(login)

# Layout
main_layout = QVBoxLayout(window)
main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

login_form = QWidget()
login_form.setMaximumWidth(350)

login_form_layout = QVBoxLayout(login_form)
login_form_layout.setSpacing(20)
login_form_layout.setContentsMargins(30, 0, 30, 0)

login_form_layout.addWidget(title)
login_form_layout.addWidget(username)
login_form_layout.addWidget(password)
login_form_layout.addWidget(error_label)
login_form_layout.addWidget(login_button)

main_layout.addWidget(login_form)

# Styling
app.setStyleSheet(
    """
    QWidget {
        background-color: #202124;
        font-family: "Consolas", "Courier New", monospace;
        color: white;
    }
    #title {
        font-size: 25px;
        font-weight: 700;
        color: white;
        margin-bottom: 20px;
    }
    *[class=user-pass] {
        color: white;
        font-size: 15px;
        padding: 10px 8px;
        border: 1px solid #5f6368;
        border-radius: 6px;
    }
    *[class=user-pass]:focus {
        border: 1px solid #8ab4f8;
    }
    #login-button {
        color: black;
        font-size: 15px;
        font-weight: 600;
        padding: 10px;
        background-color: white;
        border-radius: 6px;
    }
    #login-button:hover {
        background-color: #dedede;
    }
    #error-label {
        color: #f28b82;
        font-size: 13px;
        font-weight: bold;
    }
    """
)

# --- Show Window and Run Application ---
window.show()
sys.exit(app.exec())