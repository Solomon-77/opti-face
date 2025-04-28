import sys
import os
import shutil
import numpy as np
import torch
import cv2 # Added for video processing
from datetime import datetime # Added for timestamp formatting
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
    QScrollArea, # Added
    QFrame,      # Added for separators
    QTableWidget, # Added for records
    QTableWidgetItem, # Added for records table items
    QHeaderView, # Added for table header styling
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

# --- Person Entry Widget ---
class PersonEntryWidget(QWidget):
    """Widget for entering details for a single person."""
    def __init__(self, parent_admin_window):
        super().__init__()
        self.parent_admin_window = parent_admin_window
        self.selected_image_paths = []
        self.selected_video_path = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 10, 10)  # Changed left margin to 0
        layout.setSpacing(10) # Keep existing spacing between main elements
        self.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")

        # File selection row
        file_selection_layout = QHBoxLayout()
        file_selection_layout.setContentsMargins(0, 0, 0, 0)  # Added this line
        self.upload_image_button = QPushButton("Select Image")
        self.upload_image_button.setCursor(cursor_pointer)
        self.upload_image_button.clicked.connect(lambda: self.parent_admin_window.select_images(self))
        # Add styling for the image button
        self.upload_image_button.setStyleSheet("""        
            QPushButton {
                background-color: #5f6368; color: white;
                border: none; border-radius: 4px; padding: 6px 10px;
            }
            QPushButton:hover { background-color: #707478; }
        """)

        self.upload_video_button = QPushButton("Select Video")
        self.upload_video_button.setCursor(cursor_pointer)
        self.upload_video_button.clicked.connect(lambda: self.parent_admin_window.select_video(self))
        # Add styling for the video button
        self.upload_video_button.setStyleSheet("""        
            QPushButton {
                background-color: #5f6368; color: white;
                border: none; border-radius: 4px; padding: 6px 10px;
            }
            QPushButton:hover { background-color: #707478; }
        """)

        self.selected_files_label = QLabel("No files selected.")
        # Remove padding and explicitly set background to none
        self.selected_files_label.setStyleSheet("font-size: 11px; color: #aaa; background: none;")

        # Add remove button to file selection row
        self.remove_button = QPushButton("Remove")  # Using × symbol for delete
        self.remove_button.setCursor(cursor_pointer)
        self.remove_button.setStyleSheet("""        
            QPushButton {
                background-color: #f44336; 
                color: white; 
                font-weight: bold;
                border: none; 
                border-radius: 4px; 
                padding: 5px 10px;
            }
            QPushButton:hover { background-color: #da190b; }
        """)
        self.remove_button.clicked.connect(lambda: self.parent_admin_window.remove_person_entry_widget(self))

        file_selection_layout.addWidget(self.upload_image_button)
        file_selection_layout.addWidget(self.upload_video_button)
        file_selection_layout.addWidget(self.selected_files_label)
        file_selection_layout.addStretch()
        file_selection_layout.addWidget(self.remove_button)
        layout.addLayout(file_selection_layout)

        # Add spacing after file selection row
        layout.addSpacing(10) # Adjust the value (10) as needed

        # Name input row
        name_input_layout = QHBoxLayout()
        name_label = QLabel("Person's Name:")
        # Make label bold and remove background
        name_label.setStyleSheet("background: none; font-weight: bold;")
        self.person_name_input = QLineEdit()
        self.person_name_input.setPlaceholderText("Enter name")
        # Add styling similar to previous input fields if needed
        self.person_name_input.setStyleSheet("""        
            QLineEdit {
                padding: 5px; border: 1px solid #5f6368; border-radius: 4px;
                background-color: #3a3b3e; color: white;
            }
            QLineEdit:focus { border: 1px solid #8ab4f8; }
        """)
        name_input_layout.addWidget(name_label)
        name_input_layout.addWidget(self.person_name_input)
        layout.addLayout(name_input_layout)

        # Add spacing after name input row
        layout.addSpacing(10) # Adjust the value (10) as needed

        # Status label
        self.train_status_label = QLabel("")
        self.train_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.train_status_label.setStyleSheet("font-size: 12px; font-weight: bold; min-height: 18px;") # Added min-height
        layout.addWidget(self.train_status_label)

# Admin window
class AdminWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Admin Screen")
        self.resize(900, 600)
        self.is_feed_running = False # Track feed state
        self.face_database_dir = './src/backend/face_database/' # Define database path
        self.person_entry_widgets = [] # List to hold PersonEntryWidget instances

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

        # Database button (New)
        self.database_button = QPushButton("Face DB")
        self.database_button.setIconSize(QSize(18, 18))
        self.database_button.setCursor(cursor_pointer)
        self.database_button.setProperty("class", "sidebar-buttons")
        self.database_button.setCheckable(True)
        sidebar_layout.addWidget(self.database_button)

        # Records button (New)
        self.records_button = QPushButton("Records")
        self.records_button.setIconSize(QSize(18, 18))
        self.records_button.setCursor(cursor_pointer)
        self.records_button.setProperty("class", "sidebar-buttons")
        self.records_button.setCheckable(True)
        sidebar_layout.addWidget(self.records_button)

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
        train_main_layout = QVBoxLayout(train_page) # Main layout for the page
        train_main_layout.setContentsMargins(25, 25, 25, 25)
        train_main_layout.setSpacing(15)

        # Button to add new person entries
        self.add_person_entry_button = QPushButton("+ Add Person")
        self.add_person_entry_button.setCursor(cursor_pointer)
        self.add_person_entry_button.setStyleSheet("""        
            QPushButton {
                background-color: #8ab4f8; color: black; font-weight: bold;
                border: none; border-radius: 4px; padding: 8px; max-width: 150px;
            }
            QPushButton:hover { background-color: #9ac0f9; }
        """)
        self.add_person_entry_button.clicked.connect(self.add_person_entry_widget)
        train_main_layout.addWidget(self.add_person_entry_button, alignment=Qt.AlignmentFlag.AlignLeft) # Add button to main layout

        # Scroll Area for Person Entries
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #202124; }") # Style scroll area

        # Container widget inside Scroll Area
        self.scroll_content_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_content_widget)

        # Layout for the container widget (holds PersonEntryWidgets)
        self.person_entries_layout = QVBoxLayout(self.scroll_content_widget)
        self.person_entries_layout.setContentsMargins(0, 0, 0, 0) # No margins for the inner layout
        self.person_entries_layout.setSpacing(10) # Spacing between person entries
        self.person_entries_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align entries to the top

        train_main_layout.addWidget(self.scroll_area) # Add scroll area to main layout

        # Add the "Train All" button below the scroll area
        self.train_all_button = QPushButton("Train All Added Persons")
        self.train_all_button.setCursor(cursor_pointer)
        self.train_all_button.setStyleSheet("""        
            QPushButton {
                background-color: #4CAF50; color: white; font-weight: bold;
                border: none; border-radius: 4px; padding: 10px; margin-top: 10px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.train_all_button.clicked.connect(self.train_all_persons_action)
        self.train_all_button.setMaximumWidth(200)
        # Add button with center alignment
        train_main_layout.addWidget(self.train_all_button, alignment=Qt.AlignmentFlag.AlignCenter) # Center align

        # Add the first entry widget automatically
        self.add_person_entry_widget()

        # --- Face Database page (New) ---
        database_page = QWidget()
        database_layout = QVBoxLayout(database_page)
        database_layout.setContentsMargins(25, 25, 25, 25)
        database_layout.setSpacing(15)

        db_top_layout = QHBoxLayout()
        db_label = QLabel("Registered Faces")
        db_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.refresh_db_button = QPushButton("Refresh List")
        self.refresh_db_button.setCursor(cursor_pointer)
        self.refresh_db_button.clicked.connect(self.populate_database_list)
        self.refresh_db_button.setStyleSheet("""        
            QPushButton {
                background-color: #5f6368; color: white;
                border: none; border-radius: 4px; padding: 6px 10px; max-width: 120px;
            }
            QPushButton:hover { background-color: #707478; }
        """)
        db_top_layout.addWidget(db_label)
        db_top_layout.addStretch()
        db_top_layout.addWidget(self.refresh_db_button)
        database_layout.addLayout(db_top_layout)


        self.db_scroll_area = QScrollArea()
        self.db_scroll_area.setWidgetResizable(True)
        self.db_scroll_area.setStyleSheet("QScrollArea { border: 1px solid #3a3b3e; background-color: #2a2b2e; border-radius: 6px; }") # Style scroll area

        self.db_scroll_content_widget = QWidget()
        self.db_scroll_area.setWidget(self.db_scroll_content_widget)

        self.database_list_layout = QVBoxLayout(self.db_scroll_content_widget)
        self.database_list_layout.setContentsMargins(10, 10, 10, 10) # Margins inside scroll area
        self.database_list_layout.setSpacing(8) # Spacing between entries
        self.database_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align entries to the top

        database_layout.addWidget(self.db_scroll_area) # Add scroll area to the database page layout

        # --- Records page (New) ---
        records_page = QWidget()
        records_layout = QVBoxLayout(records_page)
        records_layout.setContentsMargins(25, 25, 25, 25)
        records_layout.setSpacing(15)

        records_top_layout = QHBoxLayout()
        records_label = QLabel("Detection Records")
        records_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.refresh_records_button = QPushButton("Refresh Records")
        self.refresh_records_button.setCursor(cursor_pointer)
        self.refresh_records_button.clicked.connect(self.populate_records_table) # Connect refresh
        self.refresh_records_button.setStyleSheet("""
            QPushButton {
                background-color: #5f6368; color: white;
                border: none; border-radius: 4px; padding: 6px 10px; max-width: 150px;
            }
            QPushButton:hover { background-color: #707478; }
        """)
        records_top_layout.addWidget(records_label)
        records_top_layout.addStretch()
        records_top_layout.addWidget(self.refresh_records_button)
        records_layout.addLayout(records_top_layout)

        # Records Table
        self.records_table = QTableWidget()
        self.records_table.setColumnCount(4) # Name, Date, Time, Accuracy
        self.records_table.setHorizontalHeaderLabels(["Name", "Date", "Time", "Accuracy"])
        self.records_table.verticalHeader().setVisible(False) # Hide row numbers
        self.records_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers) # Read-only
        self.records_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows) # Select whole rows
        self.records_table.setAlternatingRowColors(True) # Zebra striping

        # Style the table and header
        self.records_table.setStyleSheet("""
            QTableWidget {
                background-color: #2a2b2e;
                border: 1px solid #3a3b3e;
                border-radius: 6px;
                gridline-color: #3a3b3e; /* Color of the grid lines */
                color: white; /* Text color */
            }
            QTableWidget::item {
                padding: 5px; /* Padding within cells */
            }
            QTableWidget::item:selected {
                background-color: #5f6368; /* Background color of selected row */
                color: white;
            }
             QHeaderView::section {
                background-color: #3a3b3e; /* Header background */
                color: white; /* Header text color */
                padding: 5px;
                border: none; /* No borders between header sections */
                border-bottom: 1px solid #5f6368; /* Bottom border for header */
                font-weight: bold;
            }
            QTableCornerButton::section { /* Style the top-left corner */
                 background-color: #3a3b3e;
                 border: none;
                 border-bottom: 1px solid #5f6368;
            }
            /* Style alternating rows */
             QTableView::item:alternate {
                 background-color: #313235; /* Slightly different background for alternate rows */
             }
        """)
        # Make columns resize nicely
        header = self.records_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch) # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) # Date
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents) # Time
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) # Accuracy

        records_layout.addWidget(self.records_table)

        # --- Settings page ---
        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_label = QLabel("This is settings.")
        settings_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        settings_layout.addWidget(settings_label)

        # Add pages to stack
        self.contentStack.addWidget(dashboard_page)
        self.contentStack.addWidget(train_page)
        self.contentStack.addWidget(database_page) # Add new database page
        self.contentStack.addWidget(records_page) # Add new records page
        self.contentStack.addWidget(settings_page)

        admin_layout.addWidget(sidebar)
        admin_layout.addWidget(self.contentStack)

        self.dashboard_button.clicked.connect(lambda: self._update_button_states(0, self.dashboard_button))
        self.train_button.clicked.connect(lambda: self._update_button_states(1, self.train_button))
        self.database_button.clicked.connect(lambda: self._update_button_states(2, self.database_button)) # Connect new button
        self.records_button.clicked.connect(lambda: self._update_button_states(3, self.records_button)) # Connect new records button
        self.settings_button.clicked.connect(lambda: self._update_button_states(4, self.settings_button)) # Adjust index

        # Initial population of database list when window is created
        self.populate_database_list()
        # Initial population of records table
        self.populate_records_table()

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
        self._update_icon_color(self.database_button) # Update icon for new button
        self._update_icon_color(self.records_button) # Update icon for records button
        self._update_icon_color(self.settings_button)

    def add_person_entry_widget(self):
        """Creates and adds a new PersonEntryWidget to the scroll area."""
        entry_widget = PersonEntryWidget(self)
        self.person_entries_layout.addWidget(entry_widget)
        self.person_entry_widgets.append(entry_widget)
        # Optional: Add a separator line if more than one widget exists
        # Insert separator before the last added widget
        if self.person_entries_layout.count() > 1: # Check if more than one widget exists *before* adding separator
             # Find the widget added just before this one
             previous_widget_index = self.person_entries_layout.count() - 2
             # Check if the item at that index is indeed a PersonEntryWidget (not already a separator)
             item = self.person_entries_layout.itemAt(previous_widget_index)
             if item and isinstance(item.widget(), PersonEntryWidget):
                 separator = QFrame()
                 separator.setFrameShape(QFrame.Shape.HLine)
                 separator.setFrameShadow(QFrame.Shadow.Sunken)
                 separator.setStyleSheet("border: 1px solid #3a3b3e;") # Basic styling
                 # Insert separator *before* the newly added widget
                 self.person_entries_layout.insertWidget(self.person_entries_layout.count() - 1, separator)


    def remove_person_entry_widget(self, target_widget: PersonEntryWidget):
        """Removes a specific PersonEntryWidget and its preceding separator."""
        if target_widget in self.person_entry_widgets:
            try:
                # Find the index of the target widget in the layout
                index = self.person_entries_layout.indexOf(target_widget)

                # Remove the widget itself
                self.person_entries_layout.removeWidget(target_widget)
                target_widget.deleteLater()
                self.person_entry_widgets.remove(target_widget)
                print(f"Removed person entry widget.")

                # Check if there's a separator *before* this widget (index > 0)
                if index > 0:
                    item_before = self.person_entries_layout.itemAt(index - 1)
                    if item_before and isinstance(item_before.widget(), QFrame):
                        separator_widget = item_before.widget()
                        self.person_entries_layout.removeWidget(separator_widget)
                        separator_widget.deleteLater()
                        print("Removed preceding separator.")
                # Check if there's a separator *after* this widget (if it was the last actual entry)
                # This handles removing the separator when the *last* entry is removed
                elif index == 0 and self.person_entries_layout.count() > 0: # If it was the first item and there's still something left
                     item_after = self.person_entries_layout.itemAt(0) # Check the new first item
                     if item_after and isinstance(item_after.widget(), QFrame):
                         separator_widget = item_after.widget()
                         self.person_entries_layout.removeWidget(separator_widget)
                         separator_widget.deleteLater()
                         print("Removed succeeding separator (was first item).")


            except Exception as e:
                print(f"Error removing widget: {e}")
                # Optionally show a message box
                QMessageBox.warning(self, "Error", f"Could not remove the entry: {e}")

    def select_images(self, target_widget: PersonEntryWidget):
        """Opens a file dialog to select multiple image files for a specific person entry."""
        # Define supported image file extensions
        image_extensions = "*.png *.jpg *.jpeg *.bmp *.tiff"
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images for Training",
            "", # Start directory (optional)
            f"Image Files ({image_extensions});;All Files (*)"
        )
        if files:
            target_widget.selected_image_paths = files
            target_widget.selected_video_path = None # Clear video selection
            target_widget.selected_files_label.setText(f"{len(files)} image(s) selected.")
            print(f"Selected images for widget: {files}")
        else:
            # Decide if clearing is desired on cancellation
            # target_widget.selected_image_paths = []
            # target_widget.selected_files_label.setText("No files selected.")
            print("No images selected for widget.")

    def select_video(self, target_widget: PersonEntryWidget):
        """Opens a file dialog to select a single video file for a specific person entry."""
        # Define supported video file extensions
        video_extensions = "*.mp4 *.avi *.mov *.mkv"
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video for Training",
            "", # Start directory (optional)
            f"Video Files ({video_extensions});;All Files (*)"
        )
        if file:
            target_widget.selected_video_path = file
            target_widget.selected_image_paths = [] # Clear image selection
            target_widget.selected_files_label.setText(f"Video: {os.path.basename(file)}")
            print(f"Selected video for widget: {file}")
        else:
            print("No video selected for widget.")

    def train_all_persons_action(self):
        """Processes and trains all valid person entries."""
        print("Starting training for all added persons...")
        if self.training_model is None or self.training_device is None:
             QMessageBox.critical(self, "Error", "Training model not loaded. Cannot train.")
             # Optionally set status on all widgets?
             return

        processed_count = 0
        error_count = 0
        widgets_to_clear = []

        for entry_widget in self.person_entry_widgets:
            person_name = entry_widget.person_name_input.text().strip()
            use_video = entry_widget.selected_video_path is not None
            use_images = bool(entry_widget.selected_image_paths)

            # Skip if no name or no files/video selected for this entry
            if not person_name or (not use_video and not use_images):
                # Optionally clear status or set a "Skipped" status
                # entry_widget.train_status_label.setText("Skipped (Missing info)")
                # entry_widget.train_status_label.setStyleSheet("color: #aaa;")
                continue # Move to the next widget

            # Process this single person entry
            npz_path = self._process_single_person(entry_widget)

            if npz_path:
                processed_count += 1
                widgets_to_clear.append(entry_widget) # Mark for clearing after loop
            else:
                error_count += 1
                # Status label is already set within _process_single_person on error/warning

        # --- Summary Message ---
        summary_message = f"Training finished. Processed: {processed_count}, Errors/Warnings: {error_count}."
        QMessageBox.information(self, "Training Complete", summary_message)
        print(summary_message)

        # --- Clear successful entries AFTER the loop ---
        # It's generally safer to modify UI elements (like clearing inputs)
        # after iterating and processing is fully done.
        # for widget in widgets_to_clear:
        #     widget.person_name_input.clear()
        #     widget.selected_image_paths = []
        #     widget.selected_video_path = None
        #     widget.selected_files_label.setText("No files selected.")
            # Keep the success status label visible

    def _process_single_person(self, target_widget: PersonEntryWidget):
        """Handles processing (validation, file handling, embedding) for one PersonEntryWidget. Returns npz_path on success, None on failure."""
        person_name = target_widget.person_name_input.text().strip()
        use_video = target_widget.selected_video_path is not None
        use_images = bool(target_widget.selected_image_paths)

        # --- Input Validation ---
        # Basic checks are done in train_all_persons_action before calling this
        # Add any more specific validation here if needed

        # --- Directory and File Handling ---
        person_folder = os.path.join(self.face_database_dir, person_name)
        extracted_frames_count = 0
        try:
            os.makedirs(person_folder, exist_ok=True)
            print(f"Processing entry for: {person_name}") # Log start

            if use_video:
                # --- Video Frame Extraction ---
                target_widget.train_status_label.setText("Extracting frames...")
                target_widget.train_status_label.setStyleSheet("color: #8ab4f8;")
                QApplication.processEvents()

                cap = cv2.VideoCapture(target_widget.selected_video_path)
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file: {target_widget.selected_video_path}")

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
                         target_widget.train_status_label.setText(f"Extracting... (Frame {frame_count})") # Update correct label
                         QApplication.processEvents()


                cap.release()
                extracted_frames_count = saved_frame_count
                print(f"Extracted {extracted_frames_count} frames from video.")

                if extracted_frames_count == 0:
                     target_widget.train_status_label.setText("Error: Failed to extract frames.")
                     target_widget.train_status_label.setStyleSheet("color: #f28b82;")
                     return None # Indicate failure

                target_widget.train_status_label.setText(f"Extracted {extracted_frames_count} frames. Training...") # Update correct label
                target_widget.train_status_label.setStyleSheet("color: #8ab4f8;")
                QApplication.processEvents() # Update UI

            elif use_images:
                # --- Copy Image Files ---
                target_widget.train_status_label.setText("Copying images...") # Update status
                target_widget.train_status_label.setStyleSheet("color: #8ab4f8;")
                QApplication.processEvents()
                copied_count = 0
                for img_path in target_widget.selected_image_paths: # Use target_widget's paths
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
                     target_widget.train_status_label.setText("Error: Failed to copy images.")
                     target_widget.train_status_label.setStyleSheet("color: #f28b82;")
                     return None # Indicate failure

                target_widget.train_status_label.setText(f"Copied {copied_count} images. Training...") # Update correct label
                target_widget.train_status_label.setStyleSheet("color: #8ab4f8;") # Processing color
                QApplication.processEvents() # Update UI

        except IOError as vid_err: # Specific error for video opening
             target_widget.train_status_label.setText(f"Error opening video: {vid_err}")
             target_widget.train_status_label.setStyleSheet("color: #f28b82;")
             return None
        except Exception as dir_err:
            target_widget.train_status_label.setText(f"Error creating directory: {dir_err}")
            target_widget.train_status_label.setStyleSheet("color: #f28b82;")
            return None

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
                target_widget.train_status_label.setText(f"Success: Embeddings saved.")
                target_widget.train_status_label.setStyleSheet("color: #4CAF50;") # Success color

                # --- Update Live Recognition Pipeline ---
                if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
                    self.camera_widget.pipeline.load_new_person(person_name, npz_path)
                    print(f"Requested pipeline to load new embeddings for {person_name}")
                else:
                     print(f"Warning: CameraWidget pipeline not available for {person_name}.")

                # --- Don't clear inputs here, handled after loop in train_all_persons_action ---
                return npz_path # Indicate success

            else:
                # Handle case where embedding generation failed (e.g., no faces found)
                target_widget.train_status_label.setText(f"Warning: No embeddings generated.")
                target_widget.train_status_label.setStyleSheet("color: #fbbc04;") # Warning color
                return None # Indicate failure/warning


        except Exception as train_err:
            target_widget.train_status_label.setText(f"Error during training: {train_err}")
            target_widget.train_status_label.setStyleSheet("color: #f28b82;")
            print(f"Training error for {person_name}: {train_err}")
            return None # Indicate failure

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
        self.database_button.setChecked(False) # Add database button
        self.records_button.setChecked(False) # Add records button
        self.settings_button.setChecked(False)

        # Check the clicked button
        clicked_button.setChecked(True)

        # Switch to the correct page in the stack
        self.contentStack.setCurrentIndex(index)

        # Update icons for all buttons
        self._update_icon_color(self.dashboard_button)
        self._update_icon_color(self.train_button)
        self._update_icon_color(self.database_button) # Add database button
        self._update_icon_color(self.records_button) # Add records button
        self._update_icon_color(self.settings_button)

        # Populate records if switching to the records page
        if index == 3: # Index of the records page
            self.populate_records_table()

    def _update_icon_color(self, button):
        icon_base_name = ""
        icon_folder = "src/gui/icons"

        if button == self.dashboard_button:
            icon_base_name = 'dashboard'
        elif button == self.train_button:
            icon_base_name = 'camera' # Consider renaming icon file if 'train' is better
        elif button == self.database_button: # Add database button case
            icon_base_name = 'database' # This line handles the database icon
        elif button == self.records_button: # Add records button case
            icon_base_name = 'records'
        elif button == self.settings_button:
            icon_base_name = 'settings'

        if icon_base_name:
            if button.isChecked():
                icon_path = f"{icon_folder}/{icon_base_name}_black.svg"
            else:
                icon_path = f"{icon_folder}/{icon_base_name}_white.svg"

            try:
                icon = QIcon(QPixmap(icon_path)) # Using QPixmap for SVG might need QtSvg module
                # Consider using QIcon(icon_path) directly if QtSvg is available and configured
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

    def populate_database_list(self):
        """Clears and repopulates the face database list view."""
        # Clear existing widgets in the layout
        while self.database_list_layout.count():
            item = self.database_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        print("Populating database list...")
        try:
            found_files = False
            for filename in sorted(os.listdir(self.face_database_dir)):
                if filename.endswith('.npz'):
                    found_files = True
                    person_name = os.path.splitext(filename)[0]
                    npz_path = os.path.join(self.face_database_dir, filename)

                    # Create a widget for each person entry
                    entry_widget = QWidget()
                    entry_layout = QHBoxLayout(entry_widget)
                    entry_layout.setContentsMargins(5, 5, 5, 5)
                    entry_widget.setStyleSheet("background-color: #3a3b3e; border-radius: 4px;") # Slightly different bg

                    name_label = QLabel(person_name)
                    name_label.setStyleSheet("font-size: 13px; font-weight: bold; background: none;")

                    delete_button = QPushButton("Delete")
                    delete_button.setCursor(cursor_pointer)
                    delete_button.setFixedWidth(80)
                    delete_button.setStyleSheet("""        
                        QPushButton {
                            background-color: #f44336; color: white; font-weight: bold;
                            border: none; border-radius: 4px; padding: 5px;
                        }
                        QPushButton:hover { background-color: #da190b; }
                    """)
                    # Use lambda to capture the current person_name for the slot
                    delete_button.clicked.connect(lambda checked=False, name=person_name: self.delete_person_action(name))

                    entry_layout.addWidget(name_label)
                    entry_layout.addStretch()
                    entry_layout.addWidget(delete_button)

                    self.database_list_layout.addWidget(entry_widget)

            if not found_files:
                 no_faces_label = QLabel("No registered faces found in the database.")
                 no_faces_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                 no_faces_label.setStyleSheet("color: #aaa; font-style: italic;")
                 self.database_list_layout.addWidget(no_faces_label)

            # Add a stretch at the end to push items to the top if the list is short
            self.database_list_layout.addStretch()

        except FileNotFoundError:
            print(f"Database directory not found: {self.face_database_dir}")
            error_label = QLabel(f"Error: Database directory not found at\n{self.face_database_dir}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setStyleSheet("color: #f28b82;")
            self.database_list_layout.addWidget(error_label)
        except Exception as e:
            print(f"Error populating database list: {e}")
            error_label = QLabel(f"An error occurred: {e}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setStyleSheet("color: #f28b82;")
            self.database_list_layout.addWidget(error_label)

    def delete_person_action(self, person_name):
        """Handles the deletion of a person from the database."""
        print(f"Attempting to delete person: {person_name}")

        reply = QMessageBox.question(self, 'Confirm Deletion',
                                     f"Are you sure you want to delete '{person_name}'?\n"
                                     f"This will remove their embeddings (.npz) and their image folder.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            npz_path = os.path.join(self.face_database_dir, f"{person_name}.npz")
            person_image_folder = os.path.join(self.face_database_dir, person_name)
            deleted_npz = False
            deleted_folder = False
            pipeline_updated = False

            # 1. Delete .npz file
            try:
                if os.path.exists(npz_path):
                    os.remove(npz_path)
                    print(f"Deleted .npz file: {npz_path}")
                    deleted_npz = True
                else:
                    print(f"Warning: .npz file not found, cannot delete: {npz_path}")
            except Exception as e:
                print(f"Error deleting .npz file {npz_path}: {e}")
                QMessageBox.warning(self, "Deletion Error", f"Could not delete the .npz file for {person_name}: {e}")

            # 2. Delete image folder (optional, but recommended)
            try:
                if os.path.isdir(person_image_folder):
                    shutil.rmtree(person_image_folder)
                    print(f"Deleted image folder: {person_image_folder}")
                    deleted_folder = True
                else:
                    print(f"Info: Image folder not found, skipping deletion: {person_image_folder}")
                    # If the folder doesn't exist, we can still consider it 'successful' in terms of cleanup
                    deleted_folder = True # Or set based on whether deletion was *attempted* and failed vs not needed
            except Exception as e:
                print(f"Error deleting image folder {person_image_folder}: {e}")
                QMessageBox.warning(self, "Deletion Error", f"Could not delete the image folder for {person_name}: {e}")

            # 3. Update live recognition pipeline
            try:
                if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
                    self.camera_widget.pipeline.remove_person(person_name)
                    print(f"Requested pipeline to remove embeddings for {person_name}")
                    pipeline_updated = True
                else:
                    print(f"Warning: CameraWidget pipeline not available to remove {person_name}.")
                    # Consider if this is an error or just info
            except Exception as e:
                 print(f"Error updating pipeline to remove {person_name}: {e}")
                 QMessageBox.warning(self, "Pipeline Error", f"Could not update the live recognition pipeline: {e}")


            # 4. Refresh the UI list
            self.populate_database_list()

            # Optional: Show success message if key parts succeeded
            if deleted_npz and deleted_folder and pipeline_updated:
                 QMessageBox.information(self, "Deletion Successful", f"Successfully removed {person_name} from the database and live recognition.")
            elif deleted_npz: # At least the core file was deleted
                 QMessageBox.information(self, "Deletion Partially Successful", f"Removed {person_name}'s .npz file. Check console for details on folder/pipeline updates.")


        else:
            print(f"Deletion cancelled for {person_name}.")

    def populate_records_table(self):
        """Fetches detection records and populates the records table."""
        print("Populating detection records table...")
        self.records_table.setRowCount(0) # Clear existing rows

        try:
            # Check if pipeline exists and has the method
            if hasattr(self.camera_widget, 'pipeline') and \
               self.camera_widget.pipeline and \
               hasattr(self.camera_widget.pipeline, 'get_detection_log'):

                detection_log = self.camera_widget.pipeline.get_detection_log()

                if not detection_log:
                    # Optional: Display a message in the table if no records
                    self.records_table.setRowCount(1)
                    no_records_item = QTableWidgetItem("No detection records available yet.")
                    no_records_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.records_table.setItem(0, 0, no_records_item)
                    # Span the message across all columns
                    self.records_table.setSpan(0, 0, 1, self.records_table.columnCount())
                    print("No detection records found in the pipeline.")
                    return

                self.records_table.setRowCount(len(detection_log))
                for row, record in enumerate(reversed(detection_log)): # Show newest first
                    name, timestamp, similarity = record

                    # Format timestamp
                    dt_object = datetime.fromtimestamp(timestamp)
                    date_str = dt_object.strftime("%Y-%m-%d")
                    time_str = dt_object.strftime("%H:%M:%S")
                    accuracy_str = f"{similarity:.2f}" # Format similarity

                    # Create table items
                    name_item = QTableWidgetItem(name)
                    date_item = QTableWidgetItem(date_str)
                    time_item = QTableWidgetItem(time_str)
                    accuracy_item = QTableWidgetItem(accuracy_str)

                    # Center align time and accuracy
                    time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    accuracy_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                    # Add items to the table row
                    self.records_table.setItem(row, 0, name_item)
                    self.records_table.setItem(row, 1, date_item)
                    self.records_table.setItem(row, 2, time_item)
                    self.records_table.setItem(row, 3, accuracy_item)

                print(f"Populated table with {len(detection_log)} records.")

            else:
                print("Warning: Camera pipeline or get_detection_log method not available.")
                # Optional: Display an error message in the table
                self.records_table.setRowCount(1)
                error_item = QTableWidgetItem("Could not retrieve records (Pipeline not ready).")
                error_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.records_table.setItem(0, 0, error_item)
                self.records_table.setSpan(0, 0, 1, self.records_table.columnCount())


        except Exception as e:
            print(f"Error populating records table: {e}")
            # Optional: Display an error message in the table
            self.records_table.setRowCount(1)
            error_item = QTableWidgetItem(f"Error loading records: {e}")
            error_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.records_table.setItem(0, 0, error_item)
            self.records_table.setSpan(0, 0, 1, self.records_table.columnCount())


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