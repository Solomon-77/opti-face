import sys
import os
import shutil
import cv2
import datetime
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
    QScrollArea,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFormLayout,
    QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QCursor, QIcon, QPixmap, QIntValidator 
from src.backend.inference import CameraWidget
from src.backend.prepare_embeddings import generate_and_save_embeddings
from src.backend.utils.face_utils import load_face_recognition_model
from src.gui.styles import LOGIN_STYLES, ADMIN_STYLES, TABLE_STYLES

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
        layout.setContentsMargins(0, 10, 10, 10)
        layout.setSpacing(10)
        self.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")

        # File selection row
        file_selection_layout = QHBoxLayout()
        file_selection_layout.setContentsMargins(0, 0, 0, 0)
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
        self.upload_video_button.setStyleSheet("""
            QPushButton {
                background-color: #5f6368; color: white;
                border: none; border-radius: 4px; padding: 6px 10px;
            }
            QPushButton:hover { background-color: #707478; }
        """)

        self.selected_files_label = QLabel("No files selected.")
        self.selected_files_label.setStyleSheet("font-size: 11px; color: #aaa; background: none;")

        self.remove_button = QPushButton("Remove")
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
        layout.addSpacing(10)

        # Name input row
        name_input_layout = QHBoxLayout()
        name_label = QLabel("Person's Name:")
        name_label.setStyleSheet("background: none; font-weight: bold;")
        self.person_name_input = QLineEdit()
        self.person_name_input.setPlaceholderText("Enter name")
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
        layout.addSpacing(10)

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
        self.is_feed_running = False
        self.face_database_dir = './src/backend/face_database/'
        self.person_entry_widgets = []
        self.training_frame_interval = 5
        self.icon_folder = "src/gui/icons"

        try:
            model_path = "src/backend/checkpoints/edgeface_s_gamma_05.pt"
            self.training_model, self.training_device = load_face_recognition_model(model_path=model_path)
            print("Training model loaded successfully.")
        except Exception as e:
            print(f"Error loading training model: {e}")
            QMessageBox.critical(self, "Model Load Error", f"Failed to load the face recognition model for training: {e}")
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

        # --- Add Spacer to push Logout button down ---
        sidebar_layout.addStretch()

        # --- Logout Button ---
        self.logout_button = QPushButton("Logout")
        self.logout_button.setIconSize(QSize(18, 18))
        self.logout_button.setCursor(cursor_pointer)
        self.logout_button.setProperty("class", "sidebar-buttons")

        try:
            logout_icon_path = f"{self.icon_folder}/logout_white.svg"
            logout_icon = QIcon(QPixmap(logout_icon_path))
            if not logout_icon.isNull():
                self.logout_button.setIcon(logout_icon)
            else:
                print(f"Warning: Could not load logout icon from {logout_icon_path}")
        except Exception as e:
            print(f"Error loading logout icon {logout_icon_path}: {e}")

        self.logout_button.clicked.connect(self.logout_action) # Connect to logout method
        sidebar_layout.addWidget(self.logout_button)

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
        camera_top_layout.addStretch()
        camera_top_layout.addWidget(self.toggle_feed_button)

        self.camera_widget = CameraWidget()
        camera_card_layout.addLayout(camera_top_layout)
        camera_card_layout.addWidget(self.camera_widget)
        camera_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")
        camera_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        threshold_card = QWidget()
        threshold_card_layout = QHBoxLayout(threshold_card)
        threshold_card_layout.setContentsMargins(15, 10, 15, 10)
        threshold_card_layout.setSpacing(10)

        threshold_label = QLabel("Minimum Face Recognition Threshold:")
        threshold_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("0.0-1.0")
        self.threshold_input.setText("0.6")
        self.threshold_input.setFixedWidth(80)

        self.apply_threshold = QPushButton("Apply")
        self.apply_threshold.setCursor(cursor_pointer)
        self.apply_threshold.setFixedWidth(80)
        self.apply_threshold.clicked.connect(self.update_threshold)
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

        threshold_card_layout.addWidget(threshold_label)
        threshold_card_layout.addWidget(self.threshold_input)
        threshold_card_layout.addWidget(self.apply_threshold)
        threshold_card_layout.addStretch()

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
        threshold_card.setMaximumHeight(60)

        dashboard_layout.addWidget(camera_card)
        dashboard_layout.addWidget(threshold_card)

        # --- Train page ---
        train_page = QWidget()
        train_main_layout = QVBoxLayout(train_page)
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
        train_main_layout.addWidget(self.add_person_entry_button, alignment=Qt.AlignmentFlag.AlignLeft)

        # Scroll Area for Person Entries
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #202124; }")

        # Container widget inside Scroll Area
        self.scroll_content_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_content_widget)

        # Layout for the container widget (holds PersonEntryWidgets)
        self.person_entries_layout = QVBoxLayout(self.scroll_content_widget)
        self.person_entries_layout.setContentsMargins(0, 0, 0, 0)
        self.person_entries_layout.setSpacing(10)
        self.person_entries_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        train_main_layout.addWidget(self.scroll_area)

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
        self.db_scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #2a2b2e; border-radius: 6px; }")

        self.db_scroll_content_widget = QWidget()
        self.db_scroll_area.setWidget(self.db_scroll_content_widget)

        self.database_list_layout = QVBoxLayout(self.db_scroll_content_widget)
        self.database_list_layout.setContentsMargins(10, 10, 10, 10)
        self.database_list_layout.setSpacing(8)
        self.database_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        database_layout.addWidget(self.db_scroll_area)

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
        self.records_table.setColumnCount(4)
        self.records_table.setHorizontalHeaderLabels(["Name", "Date", "Time", "Accuracy"])
        self.records_table.verticalHeader().setVisible(False)
        self.records_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.records_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.records_table.setAlternatingRowColors(True)

        # Apply table styles
        self.records_table.setStyleSheet(TABLE_STYLES)

        # Styling for the admin window
        self.setStyleSheet(ADMIN_STYLES)

        header = self.records_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        records_layout.addWidget(self.records_table)

        # --- Settings page ---
        settings_page = QWidget()
        settings_main_layout = QVBoxLayout(settings_page)
        settings_main_layout.setContentsMargins(25, 25, 25, 25)
        settings_main_layout.setSpacing(15)
        settings_main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Settings Container (Card)
        settings_container = QFrame()
        settings_container.setObjectName("settingsCard")
        settings_container_layout = QVBoxLayout(settings_container)
        settings_container_layout.setContentsMargins(20, 20, 20, 20)
        settings_container_layout.setSpacing(15)
        settings_label = QLabel("Application Settings")
        settings_label.setStyleSheet("font-size: 16px; font-weight: bold; background: none;")
        settings_container_layout.addWidget(settings_label)

        # Add spacing before the form layout
        settings_container_layout.addSpacing(10)

        # Use QFormLayout for label-input pairs
        settings_form_layout = QFormLayout()
        settings_form_layout.setSpacing(12)
        settings_form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        settings_form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Add FPS Display toggle
        fps_display_label = QLabel("Show FPS Counter:")
        fps_display_label.setStyleSheet("background: none; font-weight: 600;")
        self.fps_toggle_checkbox = QCheckBox()
        self.fps_toggle_checkbox.setChecked(False)  # Default to not showing FPS
        self.fps_toggle_checkbox.setStyleSheet("""
            QCheckBox {
                background: none;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                background-color: #3a3b3e;
                border: 1px solid #5f6368;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #8ab4f8;
                image: url(src/gui/icons/check_black.svg);
            }
        """)
        settings_form_layout.addRow(fps_display_label, self.fps_toggle_checkbox)

        # Log Interval Setting
        log_interval_label = QLabel("Log Interval (seconds):")
        log_interval_label.setStyleSheet("background: none; font-weight: 600;")
        self.log_interval_input = QLineEdit()
        self.log_interval_input.setPlaceholderText("e.g., 15")
        default_log_interval = 15
        if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
             default_log_interval = self.camera_widget.pipeline.get_log_interval()
        self.log_interval_input.setText(str(default_log_interval))
        self.log_interval_input.setValidator(QIntValidator(0, 100, self))
        self.log_interval_input.setFixedWidth(80)
        self.log_interval_input.setStyleSheet("""
            QLineEdit {
                padding: 6px; border: 1px solid #5f6368; border-radius: 4px;
                background-color: #3a3b3e; color: white;
            }
            QLineEdit:focus { border: 1px solid #8ab4f8; }
        """)
        settings_form_layout.addRow(log_interval_label, self.log_interval_input)

        # Training Frame Interval Setting
        train_interval_label = QLabel("Training Frame Interval:")
        train_interval_label.setStyleSheet("background: none; font-weight: 600;")
        self.train_interval_input = QLineEdit()
        self.train_interval_input.setPlaceholderText("e.g., 5")
        self.train_interval_input.setText(str(self.training_frame_interval))
        self.train_interval_input.setValidator(QIntValidator(1, 100, self))
        self.train_interval_input.setFixedWidth(80)
        self.train_interval_input.setStyleSheet("""
             QLineEdit {
                padding: 6px; border: 1px solid #5f6368; border-radius: 4px;
                background-color: #3a3b3e; color: white;
             }
             QLineEdit:focus { border: 1px solid #8ab4f8; }
        """)
        settings_form_layout.addRow(train_interval_label, self.train_interval_input)

        # Frame Skip Interval Setting
        frame_skip_label = QLabel("Frame Skip Interval:")
        frame_skip_label.setStyleSheet("background: none; font-weight: 600;")
        self.frame_skip_input = QLineEdit()
        self.frame_skip_input.setPlaceholderText("e.g., 1")
        default_frame_skip = 1
        if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
            default_frame_skip = self.camera_widget.pipeline.get_frame_skip_interval()
        self.frame_skip_input.setText(str(default_frame_skip))
        self.frame_skip_input.setValidator(QIntValidator(1, 30, self))
        self.frame_skip_input.setFixedWidth(80)
        self.frame_skip_input.setStyleSheet("""
            QLineEdit {
                padding: 6px; border: 1px solid #5f6368; border-radius: 4px;
                background-color: #3a3b3e; color: white;
            }
            QLineEdit:focus { border: 1px solid #8ab4f8; }
        """)
        settings_form_layout.addRow(frame_skip_label, self.frame_skip_input)

        settings_container_layout.addLayout(settings_form_layout)

        # Apply Settings Button
        self.apply_settings_button = QPushButton("Apply Settings")
        self.apply_settings_button.setCursor(cursor_pointer)
        self.apply_settings_button.clicked.connect(self.apply_settings)
        self.apply_settings_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; color: white; font-weight: bold;
                border: none; border-radius: 4px; padding: 8px 15px; margin-top: 10px; /* Adjusted margin */
                max-width: 150px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        settings_container_layout.addWidget(self.apply_settings_button, alignment=Qt.AlignmentFlag.AlignLeft)

        settings_main_layout.addWidget(settings_container)
        settings_main_layout.addStretch()

        # Add pages to stack
        self.contentStack.addWidget(dashboard_page)
        self.contentStack.addWidget(train_page)
        self.contentStack.addWidget(database_page)
        self.contentStack.addWidget(records_page)
        self.contentStack.addWidget(settings_page)

        admin_layout.addWidget(sidebar)
        admin_layout.addWidget(self.contentStack)

        self.dashboard_button.clicked.connect(lambda: self._update_button_states(0, self.dashboard_button))
        self.train_button.clicked.connect(lambda: self._update_button_states(1, self.train_button))
        self.database_button.clicked.connect(lambda: self._update_button_states(2, self.database_button))
        self.records_button.clicked.connect(lambda: self._update_button_states(3, self.records_button))
        self.settings_button.clicked.connect(lambda: self._update_button_states(4, self.settings_button))

        # Initial population of database list when window is created
        self.populate_database_list()
        # Initial population of records table
        self.populate_records_table()

        # Main window styling
        app.setStyleSheet(LOGIN_STYLES)

        # Update icons for checkable buttons initially
        self._update_icon_color(self.dashboard_button)
        self._update_icon_color(self.train_button)
        self._update_icon_color(self.database_button)
        self._update_icon_color(self.records_button)
        self._update_icon_color(self.settings_button)

    def add_person_entry_widget(self):
        """Creates and adds a new PersonEntryWidget to the scroll area."""
        entry_widget = PersonEntryWidget(self)
        self.person_entries_layout.addWidget(entry_widget)
        self.person_entry_widgets.append(entry_widget)
        
        if self.person_entries_layout.count() > 1:
             previous_widget_index = self.person_entries_layout.count() - 2
             item = self.person_entries_layout.itemAt(previous_widget_index)
             if item and isinstance(item.widget(), PersonEntryWidget):
                 separator = QFrame()
                 separator.setFrameShape(QFrame.Shape.HLine)
                 separator.setFrameShadow(QFrame.Shadow.Sunken)
                 separator.setStyleSheet("border: 1px solid #3a3b3e;")
                 self.person_entries_layout.insertWidget(self.person_entries_layout.count() - 1, separator)

    def remove_person_entry_widget(self, target_widget: PersonEntryWidget):
        """Removes a specific PersonEntryWidget and its preceding separator."""
        if target_widget in self.person_entry_widgets:
            try:
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
                elif index == 0 and self.person_entries_layout.count() > 0:
                     item_after = self.person_entries_layout.itemAt(0)
                     if item_after and isinstance(item_after.widget(), QFrame):
                         separator_widget = item_after.widget()
                         self.person_entries_layout.removeWidget(separator_widget)
                         separator_widget.deleteLater()
                         print("Removed succeeding separator (was first item).")


            except Exception as e:
                print(f"Error removing widget: {e}")
                QMessageBox.warning(self, "Error", f"Could not remove the entry: {e}")

    def select_images(self, target_widget: PersonEntryWidget):
        """Opens a file dialog to select multiple image files for a specific person entry."""
        image_extensions = "*.png *.jpg *.jpeg *.bmp *.tiff"
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images for Training",
            "",
            f"Image Files ({image_extensions});;All Files (*)"
        )
        if files:
            target_widget.selected_image_paths = files
            target_widget.selected_video_path = None
            target_widget.selected_files_label.setText(f"{len(files)} image(s) selected.")
            print(f"Selected images for widget: {files}")
        else:
            print("No images selected for widget.")

    def select_video(self, target_widget: PersonEntryWidget):
        """Opens a file dialog to select a single video file for a specific person entry."""

        video_extensions = "*.mp4 *.avi *.mov *.mkv"
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video for Training",
            "",
            f"Video Files ({video_extensions});;All Files (*)"
        )
        if file:
            target_widget.selected_video_path = file
            target_widget.selected_image_paths = []
            target_widget.selected_files_label.setText(f"Video: {os.path.basename(file)}")
            print(f"Selected video for widget: {file}")
        else:
            print("No video selected for widget.")

    def train_all_persons_action(self):
        """Processes and trains all valid person entries."""
        print("Starting training for all added persons...")
        if self.training_model is None or self.training_device is None:
             QMessageBox.critical(self, "Error", "Training model not loaded. Cannot train.")
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
                continue

            # Process this single person entry
            npz_path = self._process_single_person(entry_widget)

            if npz_path:
                processed_count += 1
                widgets_to_clear.append(entry_widget)
            else:
                error_count += 1

        # --- Summary Message ---
        summary_message = f"Training finished. Processed: {processed_count}, Errors/Warnings: {error_count}."
        QMessageBox.information(self, "Training Complete", summary_message)
        print(summary_message)

    def _process_single_person(self, target_widget: PersonEntryWidget):
        """Handles processing (validation, file handling, embedding) for one PersonEntryWidget. Returns npz_path on success, None on failure."""
        person_name = target_widget.person_name_input.text().strip()
        use_video = target_widget.selected_video_path is not None
        use_images = bool(target_widget.selected_image_paths)

        # --- Directory and File Handling ---
        person_folder = os.path.join(self.face_database_dir, person_name)
        extracted_frames_count = 0
        try:
            os.makedirs(person_folder, exist_ok=True)
            print(f"Processing entry for: {person_name}")

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
                frame_interval = self.training_frame_interval

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        # Save frame as JPEG
                        frame_filename = f"{person_name}_frame_{saved_frame_count:04d}.jpg"
                        frame_path = os.path.join(person_folder, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        saved_frame_count += 1

                    frame_count += 1
                    if frame_count % 100 == 0:
                         target_widget.train_status_label.setText(f"Extracting... (Frame {frame_count})")
                         QApplication.processEvents()


                cap.release()
                extracted_frames_count = saved_frame_count
                print(f"Extracted {extracted_frames_count} frames from video.")

                if extracted_frames_count == 0:
                     target_widget.train_status_label.setText("Error: Failed to extract frames.")
                     target_widget.train_status_label.setStyleSheet("color: #f28b82;")
                     return None

                target_widget.train_status_label.setText(f"Extracted {extracted_frames_count} frames. Training...")
                target_widget.train_status_label.setStyleSheet("color: #8ab4f8;")
                QApplication.processEvents() # Update UI

            elif use_images:
                target_widget.train_status_label.setText("Copying images...") # Update status
                target_widget.train_status_label.setStyleSheet("color: #8ab4f8;")
                QApplication.processEvents()
                copied_count = 0
                for img_path in target_widget.selected_image_paths:
                    try:
                        dest_filename = os.path.basename(img_path)
                        dest_path = os.path.join(person_folder, dest_filename)
                        shutil.copy(img_path, dest_path)
                        print(f"Copied {img_path} to {dest_path}")
                        copied_count += 1
                    except Exception as copy_err:
                        print(f"Error copying file {img_path}: {copy_err}")

                if copied_count == 0:
                     target_widget.train_status_label.setText("Error: Failed to copy images.")
                     target_widget.train_status_label.setStyleSheet("color: #f28b82;")
                     return None

                target_widget.train_status_label.setText(f"Copied {copied_count} images. Training...")
                target_widget.train_status_label.setStyleSheet("color: #8ab4f8;")
                QApplication.processEvents() # Update UI

        except IOError as vid_err:
             target_widget.train_status_label.setText(f"Error opening video: {vid_err}")
             target_widget.train_status_label.setStyleSheet("color: #f28b82;")
             return None
        except Exception as dir_err:
            target_widget.train_status_label.setText(f"Error creating directory: {dir_err}")
            target_widget.train_status_label.setStyleSheet("color: #f28b82;")
            return None

        # --- Embedding Generation (uses images in person_folder) ---
        try:
            npz_path = generate_and_save_embeddings(
                person_name=person_name,
                person_folder=person_folder,
                output_dir=self.face_database_dir, # Save .npz in the main db dir
                model=self.training_model,
                device=self.training_device
            )

            if npz_path:
                target_widget.train_status_label.setText(f"Success: Embeddings saved.")
                target_widget.train_status_label.setStyleSheet("color: #4CAF50;")

                # --- Update Live Recognition Pipeline ---
                if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
                    self.camera_widget.pipeline.load_new_person(person_name, npz_path)
                    print(f"Requested pipeline to load new embeddings for {person_name}")
                else:
                     print(f"Warning: CameraWidget pipeline not available for {person_name}.")

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
            return None

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
        # Uncheck all navigation buttons first
        self.dashboard_button.setChecked(False)
        self.train_button.setChecked(False)
        self.database_button.setChecked(False)
        self.records_button.setChecked(False)
        self.settings_button.setChecked(False)

        # Check the clicked navigation button
        clicked_button.setChecked(True)

        # Switch to the correct page in the stack
        self.contentStack.setCurrentIndex(index)

        # Update icons for all checkable buttons
        self._update_icon_color(self.dashboard_button)
        self._update_icon_color(self.train_button)
        self._update_icon_color(self.database_button)
        self._update_icon_color(self.records_button)
        self._update_icon_color(self.settings_button)

        # Populate records if switching to the records page
        if index == 3:
            self.populate_records_table()

    def _update_icon_color(self, button):
        """Updates the icon color based on the button's checked state."""
        icon_base_name = ""

        if button == self.dashboard_button:
            icon_base_name = 'dashboard'
        elif button == self.train_button:
            icon_base_name = 'camera'
        elif button == self.database_button:
            icon_base_name = 'database'
        elif button == self.records_button:
            icon_base_name = 'records'
        elif button == self.settings_button:
            icon_base_name = 'settings'

        if icon_base_name:
            if button.isChecked():
                icon_path = f"{self.icon_folder}/{icon_base_name}_black.svg"
            else:
                icon_path = f"{self.icon_folder}/{icon_base_name}_white.svg"

            try:
                icon = QIcon(QPixmap(icon_path))
                if not icon.isNull():
                    button.setIcon(icon)
                else:
                    print(f"Warning: Could not load icon from {icon_path}")
                    button.setIcon(QIcon())
            except Exception as e:
                print(f"Error loading icon {icon_path}: {e}")
                button.setIcon(QIcon())

    def update_threshold(self):
        try:
            threshold = float(self.threshold_input.text())
            if 0 <= threshold <= 1:
                if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
                    self.camera_widget.set_recognition_threshold(threshold)
                    print(f"Recognition threshold set to: {threshold}")
                else:
                    print("Warning: CameraWidget or pipeline not initialized. Cannot set threshold.")
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
        while self.database_list_layout.count():
            item = self.database_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        print("Populating database list...")
        try:
            found_files = False
            if not os.path.isdir(self.face_database_dir):
                 raise FileNotFoundError(f"Database directory not found: {self.face_database_dir}")

            for filename in sorted(os.listdir(self.face_database_dir)):
                if filename.endswith('.npz'):
                    found_files = True
                    person_name = os.path.splitext(filename)[0]
                    npz_path = os.path.join(self.face_database_dir, filename)

                    # Create a widget for each person entry
                    entry_widget = QWidget()
                    entry_layout = QHBoxLayout(entry_widget)
                    entry_layout.setContentsMargins(5, 5, 5, 5)
                    entry_widget.setStyleSheet("background-color: #3a3b3e; border-radius: 4px;")

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

        except FileNotFoundError as fnf_error:
            print(fnf_error)
            error_label = QLabel(f"Error: Database directory not found at\n{self.face_database_dir}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setStyleSheet("color: #f28b82;")
            self.database_list_layout.addWidget(error_label)
            self.database_list_layout.addStretch()
        except Exception as e:
            print(f"Error populating database list: {e}")
            error_label = QLabel(f"An error occurred: {e}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setStyleSheet("color: #f28b82;")
            self.database_list_layout.addWidget(error_label)
            self.database_list_layout.addStretch()

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
                    deleted_folder = True
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
            except Exception as e:
                 print(f"Error updating pipeline to remove {person_name}: {e}")
                 QMessageBox.warning(self, "Pipeline Error", f"Could not update the live recognition pipeline: {e}")

            # 4. Refresh the UI list
            self.populate_database_list()

            # Optional: Show success message if key parts succeeded
            if deleted_npz and deleted_folder and pipeline_updated:
                 QMessageBox.information(self, "Deletion Successful", f"Successfully removed {person_name} from the database and live recognition.")
            elif deleted_npz:
                 QMessageBox.information(self, "Deletion Partially Successful", f"Removed {person_name}'s .npz file. Check console for details on folder/pipeline updates.")  


        else:
            print(f"Deletion cancelled for {person_name}.")

    def populate_records_table(self):
        """Fetches detection records and populates the records table."""
        print("Populating detection records table...")
        self.records_table.setRowCount(0)

        try:
            # Check if pipeline exists and has the method
            if hasattr(self.camera_widget, 'pipeline') and \
               self.camera_widget.pipeline and \
               hasattr(self.camera_widget.pipeline, 'get_detection_log'):

                detection_log = self.camera_widget.pipeline.get_detection_log()

                if not detection_log:
                    self.records_table.setRowCount(1)
                    no_records_item = QTableWidgetItem("No detection records available yet.")
                    no_records_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    no_records_item.setFlags(no_records_item.flags() ^ Qt.ItemFlag.ItemIsSelectable ^ Qt.ItemFlag.ItemIsEditable)
                    self.records_table.setItem(0, 0, no_records_item)
                    self.records_table.setSpan(0, 0, 1, self.records_table.columnCount())
                    print("No detection records found in the pipeline.")
                    return

                self.records_table.setRowCount(len(detection_log))
                for row, record in enumerate(reversed(detection_log)):
                    name, timestamp, similarity = record

                    # Format timestamp
                    dt_object = datetime.datetime.fromtimestamp(timestamp)
                    date_str = dt_object.strftime("%Y-%m-%d")
                    time_str = dt_object.strftime("%I:%M %p")
                    accuracy_str = f"{similarity:.2f}"

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
                self.records_table.setRowCount(1)
                error_item = QTableWidgetItem("Could not retrieve records (Pipeline not ready).")
                error_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                error_item.setFlags(error_item.flags() ^ Qt.ItemFlag.ItemIsSelectable ^ Qt.ItemFlag.ItemIsEditable)
                self.records_table.setItem(0, 0, error_item)
                self.records_table.setSpan(0, 0, 1, self.records_table.columnCount())


        except Exception as e:
            print(f"Error populating records table: {e}")
            self.records_table.setRowCount(1)
            error_item = QTableWidgetItem(f"Error loading records: {e}")
            error_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            error_item.setFlags(error_item.flags() ^ Qt.ItemFlag.ItemIsSelectable ^ Qt.ItemFlag.ItemIsEditable)
            self.records_table.setItem(0, 0, error_item)
            self.records_table.setSpan(0, 0, 1, self.records_table.columnCount())

    def apply_settings(self):
        """Applies the settings from the Settings page."""
        print("Applying settings...")
        settings_applied = []
        try:
            # Apply FPS Display Setting
            show_fps = self.fps_toggle_checkbox.isChecked()
            if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
                self.camera_widget.pipeline.show_fps = show_fps
                settings_applied.append("FPS Display")
                print(f"FPS display set to: {show_fps}")

            # Apply Log Interval
            log_interval_str = self.log_interval_input.text()
            log_interval_state, log_interval_val, _ = self.log_interval_input.validator().validate(log_interval_str, 0)

            if log_interval_state == QIntValidator.State.Acceptable:
                log_interval = int(log_interval_val)
                if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
                    self.camera_widget.pipeline.set_log_interval(log_interval)
                    print(f"Log interval set to: {log_interval} seconds")
                    settings_applied.append("Log Interval")
                else:
                    print("Warning: Camera pipeline not available to set log interval.")
            else:
                QMessageBox.warning(self, "Invalid Input", f"Log Interval must be a number between 0 and 100. Input was '{log_interval_str}'.")
                return

            # Apply Training Frame Interval
            train_interval_str = self.train_interval_input.text()
            train_interval_state, train_interval_val, _ = self.train_interval_input.validator().validate(train_interval_str, 0)

            if train_interval_state == QIntValidator.State.Acceptable:
                new_interval = int(train_interval_val)
                if new_interval != self.training_frame_interval:
                    self.training_frame_interval = new_interval
                    print(f"Training frame interval set to: {self.training_frame_interval} frames")
                    settings_applied.append("Training Frame Interval")
                else:
                    print("Training frame interval unchanged.")
            else:
                 QMessageBox.warning(self, "Invalid Input", f"Training Frame Interval must be a number between 1 and 100. Input was '{train_interval_str}'.")
                 return

            # Apply Frame Skip Interval
            frame_skip_str = self.frame_skip_input.text()
            frame_skip_state, frame_skip_val, _ = self.frame_skip_input.validator().validate(frame_skip_str, 0)

            if frame_skip_state == QIntValidator.State.Acceptable:
                frame_skip = int(frame_skip_val)
                if hasattr(self.camera_widget, 'pipeline') and self.camera_widget.pipeline:
                    self.camera_widget.pipeline.set_frame_skip_interval(frame_skip)
                    print(f"Frame skip interval set to: {frame_skip}")
                    settings_applied.append("Frame Skip Interval")
                else:
                    print("Warning: Camera pipeline not available to set frame skip interval.")
            else:
                QMessageBox.warning(self, "Invalid Input", f"Frame Skip Interval must be a number between 1 and 30. Input was '{frame_skip_str}'.")
                return

            if settings_applied:
                QMessageBox.information(self, "Settings Applied", f"{', '.join(settings_applied)} updated successfully.")
            else:
                QMessageBox.information(self, "Settings", "No settings were changed.")


        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid input for settings: {e}")
            print(f"Error applying settings: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred while applying settings: {e}")
            print(f"Unexpected error applying settings: {e}")

    def logout_action(self):
        """Logs out the admin, closes the admin window, and shows the login window."""
        print("Logging out...")
        # Stop camera feed if running
        if self.is_feed_running:
            self.toggle_camera_feed()

        # Close the admin window
        self.close()

        # Reset and show the login window (accessing global 'window')
        username.clear()
        password.clear()
        error_label.hide()
        window.show()

    def closeEvent(self, event):
        """Ensure camera feed stops when the admin window is closed."""
        print("Admin window closing...")
        if self.is_feed_running:
            self.camera_widget.stop_feed()
            print("Camera feed stopped due to window close.")
        event.accept()

def login():
    if username.text() == ADMIN_USERNAME and password.text() == ADMIN_PASSWORD:
        global admin_window
        admin_window = AdminWindow()
        admin_window.show()
        window.hide()
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

# Main window styling
app.setStyleSheet(LOGIN_STYLES)

# --- Show Window and Run Application ---
admin_window = None
window.show()
sys.exit(app.exec())