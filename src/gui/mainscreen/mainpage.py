import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QStackedWidget, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QCursor
from src.backend.api.auth.session_manager import SessionManager
from src.backend.inference import CameraWidget

class MainScreen(QWidget):
    def __init__(self, logout_callback):
        super().__init__()
        self.setWindowTitle("Main Screen")
        self.resize(900, 600)
        self.logout_callback = logout_callback
        self._setup_ui()

    def _setup_ui(self):
        current_user = SessionManager.get_user()
        sidebarbuttons=[]
        cursor_pointer = QCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        sidebar_layout.setSpacing(10)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        sidebar.setObjectName("sidebar")

        # Buttons
        self.dashboard_button = QPushButton("Dashboard")
        self.dashboard_button.setIcon(qta.icon('ri.dashboard-line'))
        self.dashboard_button.setIconSize(QSize(18, 18))
        self.dashboard_button.setCursor(cursor_pointer)
        self.dashboard_button.setProperty("class", "sidebar-buttons")
        self.dashboard_button.setCheckable(True)
        self.dashboard_button.setChecked(True)
        sidebarbuttons.append(self.dashboard_button)
        
        if current_user and current_user.get("role") == "admin":
            self.train_button = QPushButton("Train")
            self.train_button.setIcon(qta.icon('ph.camera'))
            self.train_button.setIconSize(QSize(18, 18))
            self.train_button.setCursor(cursor_pointer)
            self.train_button.setProperty("class", "sidebar-buttons")
            self.train_button.setCheckable(True)
            sidebarbuttons.append(self.train_button)

        self.settings_button = QPushButton("Settings")
        self.settings_button.setIcon(qta.icon('ri.settings-4-line'))
        self.settings_button.setIconSize(QSize(18, 18))
        self.settings_button.setCursor(cursor_pointer)
        self.settings_button.setProperty("class", "sidebar-buttons")
        self.settings_button.setCheckable(True)
        sidebarbuttons.append(self.settings_button)

        self.logout_button = QPushButton("Log Out")
        self.logout_button.setIcon(qta.icon('ri.logout-box-line'))
        self.logout_button.setIconSize(QSize(18, 18))
        self.logout_button.setCursor(cursor_pointer)
        self.logout_button.setProperty("class", "sidebar-buttons")
        self.logout_button.clicked.connect(self.logout)
        sidebarbuttons.append(self.logout_button)

        for btn in sidebarbuttons:
            sidebar_layout.addWidget(btn)

        sidebar_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Stack
        self.stack = QStackedWidget()

        # Pages
        self.stack.addWidget(self._build_dashboard_page())
        self.stack.addWidget(self._build_train_page())
        self.stack.addWidget(self._build_settings_page())

        layout.addWidget(sidebar)
        layout.addWidget(self.stack)

        # Connect
        self.dashboard_button.clicked.connect(lambda: self._switch_page(0, self.dashboard_button))
        if hasattr(self, "train_button"):
            self.train_button.clicked.connect(lambda: self._switch_page(1, self.train_button))
        self.settings_button.clicked.connect(lambda: self._switch_page(2, self.settings_button))

        # Style
        self.setStyleSheet("""
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
        """)

        self._update_icon_colors()


    def _build_dashboard_page(self):
        page = QWidget()
        dashboard_layout = QVBoxLayout(page)
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

        cursor_pointer = QCursor(Qt.CursorShape.PointingHandCursor)  # Pointer cursor for the button
        self.toggle_feed_button = QPushButton("Start Feed")
        self.toggle_feed_button.setCursor(cursor_pointer)
        self.toggle_feed_button.setFixedWidth(100)
        self.toggle_feed_button.clicked.connect(self.toggle_feed)
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

        # Arrange the top layout (label and button)
        camera_top_layout.addWidget(camera_label)
        camera_top_layout.addStretch()  # Push button to the right
        camera_top_layout.addWidget(self.toggle_feed_button)

        # Camera widget
        self.camera_widget = CameraWidget()
        self.camera_widget.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")  # Optional style for background
        self.camera_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Allow camera widget to expand

        # Add the top row layout and camera widget to the camera card
        camera_card_layout.addLayout(camera_top_layout)
        camera_card_layout.addWidget(self.camera_widget)
        camera_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")
        camera_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Allow card to expand vertically

        # Threshold card
        threshold_card = QWidget()
        threshold_card_layout = QHBoxLayout(threshold_card)
        threshold_card_layout.setContentsMargins(15, 10, 15, 10)  # Adjust margins if needed
        threshold_card_layout.setSpacing(10)  # Add spacing between elements

        threshold_label = QLabel("Minimum Face Recognition Threshold:")  # Changed label text slightly
        threshold_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)  # Align left

        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("0.0-1.0")  # Shortened placeholder
        self.threshold_input.setText("0.6")  # Default value
        self.threshold_input.setFixedWidth(80)  # Adjust width as needed

        self.apply_threshold = QPushButton("Apply")
        self.apply_threshold.setCursor(cursor_pointer)
        self.apply_threshold.setFixedWidth(80)  # Adjust width as needed
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

        # Add widgets directly to the horizontal card layout
        threshold_card_layout.addWidget(threshold_label)
        threshold_card_layout.addWidget(self.threshold_input)
        threshold_card_layout.addWidget(self.apply_threshold)
        threshold_card_layout.addStretch()  # Push elements to the left

        # Apply additional styles for the threshold card
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

        # Adjust max height for the threshold card
        threshold_card.setMaximumHeight(60)  # Adjust max height if needed

        # Add the camera card and threshold card to the dashboard layout
        dashboard_layout.addWidget(camera_card)
        dashboard_layout.addWidget(threshold_card)

        return page

    def _build_train_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        train_label = QLabel("Live Camera for Training")
        train_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        train_label.setStyleSheet("background-color: #2a2b2e; border-radius: 6px; min-height: 400px;")

        info_label = QLabel("Training Information/Controls")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("background-color: #2a2b2e; border-radius: 6px; max-height: 100px;")

        layout.addWidget(train_label)
        layout.addWidget(info_label)
        layout.addStretch()
        return page

    def _build_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel("This is settings.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return page

    def _switch_page(self, index, button):
        self.dashboard_button.setChecked(False)
        if hasattr(self, "train_button"):
            self.train_button.setChecked(False)
        self.settings_button.setChecked(False)

        button.setChecked(True)
        self.stack.setCurrentIndex(index)
        self._update_icon_colors()

    def _update_icon_colors(self):
        icons = {
            self.dashboard_button: 'ri.dashboard-line',
            self.settings_button: 'ri.settings-4-line',
        }

        if hasattr(self, "train_button"):
            icons[self.train_button] = 'ph.camera-light'

        for btn, icon_name in icons.items():
            color = 'black' if btn.isChecked() else 'white'
            btn.setIcon(qta.icon(icon_name, color=color))

    def logout(self):
        SessionManager.clear_user()
        self.close()
        self.logout_callback()

    def toggle_feed(self):
        """Toggles the camera feed on or off."""
        if self.toggle_feed_button.text() == "Start Feed":
            self.camera_widget.start_feed()  # Start the camera feed
            self.toggle_feed_button.setText("Stop Feed")  # Change button text to "Stop Feed"
        else:
            self.camera_widget.stop_feed()  # Stop the camera feed
            self.toggle_feed_button.setText("Start Feed")  # Change button text to "Start Feed"

    def update_threshold(self):
        """Updates the face recognition threshold."""
        threshold_value = self.threshold_input.text()
        try:
            threshold_value = float(threshold_value)
            if 0.0 <= threshold_value <= 1.0:
                self.camera_widget.set_recognition_threshold(threshold_value)  # Update threshold in CameraWidget
        except ValueError:
            pass  # Ignore invalid threshold input