import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QCursor, QIcon, QPixmap

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
        camera_label = QLabel("Live Face Recognition Feed")
        camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_card_layout.addWidget(camera_label)
        camera_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")
        camera_card.setMinimumHeight(400)
        
        threshold_card = QWidget()
        threshold_card_layout = QVBoxLayout(threshold_card)
        threshold_label = QLabel("Threshold Adjustment")
        threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        threshold_card_layout.addWidget(threshold_label)
        threshold_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")
        threshold_card.setMaximumHeight(100)
        
        dashboard_layout.addWidget(camera_card)
        dashboard_layout.addWidget(threshold_card)
        dashboard_layout.addStretch()
        
        
        # --- Train page ---
        train_page = QWidget()
        train_layout = QVBoxLayout(train_page)
        train_layout.setContentsMargins(15, 15, 15, 15)
        train_layout.setSpacing(15)

        # Placeholder for Camera Feed Card
        train_camera_card = QWidget()
        train_camera_card_layout = QVBoxLayout(train_camera_card)
        train_camera_label = QLabel("Live Camera for Training")
        train_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        train_camera_card_layout.addWidget(train_camera_label)
        train_camera_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")
        train_camera_card.setMinimumHeight(400) # Example height

        # Placeholder for Generic Card
        train_info_card = QWidget()
        train_info_card_layout = QVBoxLayout(train_info_card)
        train_info_label = QLabel("Training Information/Controls")
        train_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        train_info_card_layout.addWidget(train_info_label)
        train_info_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px;")
        train_info_card.setMaximumHeight(100)

        train_layout.addWidget(train_camera_card)
        train_layout.addWidget(train_info_card)
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
            """
        )
        
        self._update_icon_color(self.dashboard_button)
        self._update_icon_color(self.train_button)
        self._update_icon_color(self.settings_button)
        
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