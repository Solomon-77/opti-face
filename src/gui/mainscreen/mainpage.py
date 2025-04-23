import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QStackedWidget, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QCursor
from backend.api.auth.session_manager import SessionManager

class MainScreen(QWidget):
    def __init__(self, logout_callback):
        super().__init__()
        self.setWindowTitle("Main Screen")
        self.resize(900, 600)
        self.logout_callback = logout_callback
        self._setup_ui()

    def _setup_ui(self):
        current_user = SessionManager.get_user()
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

        self.train_button = QPushButton("Train")
        self.train_button.setIcon(qta.icon('ph.camera'))
        self.train_button.setIconSize(QSize(18, 18))
        self.train_button.setCursor(cursor_pointer)
        self.train_button.setProperty("class", "sidebar-buttons")
        self.train_button.setCheckable(True)

        self.settings_button = QPushButton("Settings")
        self.settings_button.setIcon(qta.icon('ri.settings-4-line'))
        self.settings_button.setIconSize(QSize(18, 18))
        self.settings_button.setCursor(cursor_pointer)
        self.settings_button.setProperty("class", "sidebar-buttons")
        self.settings_button.setCheckable(True)
            
        self.logout_button = QPushButton("Log Out")
        self.logout_button.setIcon(qta.icon('ri.logout-box-line'))
        self.logout_button.setIconSize(QSize(18, 18))
        self.logout_button.setCursor(cursor_pointer)
        self.logout_button.setProperty("class", "sidebar-buttons")
        self.logout_button.clicked.connect(self.logout)

        if current_user and current_user.get("role") == "admin":
            for btn in [self.dashboard_button, self.train_button, self.settings_button, self.logout_button]:
                sidebar_layout.addWidget(btn)
        else:
            for btn in [self.dashboard_button, self.settings_button, self.logout_button]:
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
        layout = QVBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        camera_card = QLabel("Live Face Recognition Feed")
        camera_card.setAlignment(Qt.AlignmentFlag.AlignCenter)
        camera_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px; min-height: 400px;")

        threshold_card = QLabel("Threshold Adjustment")
        threshold_card.setAlignment(Qt.AlignmentFlag.AlignCenter)
        threshold_card.setStyleSheet("background-color: #2a2b2e; border-radius: 6px; max-height: 100px;")

        layout.addWidget(camera_card)
        layout.addWidget(threshold_card)
        layout.addStretch()
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
