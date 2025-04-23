from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCursor

class LoginWindow(QWidget):
    def __init__(self, login_callback):
        super().__init__()
        self.setWindowTitle("OptiFace")
        self.resize(450, 600)
        self.login_callback = login_callback

        self._setup_ui()

    def _setup_ui(self):
        # Title
        self.title = QLabel("Face Recognition System")
        self.title.setObjectName("title")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setWordWrap(True)

        # Username
        self.username = QLineEdit()
        self.username.setProperty("class", "user-pass")
        self.username.setPlaceholderText("Username")

        # Password
        self.password = QLineEdit()
        self.password.setProperty("class", "user-pass")
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.returnPressed.connect(self.attempt_login)

        # Error Label
        self.error_label = QLabel("")
        self.error_label.setObjectName("error-label")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.hide()

        # Login Button
        self.login_button = QPushButton("Login")
        self.login_button.setObjectName("login-button")
        self.login_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.login_button.clicked.connect(self.attempt_login)

        # Layout
        form_layout = QVBoxLayout()
        form_layout.setSpacing(20)
        form_layout.setContentsMargins(30, 0, 30, 0)
        form_layout.addWidget(self.title)
        form_layout.addWidget(self.username)
        form_layout.addWidget(self.password)
        form_layout.addWidget(self.error_label)
        form_layout.addWidget(self.login_button)

        container = QWidget()
        container.setMaximumWidth(350)
        container.setLayout(form_layout)

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(container)

        # Styling
        self.setStyleSheet("""
            QWidget {
                background-color: #202124;
                font-family: "Consolas", "Courier New", monospace;
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
        """)

    def attempt_login(self):
        user = self.username.text()
        pwd = self.password.text()
        self.login_callback(user, pwd)

    def show_error(self, message):
        self.error_label.setText(message)
        self.error_label.show()
        self.password.clear()
        QTimer.singleShot(5000, self.error_label.hide)
