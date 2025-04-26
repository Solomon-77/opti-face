import sys
from PyQt6.QtWidgets import QApplication
from src.gui.login.login_page import LoginWindow
from src.gui.mainscreen.mainpage import MainScreen
from src.backend.api.auth.auth_functons import verify_auth_credentials
from src.backend.api.auth.session_manager import SessionManager

class AppController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.login_window = None
        self.main_window = None

    def show_login(self):
        self.login_window = LoginWindow(self.on_login_success)
        self.login_window.show()

    def show_main(self):
        self.main_window = MainScreen(self.on_logout)
        self.main_window.show()

    def on_login_success(self, username, password):
        if verify_auth_credentials(username, password):
            print("✅ Login successful")
            user = SessionManager.get_user()
            if user:
                print(f"Current user: {user['name']}")
                print(f"Role: {user['role']}")
                self.login_window.close()
                self.show_main()
            else:
                print("❌ No active session found.")
        else:
            self.login_window.show_error("Invalid credentials")

    def on_logout(self):
        """Handles the logout event from the MainScreen"""
        print("✅ Logged out successfully")
        SessionManager.clear_user()
        self.show_login()

    def run(self):
        if SessionManager.is_authenticated():
            user = SessionManager.get_user()
            print(f"✅ Already logged in as {user['name']} ({user['role']})")
            self.show_main()
        else:
            self.show_login()

        sys.exit(self.app.exec())

if __name__ == "__main__":
    controller = AppController()
    controller.run()
