import os
from backend.api.auth.encryption.aes256 import encrypt_data, decrypt_data

SESSION_FILE = os.path.join(os.path.expanduser("~"), ".app_session.json")

class SessionManager:
    _current_user = None

    @classmethod
    def set_user(cls, name: str, role: str):
        cls._current_user = {"name": name, "role": role}
        cls._save_to_file()

    @classmethod
    def get_user(cls):
        if cls._current_user is None:
            cls._load_from_file()
        return cls._current_user

    @classmethod
    def is_authenticated(cls):
        return cls.get_user() is not None

    @classmethod
    def clear_user(cls):
        cls._current_user = None
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)

    @classmethod
    def _save_to_file(cls):
        if cls._current_user:
            try:
                encrypted_data = encrypt_data(cls._current_user)
                with open(SESSION_FILE, "w") as f:
                    f.write(encrypted_data)
            except Exception as e:
                print(f"[Session Save Error] {e}")

    @classmethod
    def _load_from_file(cls):
        if os.path.exists(SESSION_FILE):
            try:
                with open(SESSION_FILE, "r") as f:
                    encrypted_data = f.read()
                cls._current_user = decrypt_data(encrypted_data)
            except Exception as e:
                print(f"[Session Load Error] {e}")
                cls._current_user = None
