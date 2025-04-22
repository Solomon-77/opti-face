from backend.api.db.db_connection import connect_to_db
from backend.api.auth.hashing.argon2id_functions import verify_password
from backend.api.auth.session_manager import SessionManager 

def verify_auth_credentials(name: str, pw: str) -> bool:
    conn = connect_to_db()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT salt, hashed_password, role FROM users WHERE name = %s", (name,))
            result = cur.fetchone()
            if not result:
                return False

            salt, stored_hash, role = result

            if verify_password(stored_hash, pw, salt):
                SessionManager.set_user(name, role)
                return True
            else:
                return False
    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        return False
    finally:
        conn.close()
