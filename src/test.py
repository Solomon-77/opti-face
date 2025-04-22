from backend.api.auth.auth_functons import verify_auth_credentials
from backend.api.auth.session_manager import SessionManager

def login():
    print("🔑 Please log in to continue")
    username = input("Username: ")
    password = input("Password: ")
    
    if verify_auth_credentials(username, password):
        print("✅ Login successful")
        
        user = SessionManager.get_user()
        if user:
            print(f"Current user: {user['name']}")
            print(f"Role: {user['role']}")
        else:
            print("❌ No active session found.")
    else:
        print("❌ Invalid credentials")

def already_logged_in():
    if SessionManager.is_authenticated():
        user = SessionManager.get_user()
        print("✅ You are already logged in as")
        print(f"Current user: {user['name']}")
        print(f"Role: {user['role']}")

        while True:
            print("\nOptions:")
            print("1. Log out")
            print("2. Quit")
            choice = input("Select an option: ")

            if choice == "1":
                SessionManager.clear_user()
                print("✅ Logged out successfully")
                break
            elif choice == "2":
                print("Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
    else:
        login()

if __name__ == "__main__":
    already_logged_in()
