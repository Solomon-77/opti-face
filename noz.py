import numpy as np

def print_npz(file_path):
    try:
        data = np.load(file_path)
        print(f"Contents of '{file_path}':\n")
        for key in data.files:
            print(f"Key: {key}")
            print(f"Value:\n{data[key]}")
            print("-" * 40)
    except Exception as e:
        print(f"Failed to load '{file_path}': {e}")

print_npz('./src/backend/face_database/Ethan.npz')
