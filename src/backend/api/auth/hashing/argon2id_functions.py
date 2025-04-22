import os
import base64
from argon2 import PasswordHasher, Type, exceptions

ph = PasswordHasher(
    time_cost=3,       # t=3
    memory_cost=65536, # m=65536 (64 MiB)
    parallelism=4,     # p=4
    type=Type.ID       # Argon2id
)

def generate_salt(length=8) -> str:
    return base64.b64encode(os.urandom(length)).decode('utf-8')[:length]

def hash_password(password: str, salt: str) -> str:
    combined = f"{salt}{password}"
    return ph.hash(combined)

def verify_password(stored_hash: str, input_password: str, salt: str) -> bool:
    try:
        combined = f"{salt}{input_password}"
        return ph.verify(stored_hash, combined)
    except exceptions.VerifyMismatchError:
        return False
    except Exception as e:
        print(f"[Argon2id ERROR] {e}")
        return False
