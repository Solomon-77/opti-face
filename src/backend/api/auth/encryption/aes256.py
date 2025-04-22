import os
import base64
import json
from dotenv import load_dotenv
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

load_dotenv()

AES_KEY = bytes.fromhex(os.getenv("AES_KEY_HEX"))

def generate_iv():
    return os.urandom(16)

def pad_data(data: bytes) -> bytes:
    padder = padding.PKCS7(128).padder()
    return padder.update(data) + padder.finalize()

def unpad_data(data: bytes) -> bytes:
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(data) + unpadder.finalize()

def encrypt_data(data: dict) -> str:
    iv = generate_iv()
    data_bytes = json.dumps(data).encode()
    padded_data = pad_data(data_bytes)

    cipher = Cipher(algorithms.AES(AES_KEY), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()

    return base64.b64encode(iv + encrypted).decode()

def decrypt_data(encoded_data: str) -> dict:
    raw_data = base64.b64decode(encoded_data)
    iv = raw_data[:16]
    encrypted_data = raw_data[16:]

    cipher = Cipher(algorithms.AES(AES_KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
    return json.loads(unpad_data(decrypted_padded).decode())
