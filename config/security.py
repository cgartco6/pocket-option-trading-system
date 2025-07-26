import os
from cryptography.fernet import Fernet

class CryptoManager:
    def __init__(self):
        self.key = os.getenv('ENCRYPTION_KEY')
        if not self.key:
            raise ValueError("Encryption key not found in environment")
        self.cipher = Fernet(self.key.encode())
    
    def encrypt(self, data):
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data):
        return self.cipher.decrypt(encrypted_data.encode()).decode()

def secure_credentials():
    crypto = CryptoManager()
    return {
        'BINANCE_API_KEY': crypto.decrypt(os.getenv('ENC_BINANCE_API_KEY')),
        'BINANCE_SECRET': crypto.decrypt(os.getenv('ENC_BINANCE_SECRET')),
        'PO_API_KEY': crypto.decrypt(os.getenv('ENC_PO_API_KEY')),
        'PO_API_SECRET': crypto.decrypt(os.getenv('ENC_PO_API_SECRET'))
    }
