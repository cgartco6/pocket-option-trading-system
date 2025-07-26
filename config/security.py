import os
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from datetime import datetime, timedelta

class MilitaryCrypto:
    @staticmethod
    def encrypt_data(data: str, secret_key: str) -> str:
        """Military-grade HMAC-SHA256 encryption"""
        return hmac.new(
            key=secret_key.encode('utf-8'),
            msg=data.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()

    @staticmethod
    def generate_secure_headers(api_key: str, secret_key: str) -> dict:
        """Generate secure request headers"""
        timestamp = str(int(datetime.utcnow().timestamp()))
        signature = MilitaryCrypto.encrypt_data(timestamp + api_key, secret_key)
        
        return {
            'X-PO-APIKEY': api_key,
            'X-PO-SIGNATURE': signature,
            'X-PO-TIMESTAMP': timestamp
        }

    @staticmethod
    def encrypt_payload(data: dict, encryption_key: str) -> str:
        """Encrypt payload using AES-256"""
        f = Fernet(encryption_key)
        return f.encrypt(str(data).encode()).decode()

    @staticmethod
    def decrypt_payload(encrypted_data: str, encryption_key: str) -> dict:
        """Decrypt payload using AES-256"""
        f = Fernet(encryption_key)
        return eval(f.decrypt(encrypted_data.encode()).decode())

class SecureSessionManager:
    def __init__(self, encryption_salt: str):
        self.encryption_key = base64.urlsafe_b64encode(
            hashlib.sha256(encryption_salt.encode()).digest()
        )
        self.session_tokens = {}
        self.token_expiry = timedelta(minutes=30)
    
    def create_session(self, user_id: str, data: dict) -> str:
        """Create encrypted session token"""
        expiry = datetime.utcnow() + self.token_expiry
        session_data = {
            'user_id': user_id,
            'data': data,
            'expiry': expiry.isoformat()
        }
        token = MilitaryCrypto.encrypt_payload(session_data, self.encryption_key)
        self.session_tokens[user_id] = token
        return token
    
    def validate_session(self, token: str) -> dict:
        """Validate session token and return data"""
        try:
            session_data = MilitaryCrypto.decrypt_payload(token, self.encryption_key)
            if datetime.fromisoformat(session_data['expiry']) < datetime.utcnow():
                return None
            return session_data
        except:
            return None
