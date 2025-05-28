import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import hmac
import base64
import os
import json
import secrets
from dataclasses import dataclass
from enum import Enum
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt

class SecurityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class SecurityConfig:
    """Configuratie voor security module."""
    encryption_key: str = ""
    salt: str = ""
    iterations: int = 100000
    key_length: int = 32
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    max_login_attempts: int = 5
    lockout_duration: int = 300  # 5 minuten
    session_timeout: int = 3600  # 1 uur
    min_password_length: int = 12
    require_special_chars: bool = True
    require_numbers: bool = True
    require_uppercase: bool = True
    require_lowercase: bool = True

class SecurityManager:
    """Beheert security functionaliteit."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or SecurityConfig()
        self.lock = threading.Lock()
        self.login_attempts = {}
        self.sessions = {}
        
        # Initialiseer encryption
        if self.config.encryption_key:
            self._init_encryption()
            
    def _init_encryption(self):
        """Initialiseer encryption."""
        try:
            # Genereer salt als niet geconfigureerd
            if not self.config.salt:
                self.config.salt = secrets.token_hex(16)
                
            # Genereer encryption key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.config.key_length,
                salt=self.config.salt.encode(),
                iterations=self.config.iterations
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.config.encryption_key.encode()))
            self.fernet = Fernet(key)
            
        except Exception as e:
            self.logger.error(f"Fout bij initialiseren encryption: {e}")
            raise
            
    def hash_password(self, password: str) -> str:
        """Hash een wachtwoord."""
        try:
            salt = bcrypt.gensalt()
            return bcrypt.hashpw(password.encode(), salt).decode()
            
        except Exception as e:
            self.logger.error(f"Fout bij hashen wachtwoord: {e}")
            raise
            
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verifieer een wachtwoord."""
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
            
        except Exception as e:
            self.logger.error(f"Fout bij verifiëren wachtwoord: {e}")
            return False
            
    def validate_password(self, password: str) -> Tuple[bool, str]:
        """Valideer een wachtwoord."""
        try:
            if len(password) < self.config.min_password_length:
                return False, f"Wachtwoord moet minimaal {self.config.min_password_length} karakters lang zijn"
                
            if self.config.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                return False, "Wachtwoord moet speciale karakters bevatten"
                
            if self.config.require_numbers and not any(c.isdigit() for c in password):
                return False, "Wachtwoord moet nummers bevatten"
                
            if self.config.require_uppercase and not any(c.isupper() for c in password):
                return False, "Wachtwoord moet hoofdletters bevatten"
                
            if self.config.require_lowercase and not any(c.islower() for c in password):
                return False, "Wachtwoord moet kleine letters bevatten"
                
            return True, "Wachtwoord geldig"
            
        except Exception as e:
            self.logger.error(f"Fout bij valideren wachtwoord: {e}")
            return False, f"Validatie fout: {e}"
            
    def generate_api_key(self) -> Tuple[str, str]:
        """Genereer API key en secret."""
        try:
            api_key = secrets.token_hex(16)
            api_secret = secrets.token_hex(32)
            return api_key, api_secret
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren API key: {e}")
            raise
            
    def encrypt_data(self, data: str) -> str:
        """Encrypt data."""
        try:
            return self.fernet.encrypt(data.encode()).decode()
            
        except Exception as e:
            self.logger.error(f"Fout bij encrypten data: {e}")
            raise
            
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data."""
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
            
        except Exception as e:
            self.logger.error(f"Fout bij decrypten data: {e}")
            raise
            
    def generate_session_token(self) -> str:
        """Genereer session token."""
        try:
            return secrets.token_urlsafe(32)
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren session token: {e}")
            raise
            
    def create_session(self, user_id: str) -> str:
        """Maak een nieuwe session."""
        try:
            token = self.generate_session_token()
            self.sessions[token] = {
                "user_id": user_id,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + datetime.timedelta(seconds=self.config.session_timeout)
            }
            return token
            
        except Exception as e:
            self.logger.error(f"Fout bij maken session: {e}")
            raise
            
    def validate_session(self, token: str) -> bool:
        """Valideer een session."""
        try:
            if token not in self.sessions:
                return False
                
            session = self.sessions[token]
            if datetime.now() > session["expires_at"]:
                del self.sessions[token]
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Fout bij valideren session: {e}")
            return False
            
    def invalidate_session(self, token: str):
        """Invalidate een session."""
        try:
            if token in self.sessions:
                del self.sessions[token]
                
        except Exception as e:
            self.logger.error(f"Fout bij invalidaten session: {e}")
            
    def check_login_attempts(self, user_id: str) -> bool:
        """Check login attempts."""
        try:
            with self.lock:
                if user_id not in self.login_attempts:
                    return True
                    
                attempts = self.login_attempts[user_id]
                if attempts["count"] >= self.config.max_login_attempts:
                    if datetime.now().timestamp() - attempts["timestamp"] < self.config.lockout_duration:
                        return False
                    else:
                        del self.login_attempts[user_id]
                        return True
                        
                return True
                
        except Exception as e:
            self.logger.error(f"Fout bij checken login attempts: {e}")
            return False
            
    def record_login_attempt(self, user_id: str, success: bool):
        """Record login attempt."""
        try:
            with self.lock:
                if success:
                    if user_id in self.login_attempts:
                        del self.login_attempts[user_id]
                else:
                    if user_id not in self.login_attempts:
                        self.login_attempts[user_id] = {
                            "count": 0,
                            "timestamp": datetime.now().timestamp()
                        }
                    self.login_attempts[user_id]["count"] += 1
                    self.login_attempts[user_id]["timestamp"] = datetime.now().timestamp()
                    
        except Exception as e:
            self.logger.error(f"Fout bij recorden login attempt: {e}")
            
    def generate_2fa_secret(self) -> str:
        """Genereer 2FA secret."""
        try:
            return base64.b32encode(os.urandom(20)).decode()
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren 2FA secret: {e}")
            raise
            
    def verify_2fa_code(self, secret: str, code: str) -> bool:
        """Verifieer 2FA code."""
        try:
            # Implementeer TOTP verificatie
            # Dit is een vereenvoudigde versie
            current_time = int(datetime.now().timestamp() / 30)
            for i in range(-1, 2):
                expected_code = self._generate_totp(secret, current_time + i)
                if code == expected_code:
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Fout bij verifiëren 2FA code: {e}")
            return False
            
    def _generate_totp(self, secret: str, counter: int) -> str:
        """Genereer TOTP code."""
        try:
            key = base64.b32decode(secret)
            counter_bytes = counter.to_bytes(8, byteorder="big")
            hmac_obj = hmac.new(key, counter_bytes, hashlib.sha1)
            hmac_result = hmac_obj.digest()
            offset = hmac_result[-1] & 0x0F
            code = ((hmac_result[offset] & 0x7F) << 24 |
                   (hmac_result[offset + 1] & 0xFF) << 16 |
                   (hmac_result[offset + 2] & 0xFF) << 8 |
                   (hmac_result[offset + 3] & 0xFF))
            code = code % 1000000
            return f"{code:06d}"
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren TOTP: {e}")
            raise
            
    def sanitize_input(self, input_str: str) -> str:
        """Sanitize user input."""
        try:
            # Verwijder gevaarlijke karakters
            sanitized = input_str.replace("<", "&lt;").replace(">", "&gt;")
            sanitized = sanitized.replace("'", "&#39;").replace('"', "&quot;")
            return sanitized
            
        except Exception as e:
            self.logger.error(f"Fout bij sanitizen input: {e}")
            return input_str
            
    def validate_api_request(self, api_key: str, signature: str, timestamp: int, params: Dict[str, Any]) -> bool:
        """Valideer API request."""
        try:
            # Check timestamp
            if abs(int(datetime.now().timestamp() * 1000) - timestamp) > 5000:
                return False
                
            # Genereer signature
            query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            expected_signature = hmac.new(
                api_key.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"Fout bij valideren API request: {e}")
            return False
            
    def generate_csrf_token(self) -> str:
        """Genereer CSRF token."""
        try:
            return secrets.token_hex(32)
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren CSRF token: {e}")
            raise
            
    def validate_csrf_token(self, token: str, stored_token: str) -> bool:
        """Valideer CSRF token."""
        try:
            return hmac.compare_digest(token, stored_token)
            
        except Exception as e:
            self.logger.error(f"Fout bij valideren CSRF token: {e}")
            return False
            
    def encrypt_file(self, file_path: str) -> bool:
        """Encrypt een bestand."""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
                
            encrypted_data = self.fernet.encrypt(data)
            
            with open(file_path + ".enc", "wb") as f:
                f.write(encrypted_data)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Fout bij encrypten bestand: {e}")
            return False
            
    def decrypt_file(self, file_path: str) -> bool:
        """Decrypt een bestand."""
        try:
            with open(file_path, "rb") as f:
                encrypted_data = f.read()
                
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            with open(file_path[:-4], "wb") as f:
                f.write(decrypted_data)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Fout bij decrypten bestand: {e}")
            return False
            
    def secure_delete_file(self, file_path: str) -> bool:
        """Veilig verwijder een bestand."""
        try:
            if os.path.exists(file_path):
                # Overschrijf met random data
                with open(file_path, "wb") as f:
                    f.write(os.urandom(os.path.getsize(file_path)))
                    
                # Verwijder bestand
                os.remove(file_path)
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Fout bij veilig verwijderen bestand: {e}")
            return False
            
    def generate_secure_password(self) -> str:
        """Genereer een veilig wachtwoord."""
        try:
            # Genereer random wachtwoord
            password = secrets.token_urlsafe(16)
            
            # Zorg dat het voldoet aan requirements
            while not self.validate_password(password)[0]:
                password = secrets.token_urlsafe(16)
                
            return password
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren veilig wachtwoord: {e}")
            raise 