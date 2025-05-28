import logging
from typing import Dict, List, Any, Optional
import os
from datetime import datetime, timedelta
import json
import hashlib
import hmac
import base64
import secrets
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecurityManager:
    """Manages security for AI components."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Security settings
        self.security_dir = config.get('security_dir', 'security')
        self.secret_key = config.get('secret_key', secrets.token_hex(32))
        self.token_expiry = timedelta(hours=config.get('token_expiry_hours', 24))
        self.max_attempts = config.get('max_attempts', 5)
        self.lockout_duration = timedelta(minutes=config.get('lockout_minutes', 30))
        
        # Initialize security
        self.attempts = {}
        self.lockouts = {}
        self._load_security()
        
    def _load_security(self):
        """Load security data."""
        try:
            # Create security directory if it doesn't exist
            os.makedirs(self.security_dir, exist_ok=True)
            
            # Load security file
            security_file = os.path.join(self.security_dir, 'security.json')
            if os.path.exists(security_file):
                with open(security_file, 'r') as f:
                    data = json.load(f)
                    self.attempts = data.get('attempts', {})
                    self.lockouts = data.get('lockouts', {})
                    
        except Exception as e:
            self.logger.error(f"Error loading security data: {str(e)}")
            
    def _save_security(self):
        """Save security data."""
        try:
            # Create security directory if it doesn't exist
            os.makedirs(self.security_dir, exist_ok=True)
            
            # Save security data
            security_file = os.path.join(self.security_dir, 'security.json')
            with open(security_file, 'w') as f:
                json.dump({
                    'attempts': self.attempts,
                    'lockouts': self.lockouts
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving security data: {str(e)}")
            
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        try:
            # Generate salt
            salt = secrets.token_hex(16)
            
            # Hash password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            
            # Return salt and key
            return f"{salt}:{key.decode()}"
            
        except Exception as e:
            self.logger.error(f"Error hashing password: {str(e)}")
            return ""
            
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password."""
        try:
            # Split salt and key
            salt, key = hashed.split(':')
            
            # Hash password with salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000
            )
            new_key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            
            # Compare keys
            return hmac.compare_digest(key.encode(), new_key)
            
        except Exception as e:
            self.logger.error(f"Error verifying password: {str(e)}")
            return False
            
    def generate_token(self, data: Dict) -> str:
        """Generate a JWT token."""
        try:
            # Add expiry
            data['exp'] = datetime.utcnow() + self.token_expiry
            
            # Generate token
            return jwt.encode(data, self.secret_key, algorithm='HS256')
            
        except Exception as e:
            self.logger.error(f"Error generating token: {str(e)}")
            return ""
            
    def verify_token(self, token: str) -> Dict:
        """Verify a JWT token."""
        try:
            # Verify token
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
        except jwt.ExpiredSignatureError:
            self.logger.error("Token has expired")
            return {}
        except jwt.InvalidTokenError as e:
            self.logger.error(f"Invalid token: {str(e)}")
            return {}
        except Exception as e:
            self.logger.error(f"Error verifying token: {str(e)}")
            return {}
            
    def encrypt_data(self, data: str) -> str:
        """Encrypt data."""
        try:
            # Generate key
            key = base64.urlsafe_b64encode(hashlib.sha256(self.secret_key.encode()).digest())
            
            # Create cipher
            cipher = Fernet(key)
            
            # Encrypt data
            return cipher.encrypt(data.encode()).decode()
            
        except Exception as e:
            self.logger.error(f"Error encrypting data: {str(e)}")
            return ""
            
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data."""
        try:
            # Generate key
            key = base64.urlsafe_b64encode(hashlib.sha256(self.secret_key.encode()).digest())
            
            # Create cipher
            cipher = Fernet(key)
            
            # Decrypt data
            return cipher.decrypt(encrypted_data.encode()).decode()
            
        except Exception as e:
            self.logger.error(f"Error decrypting data: {str(e)}")
            return ""
            
    def check_attempts(self, identifier: str) -> bool:
        """Check if too many attempts."""
        try:
            # Check if locked out
            if identifier in self.lockouts:
                lockout_time = datetime.fromisoformat(self.lockouts[identifier])
                if datetime.now() < lockout_time:
                    return False
                else:
                    del self.lockouts[identifier]
                    
            # Check attempts
            if identifier in self.attempts:
                if self.attempts[identifier] >= self.max_attempts:
                    self.lockouts[identifier] = (datetime.now() + self.lockout_duration).isoformat()
                    self._save_security()
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking attempts: {str(e)}")
            return False
            
    def record_attempt(self, identifier: str, success: bool):
        """Record an attempt."""
        try:
            if success:
                # Reset attempts on success
                if identifier in self.attempts:
                    del self.attempts[identifier]
            else:
                # Increment attempts on failure
                self.attempts[identifier] = self.attempts.get(identifier, 0) + 1
                
            # Save security data
            self._save_security()
            
        except Exception as e:
            self.logger.error(f"Error recording attempt: {str(e)}")
            
    def get_security_stats(self) -> Dict:
        """Get security statistics."""
        try:
            return {
                'total_attempts': sum(self.attempts.values()),
                'active_lockouts': len(self.lockouts),
                'attempts_by_identifier': self.attempts,
                'lockouts_by_identifier': self.lockouts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting security stats: {str(e)}")
            return {}
            
    def cleanup_security_data(self):
        """Clean up old security data."""
        try:
            # Get current time
            now = datetime.now()
            
            # Cleanup lockouts
            self.lockouts = {
                k: v for k, v in self.lockouts.items()
                if datetime.fromisoformat(v) > now
            }
            
            # Save security data
            self._save_security()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up security data: {str(e)}")
            
    def export_security_data(self, filepath: str) -> bool:
        """Export security data to file."""
        try:
            # Export data
            with open(filepath, 'w') as f:
                json.dump({
                    'attempts': self.attempts,
                    'lockouts': self.lockouts,
                    'stats': self.get_security_stats()
                }, f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting security data: {str(e)}")
            return False 