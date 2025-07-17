"""Security utilities for credential management and data encryption."""

import os
import json
import base64
import hashlib
import secrets
from typing import Dict, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import logging


logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


@dataclass
class CredentialMetadata:
    """Metadata for stored credentials."""
    created_at: datetime
    expires_at: Optional[datetime] = None
    rotation_interval_days: Optional[int] = None
    last_rotated: Optional[datetime] = None
    rotation_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def needs_rotation(self) -> bool:
        """Check if credential needs rotation based on interval."""
        if self.rotation_interval_days is None:
            return False
        
        if self.last_rotated is None:
            rotation_due = self.created_at + timedelta(days=self.rotation_interval_days)
        else:
            rotation_due = self.last_rotated + timedelta(days=self.rotation_interval_days)
        
        return datetime.utcnow() > rotation_due


@dataclass
class EncryptedCredential:
    """Container for encrypted credential data."""
    encrypted_data: str
    metadata: CredentialMetadata
    key_id: str
    algorithm: str = "fernet"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "encrypted_data": self.encrypted_data,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "expires_at": self.metadata.expires_at.isoformat() if self.metadata.expires_at else None,
                "rotation_interval_days": self.metadata.rotation_interval_days,
                "last_rotated": self.metadata.last_rotated.isoformat() if self.metadata.last_rotated else None,
                "rotation_count": self.metadata.rotation_count,
            },
            "key_id": self.key_id,
            "algorithm": self.algorithm,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedCredential':
        """Create from dictionary."""
        metadata_dict = data["metadata"]
        metadata = CredentialMetadata(
            created_at=datetime.fromisoformat(metadata_dict["created_at"]),
            expires_at=datetime.fromisoformat(metadata_dict["expires_at"]) if metadata_dict["expires_at"] else None,
            rotation_interval_days=metadata_dict.get("rotation_interval_days"),
            last_rotated=datetime.fromisoformat(metadata_dict["last_rotated"]) if metadata_dict.get("last_rotated") else None,
            rotation_count=metadata_dict.get("rotation_count", 0),
        )
        
        return cls(
            encrypted_data=data["encrypted_data"],
            metadata=metadata,
            key_id=data["key_id"],
            algorithm=data.get("algorithm", "fernet"),
        )


class EncryptionManager:
    """Manages encryption keys and operations."""
    
    def __init__(self, key_storage_path: str = ".keys"):
        """
        Initialize encryption manager.
        
        Args:
            key_storage_path: Directory to store encryption keys
        """
        self.key_storage_path = Path(key_storage_path)
        self.key_storage_path.mkdir(mode=0o700, exist_ok=True)
        self._keys: Dict[str, bytes] = {}
        self._master_key: Optional[bytes] = None
    
    def _get_master_key(self) -> bytes:
        """Get or create master encryption key."""
        if self._master_key is not None:
            return self._master_key
        
        master_key_file = self.key_storage_path / "master.key"
        
        if master_key_file.exists():
            # Load existing master key
            try:
                with open(master_key_file, "rb") as f:
                    self._master_key = f.read()
                logger.info("Loaded existing master encryption key")
            except Exception as e:
                raise SecurityError(f"Failed to load master key: {e}")
        else:
            # Generate new master key
            self._master_key = Fernet.generate_key()
            try:
                with open(master_key_file, "wb") as f:
                    f.write(self._master_key)
                master_key_file.chmod(0o600)
                logger.info("Generated new master encryption key")
            except Exception as e:
                raise SecurityError(f"Failed to save master key: {e}")
        
        return self._master_key
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def generate_key_id(self) -> str:
        """Generate a unique key identifier."""
        return secrets.token_hex(16)
    
    def create_encryption_key(self, key_id: Optional[str] = None) -> str:
        """Create a new encryption key and return its ID."""
        if key_id is None:
            key_id = self.generate_key_id()
        
        # Generate new Fernet key
        key = Fernet.generate_key()
        self._keys[key_id] = key
        
        # Store key encrypted with master key
        master_key = self._get_master_key()
        fernet = Fernet(master_key)
        encrypted_key = fernet.encrypt(key)
        
        key_file = self.key_storage_path / f"{key_id}.key"
        try:
            with open(key_file, "wb") as f:
                f.write(encrypted_key)
            key_file.chmod(0o600)
            logger.info(f"Created encryption key: {key_id}")
        except Exception as e:
            raise SecurityError(f"Failed to store encryption key {key_id}: {e}")
        
        return key_id
    
    def get_encryption_key(self, key_id: str) -> bytes:
        """Get encryption key by ID."""
        if key_id in self._keys:
            return self._keys[key_id]
        
        key_file = self.key_storage_path / f"{key_id}.key"
        if not key_file.exists():
            raise SecurityError(f"Encryption key not found: {key_id}")
        
        try:
            master_key = self._get_master_key()
            fernet = Fernet(master_key)
            
            with open(key_file, "rb") as f:
                encrypted_key = f.read()
            
            key = fernet.decrypt(encrypted_key)
            self._keys[key_id] = key
            return key
        except Exception as e:
            raise SecurityError(f"Failed to load encryption key {key_id}: {e}")
    
    def encrypt_data(self, data: Union[str, bytes], key_id: str) -> str:
        """Encrypt data using specified key."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        key = self.get_encryption_key(key_id)
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str, key_id: str) -> bytes:
        """Decrypt data using specified key."""
        key = self.get_encryption_key(key_id)
        fernet = Fernet(key)
        
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        return fernet.decrypt(encrypted_bytes)
    
    def rotate_key(self, old_key_id: str) -> str:
        """Rotate encryption key and return new key ID."""
        new_key_id = self.create_encryption_key()
        
        # Mark old key for deletion (but don't delete immediately for safety)
        old_key_file = self.key_storage_path / f"{old_key_id}.key"
        if old_key_file.exists():
            backup_file = self.key_storage_path / f"{old_key_id}.key.backup"
            old_key_file.rename(backup_file)
            logger.info(f"Rotated encryption key from {old_key_id} to {new_key_id}")
        
        return new_key_id


class CredentialManager:
    """Manages secure storage and rotation of API credentials."""
    
    def __init__(self, storage_path: str = ".credentials", encryption_manager: Optional[EncryptionManager] = None):
        """
        Initialize credential manager.
        
        Args:
            storage_path: Directory to store encrypted credentials
            encryption_manager: Optional encryption manager instance
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(mode=0o700, exist_ok=True)
        self.encryption_manager = encryption_manager or EncryptionManager()
        self._credentials: Dict[str, EncryptedCredential] = {}
    
    def store_credential(
        self,
        name: str,
        value: str,
        expires_at: Optional[datetime] = None,
        rotation_interval_days: Optional[int] = None
    ) -> None:
        """
        Store encrypted credential.
        
        Args:
            name: Credential name/identifier
            value: Credential value to encrypt
            expires_at: Optional expiration datetime
            rotation_interval_days: Optional rotation interval in days
        """
        # Create encryption key for this credential
        key_id = self.encryption_manager.create_encryption_key()
        
        # Encrypt the credential value
        encrypted_data = self.encryption_manager.encrypt_data(value, key_id)
        
        # Create metadata
        metadata = CredentialMetadata(
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            rotation_interval_days=rotation_interval_days,
        )
        
        # Create encrypted credential
        credential = EncryptedCredential(
            encrypted_data=encrypted_data,
            metadata=metadata,
            key_id=key_id,
        )
        
        # Store to file
        self._save_credential(name, credential)
        self._credentials[name] = credential
        
        logger.info(f"Stored encrypted credential: {name}")
    
    def get_credential(self, name: str) -> Optional[str]:
        """
        Retrieve and decrypt credential.
        
        Args:
            name: Credential name/identifier
            
        Returns:
            Decrypted credential value or None if not found
        """
        credential = self._load_credential(name)
        if credential is None:
            return None
        
        # Check if credential has expired
        if credential.metadata.is_expired():
            logger.warning(f"Credential {name} has expired")
            return None
        
        try:
            decrypted_data = self.encryption_manager.decrypt_data(
                credential.encrypted_data,
                credential.key_id
            )
            return decrypted_data.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decrypt credential {name}: {e}")
            return None
    
    def rotate_credential(self, name: str, new_value: str) -> bool:
        """
        Rotate credential with new value.
        
        Args:
            name: Credential name/identifier
            new_value: New credential value
            
        Returns:
            True if rotation successful, False otherwise
        """
        credential = self._load_credential(name)
        if credential is None:
            logger.error(f"Cannot rotate non-existent credential: {name}")
            return False
        
        try:
            # Create new encryption key
            new_key_id = self.encryption_manager.create_encryption_key()
            
            # Encrypt new value
            encrypted_data = self.encryption_manager.encrypt_data(new_value, new_key_id)
            
            # Update metadata
            credential.metadata.last_rotated = datetime.utcnow()
            credential.metadata.rotation_count += 1
            
            # Update credential
            credential.encrypted_data = encrypted_data
            credential.key_id = new_key_id
            
            # Save updated credential
            self._save_credential(name, credential)
            self._credentials[name] = credential
            
            logger.info(f"Rotated credential: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate credential {name}: {e}")
            return False
    
    def check_rotation_needed(self, name: str) -> bool:
        """Check if credential needs rotation."""
        credential = self._load_credential(name)
        if credential is None:
            return False
        
        return credential.metadata.needs_rotation()
    
    def list_credentials(self) -> Dict[str, CredentialMetadata]:
        """List all stored credentials with their metadata."""
        credentials = {}
        
        for credential_file in self.storage_path.glob("*.json"):
            name = credential_file.stem
            credential = self._load_credential(name)
            if credential:
                credentials[name] = credential.metadata
        
        return credentials
    
    def delete_credential(self, name: str) -> bool:
        """Delete stored credential."""
        credential_file = self.storage_path / f"{name}.json"
        
        if credential_file.exists():
            try:
                credential_file.unlink()
                if name in self._credentials:
                    del self._credentials[name]
                logger.info(f"Deleted credential: {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete credential {name}: {e}")
                return False
        
        return False
    
    def _load_credential(self, name: str) -> Optional[EncryptedCredential]:
        """Load credential from storage."""
        if name in self._credentials:
            return self._credentials[name]
        
        credential_file = self.storage_path / f"{name}.json"
        if not credential_file.exists():
            return None
        
        try:
            with open(credential_file, 'r') as f:
                data = json.load(f)
            
            credential = EncryptedCredential.from_dict(data)
            self._credentials[name] = credential
            return credential
            
        except Exception as e:
            logger.error(f"Failed to load credential {name}: {e}")
            return None
    
    def _save_credential(self, name: str, credential: EncryptedCredential) -> None:
        """Save credential to storage."""
        credential_file = self.storage_path / f"{name}.json"
        
        try:
            with open(credential_file, 'w') as f:
                json.dump(credential.to_dict(), f, indent=2)
            credential_file.chmod(0o600)
        except Exception as e:
            raise SecurityError(f"Failed to save credential {name}: {e}")


class SecureAPIClient:
    """Base class for secure API communication."""
    
    def __init__(self, credential_manager: CredentialManager):
        """
        Initialize secure API client.
        
        Args:
            credential_manager: Credential manager instance
        """
        self.credential_manager = credential_manager
        self._session = None
    
    def get_secure_headers(self, api_key_name: str, secret_key_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get secure headers for API requests.
        
        Args:
            api_key_name: Name of API key credential
            secret_key_name: Optional name of secret key credential
            
        Returns:
            Dictionary of headers for secure API communication
        """
        headers = {
            "User-Agent": "PortfolioRebalancer/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        # Get API key
        api_key = self.credential_manager.get_credential(api_key_name)
        if api_key is None:
            raise SecurityError(f"API key not found: {api_key_name}")
        
        # Get secret key if specified
        secret_key = None
        if secret_key_name:
            secret_key = self.credential_manager.get_credential(secret_key_name)
            if secret_key is None:
                raise SecurityError(f"Secret key not found: {secret_key_name}")
        
        # Add authentication headers (implementation depends on API)
        if secret_key:
            # For APIs that use both API key and secret (like Alpaca)
            headers.update({
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            })
        else:
            # For APIs that use only API key
            headers["Authorization"] = f"Bearer {api_key}"
        
        return headers
    
    def verify_ssl_context(self) -> bool:
        """Verify SSL context for secure communication."""
        # This would implement SSL certificate verification
        # For now, return True (assuming proper SSL verification)
        return True


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
    """
    Create a secure hash of sensitive data for logging/comparison.
    
    Args:
        data: Sensitive data to hash
        salt: Optional salt for hashing
        
    Returns:
        Hexadecimal hash string
    """
    if salt is None:
        salt = "portfolio_rebalancer_default_salt"
    
    hash_input = f"{data}{salt}".encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()[:16]  # First 16 chars for brevity


def mask_sensitive_value(value: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """
    Mask sensitive value for safe logging.
    
    Args:
        value: Sensitive value to mask
        mask_char: Character to use for masking
        visible_chars: Number of characters to leave visible at the end
        
    Returns:
        Masked value string
    """
    if len(value) <= visible_chars:
        return mask_char * len(value)
    
    masked_length = len(value) - visible_chars
    return mask_char * masked_length + value[-visible_chars:]


# Global instances
_encryption_manager = None
_credential_manager = None


def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager instance."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def get_credential_manager() -> CredentialManager:
    """Get global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager(encryption_manager=get_encryption_manager())
    return _credential_manager