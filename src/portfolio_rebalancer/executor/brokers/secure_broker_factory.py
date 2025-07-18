"""Secure broker factory with encrypted credential management."""

import logging
from typing import Optional, Dict, Any
from ...common.config import Config, BrokerType
from ...common.security import get_credential_manager, SecureAPIClient, SecurityError
from .base_broker import BrokerInterface
from .alpaca_broker import AlpacaBroker
from .ib_broker import IBBroker


logger = logging.getLogger(__name__)


class SecureBrokerFactory:
    """Factory for creating broker instances with secure credential management."""
    
    def __init__(self, config: object):
        """Initialize secure broker factory."""
        self.config = config
        from ...common.security import EncryptionManager, CredentialManager
        self.encryption_manager = EncryptionManager(
            key_storage_path=config.security.key_storage_path
        )
        self.credential_manager = CredentialManager(
            storage_path=config.security.credential_storage_path,
            encryption_manager=self.encryption_manager
        )
        self._setup_credentials()
    
    def _setup_credentials(self) -> None:
        """Set up encrypted credentials if enabled."""
        if not self.config.broker.use_encrypted_credentials:
            logger.info("Encrypted credentials disabled, using environment variables")
            return
        
        if not self.config.security.enable_encryption:
            logger.warning("Encryption disabled but encrypted credentials requested")
            return
        
        # Store credentials if they exist in config but not in encrypted storage
        self._migrate_credentials_if_needed()
    
    def _migrate_credentials_if_needed(self) -> None:
        """Migrate plain text credentials to encrypted storage."""
        try:
            # Check Alpaca credentials
            if (self.config.broker.alpaca_api_key and 
                self.config.broker.alpaca_secret_key and
                not self.credential_manager.get_credential("alpaca_api_key")):
                
                logger.info("Migrating Alpaca credentials to encrypted storage")
                self.credential_manager.store_credential(
                    "alpaca_api_key",
                    self.config.broker.alpaca_api_key,
                    rotation_interval_days=self.config.security.credential_rotation_days
                )
                self.credential_manager.store_credential(
                    "alpaca_secret_key", 
                    self.config.broker.alpaca_secret_key,
                    rotation_interval_days=self.config.security.credential_rotation_days
                )
                logger.info("Alpaca credentials migrated successfully")
        
        except Exception as e:
            logger.error(f"Failed to migrate credentials: {e}")
            raise SecurityError(f"Credential migration failed: {e}")
    
    def create_broker(self, broker_type: Optional[str] = None) -> Optional[BrokerInterface]:
        """
        Create broker instance with secure credentials. Returns None on error.
        """
        try:
            if broker_type is None:
                broker_type = self.config.executor.broker_type
            logger.info(f"Creating secure broker instance: {broker_type}")
            if broker_type == BrokerType.ALPACA.value:
                return self._create_alpaca_broker()
            elif broker_type == BrokerType.IB.value:
                return self._create_ib_broker()
            else:
                logger.error(f"Unsupported broker type: {broker_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create broker: {e}")
            return None
    
    def _create_alpaca_broker(self) -> Optional[AlpacaBroker]:
        """Create Alpaca broker with secure credentials. Returns None on error."""
        try:
            if self.config.broker.use_encrypted_credentials and self.config.security.enable_encryption:
                api_key = self.credential_manager.get_credential("alpaca_api_key")
                secret_key = self.credential_manager.get_credential("alpaca_secret_key")
                if not api_key or not secret_key:
                    logger.error("Alpaca encrypted credentials not found")
                    return None
                secure_config = self._create_secure_config_copy()
                secure_config.broker.alpaca_api_key = api_key
                secure_config.broker.alpaca_secret_key = secret_key
                logger.info("Using encrypted Alpaca credentials")
            else:
                secure_config = self.config
                logger.info("Using plain text Alpaca credentials")
            return AlpacaBroker(secure_config)
        except Exception as e:
            logger.error(f"Failed to create Alpaca broker: {e}")
            return None
    
    def _create_ib_broker(self) -> Optional[IBBroker]:
        """Create Interactive Brokers broker instance. Returns None on error."""
        try:
            secure_config = self._create_secure_config_copy()
            return IBBroker(secure_config)
        except Exception as e:
            logger.error(f"Failed to create IB broker: {e}")
            return None
    
    def _create_secure_config_copy(self) -> Config:
        """Create a copy of config for secure credential injection."""
        # Create a shallow copy to avoid modifying the original
        import copy
        return copy.deepcopy(self.config)
    
    def check_credential_rotation(self) -> Dict[str, bool]:
        """
        Check if any credentials need rotation.
        
        Returns:
            Dictionary mapping credential names to rotation needed status
        """
        if not self.config.broker.use_encrypted_credentials:
            return {}
        
        rotation_status = {}
        
        try:
            # Check Alpaca credentials
            if self.credential_manager.get_credential("alpaca_api_key"):
                rotation_status["alpaca_api_key"] = self.credential_manager.check_rotation_needed("alpaca_api_key")
                rotation_status["alpaca_secret_key"] = self.credential_manager.check_rotation_needed("alpaca_secret_key")
        
        except Exception as e:
            logger.error(f"Failed to check credential rotation status: {e}")
        
        return rotation_status
    
    def rotate_credentials(self, credential_name: str, new_value: str) -> bool:
        """
        Rotate a specific credential.
        
        Args:
            credential_name: Name of credential to rotate
            new_value: New credential value
            
        Returns:
            True if rotation successful, False otherwise
        """
        if not self.config.broker.use_encrypted_credentials:
            logger.warning("Credential rotation requested but encrypted credentials disabled")
            return False
        
        try:
            success = self.credential_manager.rotate_credential(credential_name, new_value)
            if success:
                logger.info(f"Successfully rotated credential: {credential_name}")
            else:
                logger.error(f"Failed to rotate credential: {credential_name}")
            return success
        
        except Exception as e:
            logger.error(f"Error rotating credential {credential_name}: {e}")
            return False
    
    def get_credential_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all stored credentials.
        
        Returns:
            Dictionary with credential metadata
        """
        if not self.config.broker.use_encrypted_credentials:
            return {}
        
        try:
            credentials = self.credential_manager.list_credentials()
            status = {}
            
            for name, metadata in credentials.items():
                status[name] = {
                    "created_at": metadata.created_at.isoformat(),
                    "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                    "rotation_count": metadata.rotation_count,
                    "needs_rotation": metadata.needs_rotation(),
                    "is_expired": metadata.is_expired(),
                }
            
            return status
        
        except Exception as e:
            logger.error(f"Failed to get credential status: {e}")
            return {}


class SecureAlpacaClient(SecureAPIClient):
    """Secure Alpaca API client with encrypted credentials."""
    
    def __init__(self, credential_manager):
        """Initialize secure Alpaca client."""
        super().__init__(credential_manager)
    
    def get_alpaca_headers(self) -> Dict[str, str]:
        """Get secure headers for Alpaca API requests."""
        return self.get_secure_headers("alpaca_api_key", "alpaca_secret_key")
    
    def make_secure_request(self, method: str, url: str, **kwargs) -> Any:
        """
        Make a secure API request with proper headers and SSL verification.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Response data
        """
        import requests
        
        # Get secure headers
        headers = self.get_alpaca_headers()
        
        # Merge with any provided headers
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        kwargs['headers'] = headers
        
        # Ensure SSL verification
        kwargs.setdefault('verify', self.verify_ssl_context())
        
        # Set timeout if not provided
        kwargs.setdefault('timeout', 30)
        
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else None
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Secure API request failed: {e}")
            raise SecurityError(f"API request failed: {e}")


def create_secure_broker(config: Config, broker_type: Optional[str] = None) -> BrokerInterface:
    """
    Convenience function to create a secure broker instance.
    
    Args:
        config: Application configuration
        broker_type: Optional broker type override
        
    Returns:
        Configured broker instance
    """
    factory = SecureBrokerFactory(config)
    return factory.create_broker(broker_type)