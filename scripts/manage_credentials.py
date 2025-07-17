#!/usr/bin/env python3
"""
Credential management CLI tool for Portfolio Rebalancer.

This script provides a command-line interface for managing encrypted credentials
used by the portfolio rebalancer system.
"""

import os
import sys
import argparse
import getpass
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_rebalancer.common.security import (
    get_credential_manager,
    get_encryption_manager,
    SecurityError,
    mask_sensitive_value
)
from portfolio_rebalancer.common.config import get_config


def setup_credential_manager():
    """Set up credential manager with configuration."""
    try:
        config = get_config()
        
        # Override paths if provided via environment
        cred_path = os.getenv('CREDENTIAL_STORAGE_PATH', config.security.credential_storage_path)
        key_path = os.getenv('KEY_STORAGE_PATH', config.security.key_storage_path)
        
        # Ensure directories exist
        Path(cred_path).mkdir(mode=0o700, exist_ok=True)
        Path(key_path).mkdir(mode=0o700, exist_ok=True)
        
        from portfolio_rebalancer.common.security import EncryptionManager, CredentialManager
        encryption_manager = EncryptionManager(key_storage_path=key_path)
        credential_manager = CredentialManager(
            storage_path=cred_path,
            encryption_manager=encryption_manager
        )
        
        return credential_manager, config
        
    except Exception as e:
        print(f"Error setting up credential manager: {e}")
        sys.exit(1)


def store_credential(args):
    """Store a new credential."""
    credential_manager, config = setup_credential_manager()
    
    # Get credential value
    if args.value:
        value = args.value
    else:
        value = getpass.getpass(f"Enter value for credential '{args.name}': ")
    
    if not value:
        print("Error: Credential value cannot be empty")
        sys.exit(1)
    
    # Set expiration if provided
    expires_at = None
    if args.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=args.expires_days)
    
    # Set rotation interval
    rotation_days = args.rotation_days or config.security.credential_rotation_days
    
    try:
        credential_manager.store_credential(
            args.name,
            value,
            expires_at=expires_at,
            rotation_interval_days=rotation_days
        )
        print(f"Successfully stored credential: {args.name}")
        
        if expires_at:
            print(f"Expires at: {expires_at.isoformat()}")
        if rotation_days:
            print(f"Rotation interval: {rotation_days} days")
            
    except SecurityError as e:
        print(f"Error storing credential: {e}")
        sys.exit(1)


def get_credential(args):
    """Retrieve and display a credential."""
    credential_manager, _ = setup_credential_manager()
    
    try:
        value = credential_manager.get_credential(args.name)
        if value is None:
            print(f"Credential not found: {args.name}")
            sys.exit(1)
        
        if args.mask:
            print(f"{args.name}: {mask_sensitive_value(value)}")
        else:
            print(f"{args.name}: {value}")
            
    except SecurityError as e:
        print(f"Error retrieving credential: {e}")
        sys.exit(1)


def list_credentials(args):
    """List all stored credentials."""
    credential_manager, _ = setup_credential_manager()
    
    try:
        credentials = credential_manager.list_credentials()
        
        if not credentials:
            print("No credentials found")
            return
        
        print(f"Found {len(credentials)} credential(s):")
        print()
        
        for name, metadata in credentials.items():
            print(f"Name: {name}")
            print(f"  Created: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            if metadata.expires_at:
                expired = metadata.is_expired()
                status = "EXPIRED" if expired else "Valid"
                print(f"  Expires: {metadata.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')} ({status})")
            
            if metadata.rotation_interval_days:
                needs_rotation = metadata.needs_rotation()
                status = "NEEDED" if needs_rotation else "Not needed"
                print(f"  Rotation: Every {metadata.rotation_interval_days} days ({status})")
                
                if metadata.last_rotated:
                    print(f"  Last rotated: {metadata.last_rotated.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                
                print(f"  Rotation count: {metadata.rotation_count}")
            
            print()
            
    except SecurityError as e:
        print(f"Error listing credentials: {e}")
        sys.exit(1)


def rotate_credential(args):
    """Rotate an existing credential."""
    credential_manager, _ = setup_credential_manager()
    
    # Check if credential exists
    if credential_manager.get_credential(args.name) is None:
        print(f"Credential not found: {args.name}")
        sys.exit(1)
    
    # Get new credential value
    if args.value:
        new_value = args.value
    else:
        new_value = getpass.getpass(f"Enter new value for credential '{args.name}': ")
    
    if not new_value:
        print("Error: New credential value cannot be empty")
        sys.exit(1)
    
    try:
        success = credential_manager.rotate_credential(args.name, new_value)
        if success:
            print(f"Successfully rotated credential: {args.name}")
        else:
            print(f"Failed to rotate credential: {args.name}")
            sys.exit(1)
            
    except SecurityError as e:
        print(f"Error rotating credential: {e}")
        sys.exit(1)


def delete_credential(args):
    """Delete a stored credential."""
    credential_manager, _ = setup_credential_manager()
    
    # Confirm deletion unless --force is used
    if not args.force:
        confirm = input(f"Are you sure you want to delete credential '{args.name}'? (y/N): ")
        if confirm.lower() not in ['y', 'yes']:
            print("Deletion cancelled")
            return
    
    try:
        success = credential_manager.delete_credential(args.name)
        if success:
            print(f"Successfully deleted credential: {args.name}")
        else:
            print(f"Credential not found: {args.name}")
            
    except SecurityError as e:
        print(f"Error deleting credential: {e}")
        sys.exit(1)


def check_rotation(args):
    """Check which credentials need rotation."""
    credential_manager, _ = setup_credential_manager()
    
    try:
        credentials = credential_manager.list_credentials()
        
        if not credentials:
            print("No credentials found")
            return
        
        needs_rotation = []
        for name, metadata in credentials.items():
            if metadata.needs_rotation():
                needs_rotation.append((name, metadata))
        
        if not needs_rotation:
            print("No credentials need rotation")
            return
        
        print(f"Found {len(needs_rotation)} credential(s) that need rotation:")
        print()
        
        for name, metadata in needs_rotation:
            print(f"Name: {name}")
            if metadata.last_rotated:
                print(f"  Last rotated: {metadata.last_rotated.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            else:
                print(f"  Created: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"  Rotation interval: {metadata.rotation_interval_days} days")
            print()
            
    except SecurityError as e:
        print(f"Error checking rotation status: {e}")
        sys.exit(1)


def migrate_env_credentials(args):
    """Migrate credentials from environment variables to encrypted storage."""
    credential_manager, _ = setup_credential_manager()
    
    # Define credential mappings
    env_mappings = {
        'alpaca_api_key': 'ALPACA_API_KEY',
        'alpaca_secret_key': 'ALPACA_SECRET_KEY',
    }
    
    migrated = []
    skipped = []
    
    for cred_name, env_var in env_mappings.items():
        env_value = os.getenv(env_var)
        
        if not env_value:
            print(f"Environment variable {env_var} not found, skipping {cred_name}")
            skipped.append(cred_name)
            continue
        
        # Check if credential already exists
        existing = credential_manager.get_credential(cred_name)
        if existing and not args.force:
            print(f"Credential {cred_name} already exists, skipping (use --force to overwrite)")
            skipped.append(cred_name)
            continue
        
        try:
            credential_manager.store_credential(
                cred_name,
                env_value,
                rotation_interval_days=args.rotation_days or 90
            )
            migrated.append(cred_name)
            print(f"Migrated {cred_name} from {env_var}")
            
        except SecurityError as e:
            print(f"Error migrating {cred_name}: {e}")
            skipped.append(cred_name)
    
    print()
    print(f"Migration complete: {len(migrated)} migrated, {len(skipped)} skipped")
    
    if migrated:
        print("Migrated credentials:")
        for name in migrated:
            print(f"  - {name}")
    
    if skipped:
        print("Skipped credentials:")
        for name in skipped:
            print(f"  - {name}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage encrypted credentials for Portfolio Rebalancer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Store a new credential
  python manage_credentials.py store alpaca_api_key --rotation-days 90
  
  # List all credentials
  python manage_credentials.py list
  
  # Get a credential (masked)
  python manage_credentials.py get alpaca_api_key --mask
  
  # Rotate a credential
  python manage_credentials.py rotate alpaca_api_key
  
  # Check which credentials need rotation
  python manage_credentials.py check-rotation
  
  # Migrate from environment variables
  python manage_credentials.py migrate --rotation-days 90
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Store command
    store_parser = subparsers.add_parser('store', help='Store a new credential')
    store_parser.add_argument('name', help='Credential name')
    store_parser.add_argument('--value', help='Credential value (will prompt if not provided)')
    store_parser.add_argument('--expires-days', type=int, help='Days until expiration')
    store_parser.add_argument('--rotation-days', type=int, help='Rotation interval in days')
    store_parser.set_defaults(func=store_credential)
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Retrieve a credential')
    get_parser.add_argument('name', help='Credential name')
    get_parser.add_argument('--mask', action='store_true', help='Mask the credential value')
    get_parser.set_defaults(func=get_credential)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all credentials')
    list_parser.set_defaults(func=list_credentials)
    
    # Rotate command
    rotate_parser = subparsers.add_parser('rotate', help='Rotate an existing credential')
    rotate_parser.add_argument('name', help='Credential name')
    rotate_parser.add_argument('--value', help='New credential value (will prompt if not provided)')
    rotate_parser.set_defaults(func=rotate_credential)
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a credential')
    delete_parser.add_argument('name', help='Credential name')
    delete_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    delete_parser.set_defaults(func=delete_credential)
    
    # Check rotation command
    check_parser = subparsers.add_parser('check-rotation', help='Check which credentials need rotation')
    check_parser.set_defaults(func=check_rotation)
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate credentials from environment variables')
    migrate_parser.add_argument('--rotation-days', type=int, default=90, help='Rotation interval for migrated credentials')
    migrate_parser.add_argument('--force', action='store_true', help='Overwrite existing credentials')
    migrate_parser.set_defaults(func=migrate_env_credentials)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Call the appropriate function
    args.func(args)


if __name__ == '__main__':
    main()