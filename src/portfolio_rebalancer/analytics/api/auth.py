"""Authentication and authorization system for analytics API."""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from functools import wraps
from enum import Enum

import jwt
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import redis
from passlib.context import CryptContext

from ..exceptions import AuthenticationError, AuthorizationError, RateLimitError

logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET_KEY = "your-secret-key-here"  # Should be from environment
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
REFRESH_TOKEN_EXPIRATION_DAYS = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


class UserRole(str, Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(str, Enum):
    """Permissions for fine-grained access control."""
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_WRITE = "analytics:write"
    ANALYTICS_DELETE = "analytics:delete"
    
    # Backtest permissions
    BACKTEST_RUN = "backtest:run"
    BACKTEST_READ = "backtest:read"
    BACKTEST_DELETE = "backtest:delete"
    
    # Monte Carlo permissions
    MONTE_CARLO_RUN = "monte_carlo:run"
    MONTE_CARLO_READ = "monte_carlo:read"
    
    # Risk analysis permissions
    RISK_ANALYSIS_RUN = "risk:run"
    RISK_ANALYSIS_READ = "risk:read"
    
    # Performance permissions
    PERFORMANCE_READ = "performance:read"
    PERFORMANCE_WRITE = "performance:write"
    
    # Dividend permissions
    DIVIDEND_READ = "dividend:read"
    DIVIDEND_WRITE = "dividend:write"
    
    # Export permissions
    EXPORT_DATA = "export:data"
    EXPORT_REPORTS = "export:reports"
    
    # Admin permissions
    USER_MANAGEMENT = "user:management"
    SYSTEM_CONFIG = "system:config"
    METRICS_READ = "metrics:read"


# Role-permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        # All permissions
        Permission.ANALYTICS_READ,
        Permission.ANALYTICS_WRITE,
        Permission.ANALYTICS_DELETE,
        Permission.BACKTEST_RUN,
        Permission.BACKTEST_READ,
        Permission.BACKTEST_DELETE,
        Permission.MONTE_CARLO_RUN,
        Permission.MONTE_CARLO_READ,
        Permission.RISK_ANALYSIS_RUN,
        Permission.RISK_ANALYSIS_READ,
        Permission.PERFORMANCE_READ,
        Permission.PERFORMANCE_WRITE,
        Permission.DIVIDEND_READ,
        Permission.DIVIDEND_WRITE,
        Permission.EXPORT_DATA,
        Permission.EXPORT_REPORTS,
        Permission.USER_MANAGEMENT,
        Permission.SYSTEM_CONFIG,
        Permission.METRICS_READ,
    ],
    UserRole.ANALYST: [
        # Full analytics capabilities
        Permission.ANALYTICS_READ,
        Permission.ANALYTICS_WRITE,
        Permission.BACKTEST_RUN,
        Permission.BACKTEST_READ,
        Permission.MONTE_CARLO_RUN,
        Permission.MONTE_CARLO_READ,
        Permission.RISK_ANALYSIS_RUN,
        Permission.RISK_ANALYSIS_READ,
        Permission.PERFORMANCE_READ,
        Permission.PERFORMANCE_WRITE,
        Permission.DIVIDEND_READ,
        Permission.DIVIDEND_WRITE,
        Permission.EXPORT_DATA,
        Permission.EXPORT_REPORTS,
    ],
    UserRole.VIEWER: [
        # Read-only access
        Permission.ANALYTICS_READ,
        Permission.BACKTEST_READ,
        Permission.MONTE_CARLO_READ,
        Permission.RISK_ANALYSIS_READ,
        Permission.PERFORMANCE_READ,
        Permission.DIVIDEND_READ,
        Permission.EXPORT_DATA,
    ],
    UserRole.API_USER: [
        # API access for integrations
        Permission.ANALYTICS_READ,
        Permission.BACKTEST_RUN,
        Permission.BACKTEST_READ,
        Permission.MONTE_CARLO_RUN,
        Permission.MONTE_CARLO_READ,
        Permission.RISK_ANALYSIS_RUN,
        Permission.RISK_ANALYSIS_READ,
        Permission.PERFORMANCE_READ,
        Permission.DIVIDEND_READ,
        Permission.EXPORT_DATA,
    ],
}


class User(BaseModel):
    """User model."""
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    role: UserRole = Field(..., description="User role")
    permissions: List[Permission] = Field(..., description="User permissions")
    portfolio_access: List[str] = Field(default_factory=list, description="Accessible portfolio IDs")
    is_active: bool = Field(True, description="Whether user is active")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")


class TokenData(BaseModel):
    """JWT token data."""
    user_id: str
    username: str
    role: UserRole
    permissions: List[Permission]
    portfolio_access: List[str]
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for token revocation


class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: Dict[str, Any] = Field(..., description="User information")


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., description="Refresh token")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(60, description="Requests per minute")
    requests_per_hour: int = Field(1000, description="Requests per hour")
    requests_per_day: int = Field(10000, description="Requests per day")
    burst_limit: int = Field(10, description="Burst request limit")


# Mock user database (in production, this would be a real database)
MOCK_USERS = {
    "admin": {
        "user_id": "user_001",
        "username": "admin",
        "email": "admin@example.com",
        "password_hash": pwd_context.hash("admin123"),
        "role": UserRole.ADMIN,
        "portfolio_access": ["*"],  # Access to all portfolios
        "is_active": True,
    },
    "analyst": {
        "user_id": "user_002",
        "username": "analyst",
        "email": "analyst@example.com",
        "password_hash": pwd_context.hash("analyst123"),
        "role": UserRole.ANALYST,
        "portfolio_access": ["portfolio_001", "portfolio_002"],
        "is_active": True,
    },
    "viewer": {
        "user_id": "user_003",
        "username": "viewer",
        "email": "viewer@example.com",
        "password_hash": pwd_context.hash("viewer123"),
        "role": UserRole.VIEWER,
        "portfolio_access": ["portfolio_001"],
        "is_active": True,
    },
    "api_user": {
        "user_id": "user_004",
        "username": "api_user",
        "email": "api@example.com",
        "password_hash": pwd_context.hash("api123"),
        "role": UserRole.API_USER,
        "portfolio_access": ["portfolio_001", "portfolio_002", "portfolio_003"],
        "is_active": True,
    },
}


class AuthService:
    """Authentication and authorization service."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize auth service."""
        self.redis_client = redis_client
        self.rate_limits = {
            UserRole.ADMIN: RateLimitConfig(requests_per_minute=120, requests_per_hour=2000),
            UserRole.ANALYST: RateLimitConfig(requests_per_minute=100, requests_per_hour=1500),
            UserRole.VIEWER: RateLimitConfig(requests_per_minute=60, requests_per_hour=1000),
            UserRole.API_USER: RateLimitConfig(requests_per_minute=200, requests_per_hour=5000),
        }
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials."""
        try:
            user_data = MOCK_USERS.get(username)
            if not user_data:
                return None
            
            if not user_data["is_active"]:
                return None
            
            if not pwd_context.verify(password, user_data["password_hash"]):
                return None
            
            # Create user object
            permissions = ROLE_PERMISSIONS.get(user_data["role"], [])
            
            user = User(
                user_id=user_data["user_id"],
                username=user_data["username"],
                email=user_data["email"],
                role=user_data["role"],
                permissions=permissions,
                portfolio_access=user_data["portfolio_access"],
                is_active=user_data["is_active"],
                last_login=datetime.now(timezone.utc)
            )
            
            return user
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        try:
            now = datetime.now(timezone.utc)
            exp = now + timedelta(hours=JWT_EXPIRATION_HOURS)
            jti = secrets.token_urlsafe(32)
            
            token_data = TokenData(
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                permissions=user.permissions,
                portfolio_access=user.portfolio_access,
                exp=exp,
                iat=now,
                jti=jti
            )
            
            payload = {
                "user_id": token_data.user_id,
                "username": token_data.username,
                "role": token_data.role.value,
                "permissions": [p.value for p in token_data.permissions],
                "portfolio_access": token_data.portfolio_access,
                "exp": int(exp.timestamp()),
                "iat": int(now.timestamp()),
                "jti": jti
            }
            
            token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            
            # Store token in Redis for revocation tracking
            if self.redis_client:
                self.redis_client.setex(
                    f"token:{jti}",
                    int(timedelta(hours=JWT_EXPIRATION_HOURS).total_seconds()),
                    "active"
                )
            
            return token
            
        except Exception as e:
            logger.error(f"Token creation error: {e}")
            raise AuthenticationError("Failed to create access token")
    
    def create_refresh_token(self, user: User) -> str:
        """Create refresh token."""
        try:
            now = datetime.now(timezone.utc)
            exp = now + timedelta(days=REFRESH_TOKEN_EXPIRATION_DAYS)
            jti = secrets.token_urlsafe(32)
            
            payload = {
                "user_id": user.user_id,
                "type": "refresh",
                "exp": int(exp.timestamp()),
                "iat": int(now.timestamp()),
                "jti": jti
            }
            
            token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            
            # Store refresh token in Redis
            if self.redis_client:
                self.redis_client.setex(
                    f"refresh_token:{jti}",
                    int(timedelta(days=REFRESH_TOKEN_EXPIRATION_DAYS).total_seconds()),
                    user.user_id
                )
            
            return token
            
        except Exception as e:
            logger.error(f"Refresh token creation error: {e}")
            raise AuthenticationError("Failed to create refresh token")
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if self.redis_client and jti:
                if not self.redis_client.exists(f"token:{jti}"):
                    raise AuthenticationError("Token has been revoked")
            
            # Create token data object
            token_data = TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                role=UserRole(payload["role"]),
                permissions=[Permission(p) for p in payload["permissions"]],
                portfolio_access=payload["portfolio_access"],
                exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
                iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
                jti=jti
            )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise AuthenticationError("Token verification failed")
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Refresh access token using refresh token."""
        try:
            payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid refresh token")
            
            # Check if refresh token is revoked
            jti = payload.get("jti")
            if self.redis_client and jti:
                user_id = self.redis_client.get(f"refresh_token:{jti}")
                if not user_id:
                    raise AuthenticationError("Refresh token has been revoked")
                user_id = user_id.decode() if isinstance(user_id, bytes) else user_id
            else:
                user_id = payload["user_id"]
            
            # Get user data and create new access token
            user_data = None
            for username, data in MOCK_USERS.items():
                if data["user_id"] == user_id:
                    user_data = data
                    break
            
            if not user_data or not user_data["is_active"]:
                raise AuthenticationError("User not found or inactive")
            
            permissions = ROLE_PERMISSIONS.get(user_data["role"], [])
            user = User(
                user_id=user_data["user_id"],
                username=user_data["username"],
                email=user_data["email"],
                role=user_data["role"],
                permissions=permissions,
                portfolio_access=user_data["portfolio_access"],
                is_active=user_data["is_active"]
            )
            
            return self.create_access_token(user)
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid refresh token: {str(e)}")
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            raise AuthenticationError("Token refresh failed")
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            jti = payload.get("jti")
            
            if self.redis_client and jti:
                # Remove token from Redis
                if payload.get("type") == "refresh":
                    self.redis_client.delete(f"refresh_token:{jti}")
                else:
                    self.redis_client.delete(f"token:{jti}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Token revocation error: {e}")
            return False
    
    def check_rate_limit(self, user_id: str, role: UserRole, request: Request) -> bool:
        """Check if user has exceeded rate limits."""
        if not self.redis_client:
            return True  # No rate limiting without Redis
        
        try:
            rate_config = self.rate_limits.get(role, RateLimitConfig())
            now = datetime.now()
            
            # Check different time windows
            windows = [
                ("minute", 60, rate_config.requests_per_minute),
                ("hour", 3600, rate_config.requests_per_hour),
                ("day", 86400, rate_config.requests_per_day),
            ]
            
            for window_name, window_seconds, limit in windows:
                key = f"rate_limit:{user_id}:{window_name}:{now.strftime('%Y%m%d%H%M' if window_name == 'minute' else '%Y%m%d%H' if window_name == 'hour' else '%Y%m%d')}"
                
                current_count = self.redis_client.get(key)
                current_count = int(current_count) if current_count else 0
                
                if current_count >= limit:
                    raise RateLimitError(
                        f"Rate limit exceeded for {window_name}",
                        limit=limit,
                        window_seconds=window_seconds,
                        retry_after=window_seconds
                    )
                
                # Increment counter
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, window_seconds)
                pipe.execute()
            
            return True
            
        except RateLimitError:
            raise
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow request on error
    
    def has_permission(self, user: TokenData, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in user.permissions
    
    def has_portfolio_access(self, user: TokenData, portfolio_id: str) -> bool:
        """Check if user has access to specific portfolio."""
        if "*" in user.portfolio_access:  # Admin access
            return True
        return portfolio_id in user.portfolio_access


# Global auth service instance
auth_service = AuthService()


# Dependency functions
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """Get current authenticated user."""
    try:
        token = credentials.credentials
        token_data = auth_service.verify_token(token)
        return token_data
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Get current active user."""
    # In a real implementation, check if user is still active in database
    return current_user


def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                # Try to get from dependencies
                for arg in args:
                    if isinstance(arg, TokenData):
                        current_user = arg
                        break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not auth_service.has_permission(current_user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission.value}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role: UserRole):
    """Decorator to require specific role."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                # Try to get from dependencies
                for arg in args:
                    if isinstance(arg, TokenData):
                        current_user = arg
                        break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if current_user.role != role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {role.value}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_portfolio_access(portfolio_id_param: str = "portfolio_id"):
    """Decorator to require access to specific portfolio."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user and portfolio_id from kwargs
            current_user = kwargs.get('current_user')
            portfolio_id = kwargs.get(portfolio_id_param)
            
            if not current_user:
                # Try to get from dependencies
                for arg in args:
                    if isinstance(arg, TokenData):
                        current_user = arg
                        break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not portfolio_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Portfolio ID required"
                )
            
            if not auth_service.has_portfolio_access(current_user, portfolio_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied to portfolio: {portfolio_id}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


async def check_rate_limit(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
):
    """Check rate limits for current user."""
    try:
        auth_service.check_rate_limit(current_user.user_id, current_user.role, request)
    except RateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
            headers={"Retry-After": str(e.retry_after)} if e.retry_after else {}
        )