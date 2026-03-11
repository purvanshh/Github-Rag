"""
security.py — JWT authentication, rate limiting, and inputs validation.
"""

from __future__ import annotations

import re
import hmac
import time
import base64
import json
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional

SECRET_KEY = "github-rag-default-security-secret-key-signature"
API_KEYS = {"test-api-key-12345": "test-user"}
RATE_LIMIT_STAMPS = defaultdict(list)
LIMIT_PER_MINUTE = 60


def sign_jwt(payload: dict) -> str:
    """Sign a JWT token using SHA-256 HMAC."""
    header = {"alg": "HS256", "typ": "JWT"}
    # Add expiration time (default 1 hour)
    if "exp" not in payload:
        payload["exp"] = int(time.time()) + 3600

    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    
    msg = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(SECRET_KEY.encode(), msg, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def verify_jwt(token: str) -> Optional[dict]:
    """Verify a signed JWT token signature and expiration."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        
        # Verify signature
        msg = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(SECRET_KEY.encode(), msg, hashlib.sha256).digest()
        expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")
        
        if not hmac.compare_digest(sig_b64, expected_sig_b64):
            return None
            
        # Parse payload
        pad_len = 4 - (len(payload_b64) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + "=" * (pad_len if pad_len < 4 else 0))
        payload = json.loads(payload_bytes.decode())
        
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def is_rate_limited(api_key: str) -> bool:
    """Return True if the API key exceeded the limit of 60 requests per minute."""
    now = time.time()
    stamps = RATE_LIMIT_STAMPS[api_key]
    stamps = [s for s in stamps if now - s < 60]
    RATE_LIMIT_STAMPS[api_key] = stamps
    
    if len(stamps) >= LIMIT_PER_MINUTE:
        return True
    
    RATE_LIMIT_STAMPS[api_key].append(now)
    return False


def validate_repo_url(url: str) -> bool:
    """Strict regex validation for repository clone URLs."""
    pattern = r"^https?://[a-zA-Z0-9.\-_]+/[a-zA-Z0-9.\-_]+/[a-zA-Z0-9.\-_]+/?$"
    git_pattern = r"^git@[a-zA-Z0-9.\-_]+:[a-zA-Z0-9.\-_]+/[a-zA-Z0-9.\-_]+\.git$"
    local_path_pattern = r"^[a-zA-Z0-9.\-_\/]+$"
    
    return (
        bool(re.match(pattern, url)) 
        or bool(re.match(git_pattern, url))
        or bool(re.match(local_path_pattern, url))
    )
