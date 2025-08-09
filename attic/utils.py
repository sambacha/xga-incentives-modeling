from functools import lru_cache, wraps
import time
import logging
import numpy as np
from .types import ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timed_cache(seconds: int = 300, maxsize: int = 128):
    """
    Time-aware LRU cache decorator.

    Adds a time-to-live (TTL) feature to functools.lru_cache.
    The cache is invalidated every `seconds`.
    """
    def wrapper_cache(func):
        @lru_cache(maxsize=maxsize)
        def _cached_func(*args, _ttl_hash, **kwargs):
            # The _ttl_hash parameter is used to invalidate the cache
            return func(*args, **kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Calculate the current time slice to create a TTL effect
            ttl_hash = round(time.time() / seconds)
            return _cached_func(*args, _ttl_hash=ttl_hash, **kwargs)
        
        return wrapper
    return wrapper_cache

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    return numerator / denominator if denominator != 0 else default

def safe_multiplication(a: float, b: float, max_value: float = 1e15) -> float:
    """
    Safe multiplication with overflow protection
    
    Args:
        a: First operand
        b: Second operand
        max_value: Maximum allowed result (default 1 quadrillion)
    
    Returns:
        Result of multiplication, capped at max_value
    
    Raises:
        ValidationError: If result would overflow or inputs are invalid
    """
    # Check for special values
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValidationError(f"Invalid inputs for multiplication: {a} * {b}")
    
    # Check for potential overflow before multiplication
    if a != 0 and b != 0:
        # Use logarithms to check for overflow without actually overflowing
        log_result = np.log10(abs(a)) + np.log10(abs(b))
        if log_result > np.log10(max_value):
            logger.warning(f"Multiplication overflow: {a} * {b} would exceed {max_value}")
            return max_value if (a * b > 0) else -max_value
    
    result = a * b
    
    # Final bounds check
    if abs(result) > max_value:
        logger.warning(f"Multiplication result {result} exceeds maximum {max_value}")
        return max_value if result > 0 else -max_value
    
    return result

def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive"""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return value
