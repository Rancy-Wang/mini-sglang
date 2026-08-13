from __future__ import annotations


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def page_count(num_tokens: int, page_size: int) -> int:
    """Return the number of pages needed for a token length."""
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative.")
    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    return div_ceil(num_tokens, page_size)


def last_page_len(num_tokens: int, page_size: int) -> int:
    """Return the valid token count in the final page."""
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative.")
    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if num_tokens == 0:
        return 0
    remainder = num_tokens % page_size
    return page_size if remainder == 0 else remainder


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


class Unset:
    pass


UNSET = Unset()
