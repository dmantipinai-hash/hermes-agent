"""Module-level registry for DashboardAuthProvider instances.

Plugins call ``register_provider`` via the plugin context hook at startup.
The auth gate middleware iterates ``list_providers()`` and uses
``get_provider`` to dispatch on the session's ``provider`` field.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    assert_protocol_compliance,
)

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_providers: dict[str, DashboardAuthProvider] = {}


def register_provider(provider: DashboardAuthProvider) -> None:
    """Register a provider.

    Raises:
        TypeError: on protocol violation.
        ValueError: if a provider with the same name is already registered.
    """
    assert_protocol_compliance(type(provider))
    with _lock:
        if provider.name in _providers:
            raise ValueError(
                f"dashboard-auth provider already registered: {provider.name!r}"
            )
        _providers[provider.name] = provider
    _log.info(
        "dashboard-auth: registered provider %r (%s)",
        provider.name, provider.display_name,
    )


def get_provider(name: str) -> Optional[DashboardAuthProvider]:
    """Return the registered provider for ``name``, or None if unknown."""
    with _lock:
        return _providers.get(name)


def list_providers() -> List[DashboardAuthProvider]:
    """All registered providers, in registration order."""
    with _lock:
        return list(_providers.values())


def list_session_providers() -> List[DashboardAuthProvider]:
    """Providers that participate in interactive cookie-session auth.

    Filters :func:`list_providers` to those with ``supports_session = True``.
    Token-only providers (``supports_session = False``, e.g. a service
    credential) are excluded so the cookie-verify loop never consults them.
    """
    return [p for p in list_providers() if getattr(p, "supports_session", True)]


def list_token_providers() -> List[DashboardAuthProvider]:
    """Providers that participate in non-interactive bearer-token auth.

    Filters :func:`list_providers` to those with ``supports_token = True``.
    The token-auth middleware consults only these when verifying an
    ``Authorization: Bearer`` header, so session-only providers are never
    asked to recognize a token.
    """
    return [p for p in list_providers() if getattr(p, "supports_token", False)]


def clear_providers() -> None:
    """Test-only: drop all registrations."""
    with _lock:
        _providers.clear()
