"""Abstract base + dataclasses + exceptions for dashboard auth providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Session:
    """A verified identity. Returned by ``complete_login`` and ``verify_session``.

    All fields are mandatory. Providers that don't have a concept of orgs
    should set ``org_id`` to an empty string. ``access_token`` and
    ``refresh_token`` are opaque to Hermes — provider-specific.
    """

    user_id: str
    email: str
    display_name: str
    org_id: str
    provider: str
    expires_at: int  # unix seconds; the access_token's exp claim
    access_token: str
    refresh_token: str


@dataclass(frozen=True)
class TokenPrincipal:
    """A non-interactive (bearer-token) verified identity.

    Returned by ``verify_token`` on a ``supports_token`` provider. Carries the
    principal identifier (an opaque label the provider chooses — e.g.
    ``"drain-control"``), the provider's stable ``name``, and a tuple of scope
    strings the middleware attaches to ``request.state`` so route handlers can
    authorize per-scope. Opaque to Hermes — providers define what scopes mean.
    """

    principal: str
    provider: str
    scopes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LoginStart:
    """First leg of the OAuth round trip.

    ``redirect_url`` is the URL the browser must navigate to (e.g. the
    Portal's ``/oauth/authorize``). ``cookie_payload`` is a dict of cookie
    name → serialised value that the auth route will ``Set-Cookie`` on the
    response. Used for PKCE state, CSRF nonces, etc. Cookies set here MUST
    be HttpOnly + Secure (when over HTTPS) + SameSite=Lax with a TTL ≤ 10
    minutes (the login lifetime).
    """

    redirect_url: str
    cookie_payload: dict[str, str]


class ProviderError(Exception):
    """IDP unreachable, network error, or other transient failure.

    Middleware translates this to HTTP 503.
    """


class InvalidCodeError(Exception):
    """The OAuth callback ``code`` / ``state`` failed validation.

    Middleware translates this to HTTP 400.
    """


class InvalidCredentialsError(Exception):
    """Username/password (or other interactive credential) rejected.

    Raised by password-login providers (e.g. BasicAuthProvider) when the
    supplied credentials don't match. The login route maps this to HTTP 401
    with a generic "invalid username or password" body — it must NOT
    distinguish "unknown user" from "wrong password" (both return the same
    generic 401) to avoid a user-enumeration side channel.
    """


class RefreshExpiredError(Exception):
    """The refresh token is dead.

    Middleware clears cookies and forces re-login (302 → ``/login``).
    """


class DashboardAuthProvider(ABC):
    """Protocol every dashboard-auth provider plugin implements.

    Lifecycle:
      1. ``start_login`` — user clicks "Log in with X" on the login page.
         Provider returns a redirect URL and any PKCE/CSRF state to stash
         in short-lived cookies.
      2. Browser bounces through the OAuth IDP and lands at /auth/callback.
      3. ``complete_login`` — exchange the code + verifier for a Session.
      4. ``verify_session`` — called on every request to validate the
         access token in the cookie. Returns ``None`` if the token is
         expired or invalid (middleware then triggers refresh or logout).
      5. ``refresh_session`` — called when the access token is near expiry.
         Returns a new Session with rotated tokens.
      6. ``revoke_session`` — called on /auth/logout. Best-effort.

    Failure semantics:
      * ``start_login`` may raise ``ProviderError`` if the IDP is
        unreachable.
      * ``complete_login`` raises ``InvalidCodeError`` on bad code/state;
        ``ProviderError`` if the IDP is unreachable.
      * ``verify_session`` returns ``None`` on expiry / unknown token;
        raises ``ProviderError`` if the IDP is unreachable. Middleware
        treats expiry and unreachable differently (expiry → refresh;
        unreachable → 503).
      * ``refresh_session`` raises ``RefreshExpiredError`` when the
        refresh token is also invalid; middleware then forces re-login.
        Raises ``ProviderError`` on network failure.
      * ``revoke_session`` is best-effort and must not raise.

    Subclasses MUST set ``name`` (lowercase identifier, stable forever)
    and ``display_name`` (user-facing label on the login page).
    """

    name: str = ""
    display_name: str = ""

    # Capability flags — NOT abstract, so existing session-only providers keep
    # working without touching them. ``supports_session`` defaults True
    # (interactive cookie-session providers participate by default);
    # ``supports_token`` defaults False (the non-interactive bearer-token seam
    # is opt-in); ``supports_password`` defaults False (username/password
    # non-redirect login is opt-in). Registry filters
    # ``list_token_providers`` / ``list_session_providers`` on these so each
    # auth path only consults the providers that actually back it.
    supports_session: bool = True
    supports_token: bool = False
    supports_password: bool = False

    @abstractmethod
    def start_login(self, *, redirect_uri: str) -> LoginStart: ...

    @abstractmethod
    def complete_login(
        self,
        *,
        code: str,
        state: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Session: ...

    @abstractmethod
    def verify_session(self, *, access_token: str) -> Optional[Session]: ...

    @abstractmethod
    def refresh_session(self, *, refresh_token: str) -> Session: ...

    @abstractmethod
    def revoke_session(self, *, refresh_token: str) -> None: ...

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Verify a non-interactive bearer token.

        Default implementation raises :class:`NotImplementedError` — only
        providers that set ``supports_token = True`` are expected to override
        this. The token-auth middleware never calls ``verify_token`` on a
        ``supports_token = False`` provider (it filters via
        :func:`~hermes_cli.dashboard_auth.list_token_providers` first), so a
        session-only provider that never opts in will not trip the default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support non-interactive token auth "
            f"(supports_token is False). Override verify_token and set "
            f"supports_token = True to opt in."
        )

    def complete_password_login(self, *, username: str, password: str) -> Session:
        """Exchange username/password for a :class:`Session`.

        Non-redirect login path: a provider that backs a credential form
        (e.g. BasicAuthProvider) overrides this and sets
        ``supports_password = True``. The ``/auth/password-login`` route
        calls it; on :class:`InvalidCredentialsError` the route returns a
        generic HTTP 401 (no user-vs-password distinction — that would be a
        user-enumeration side channel).

        Default raises :class:`NotImplementedError` — the password-login
        route 404s for ``supports_password = False`` providers before ever
        reaching this method, so OAuth-only providers never trip the default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support password login "
            f"(supports_password is False). Override complete_password_login "
            f"and set supports_password = True to opt in."
        )


def assert_protocol_compliance(cls: type) -> None:
    """Raise ``TypeError`` if ``cls`` doesn't fully implement the provider protocol.

    Call this in every provider plugin's unit tests::

        def test_protocol_compliance():
            assert_protocol_compliance(MyProvider)

    Returns ``None`` on success so callers can assert it explicitly.
    """
    required_methods = (
        "start_login",
        "complete_login",
        "verify_session",
        "refresh_session",
        "revoke_session",
    )
    required_attrs = ("name", "display_name")

    for attr in required_attrs:
        val = getattr(cls, attr, "")
        if not val:
            raise TypeError(
                f"{cls.__name__} missing or empty attribute: {attr!r}"
            )
    for method in required_methods:
        if not callable(getattr(cls, method, None)):
            raise TypeError(f"{cls.__name__} missing method: {method}")
    # Also catch the ABC-not-overridden case.
    if getattr(cls, "__abstractmethods__", None):
        raise TypeError(
            f"{cls.__name__} has unimplemented abstract methods: "
            f"{sorted(cls.__abstractmethods__)}"
        )
