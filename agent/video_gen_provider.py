"""
Video Generation Provider ABC
=============================

Defines the pluggable-backend interface for video generation. Providers register
instances via ``PluginContext.register_video_gen_provider()``; the active one
(selected via ``video_gen.provider`` in ``config.yaml``) services every
``video_generate`` tool call.

Providers live in ``<repo>/plugins/video_gen/<name>/`` (built-in, auto-loaded
as ``kind: backend``) or ``~/.hermes/plugins/video_gen/<name>/`` (user, opt-in
via ``plugins.enabled``).

Mirrors the ``image_gen`` provider design (``agent/image_gen_provider.py``) so
the two surfaces stay learnable together.

Unified surface
---------------
One tool — ``video_generate`` — covers **text-to-video** and **image-to-video**.
The router is the presence of ``image_url``: if it's set, the provider routes
to its image-to-video endpoint; if it's omitted, the provider routes to
text-to-video. Users pick one **model family** (e.g. Pixverse v6, Veo 3.1,
Kling O3 Standard); the provider handles which underlying FAL/xAI endpoint
to hit.

Video edit and video extend are intentionally NOT exposed in this surface —
the inconsistency across backends is too large for one unified tool. If
those use cases warrant attention later they can ship as separate tools.

Response shape
--------------
All providers return a dict built by :func:`success_response` /
:func:`error_response`. Keys:

    success         bool
    video           str | None      URL or absolute file path
    model           str             provider-specific model identifier
    prompt          str             echoed prompt
    modality        str             "text" | "image" (which mode was used)
    aspect_ratio    str             provider-native (e.g. "16:9") or ""
    duration        int             seconds (0 if not applicable)
    provider        str             provider name (for diagnostics)
    error           str             only when success=False
    error_type      str             only when success=False
"""

from __future__ import annotations

import abc
import base64
import datetime
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Common aspect ratios across providers (Veo / Kling / xAI / Pixverse). The
# tool schema advertises this set as an enum hint, but providers may accept
# a narrower or wider set — they are responsible for clamping.
COMMON_ASPECT_RATIOS: Tuple[str, ...] = ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3")
DEFAULT_ASPECT_RATIO = "16:9"

COMMON_RESOLUTIONS: Tuple[str, ...] = ("480p", "540p", "720p", "1080p")
DEFAULT_RESOLUTION = "720p"


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class VideoGenProvider(abc.ABC):
    """Abstract base class for a video generation backend.

    Subclasses must implement :meth:`generate`. Everything else has sane
    defaults — override only what your provider needs.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in ``video_gen.provider`` config.

        Lowercase, no spaces. Examples: ``xai``, ``fal``, ``google``.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``. Defaults to ``name.title()``."""
        return self.name.title()

    def is_available(self) -> bool:
        """Return True when this provider can service calls.

        Typically checks for a required API key and optional-dependency
        import. Default: True.
        """
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        """Return catalog entries for ``hermes tools`` model picker.

        Each entry represents a **model family** that supports text-to-video
        and/or image-to-video routing internally::

            {
                "id": "veo-3.1",                       # required
                "display": "Veo 3.1",                  # optional; defaults to id
                "speed": "~60s",                       # optional
                "strengths": "...",                    # optional
                "price": "$0.20/s",                    # optional
                "modalities": ["text", "image"],       # optional, advisory
            }

        Default: empty list (provider has no user-selectable models).
        """
        return []

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for the ``hermes tools`` picker."""
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "",
            "env_vars": [],
        }

    def default_model(self) -> Optional[str]:
        """Return the default model id, or None if not applicable."""
        models = self.list_models()
        if models:
            return models[0].get("id")
        return None

    def capabilities(self) -> Dict[str, Any]:
        """Return what this provider supports.

        Returned dict (all keys optional)::

            {
                "modalities": ["text", "image"],      # which inputs the backend accepts
                "aspect_ratios": ["16:9", "9:16", ...],
                "resolutions": ["720p", "1080p"],
                "max_duration": 15,                   # seconds
                "min_duration": 1,
                "supports_audio": True,
                "supports_negative_prompt": True,
                "max_reference_images": 7,
            }

        Used by the tool layer for soft validation and by ``hermes tools``
        for the picker. Default: text-only.
        """
        return {
            "modalities": ["text"],
            "aspect_ratios": list(COMMON_ASPECT_RATIOS),
            "resolutions": list(COMMON_RESOLUTIONS),
            "max_duration": 10,
            "min_duration": 1,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "max_reference_images": 0,
        }

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a video from a prompt (text-to-video) or animate an image
        (image-to-video).

        Routing: if ``image_url`` is provided, the provider should route to
        its image-to-video endpoint; otherwise text-to-video. The plugin
        is responsible for picking the right underlying endpoint within
        the user's chosen model family.

        Implementations should return the dict from :func:`success_response`
        or :func:`error_response`. ``kwargs`` may contain forward-compat
        parameters future versions of the schema will expose —
        implementations MUST ignore unknown keys (no TypeError).
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _videos_cache_dir() -> Path:
    """Return ``$HERMES_HOME/cache/videos/``, creating parents as needed."""
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "videos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_b64_video(
    b64_data: str,
    *,
    prefix: str = "video",
    extension: str = "mp4",
) -> Path:
    """Decode base64 video data and write under ``$HERMES_HOME/cache/videos/``.

    Returns the absolute :class:`Path` to the saved file.

    Filename format: ``<prefix>_<YYYYMMDD_HHMMSS>_<short-uuid>.<ext>``.
    """
    raw = base64.b64decode(b64_data)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _videos_cache_dir() / f"{prefix}_{ts}_{short}.{extension}"
    path.write_bytes(raw)
    return path


def save_bytes_video(
    raw: bytes,
    *,
    prefix: str = "video",
    extension: str = "mp4",
) -> Path:
    """Write raw video bytes (e.g. an HTTP download body) to the cache."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = _videos_cache_dir() / f"{prefix}_{ts}_{short}.{extension}"
    path.write_bytes(raw)
    return path


def save_url_video(
    url: str,
    *,
    prefix: str = "video",
    extension: str = "mp4",
    timeout: float = 120.0,
) -> Path:
    """Download a video ``url`` and write it under ``$HERMES_HOME/cache/videos/``.

    Used by providers whose backend returns a delivery URL instead of inline
    bytes (e.g. DeepInfra's ``data[].url`` shape). Raises on network/HTTP
    failure — callers catch and fall back to returning the URL unchanged so
    the tool layer still surfaces a usable result.

    Returns the absolute :class:`Path` to the saved file.
    """
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return save_bytes_video(resp.content, prefix=prefix, extension=extension)


def success_response(
    *,
    video: str,
    model: str,
    prompt: str,
    modality: str = "text",
    aspect_ratio: str = "",
    duration: int = 0,
    provider: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a uniform success response dict.

    ``video`` may be an HTTP URL or an absolute filesystem path.
    ``modality`` is ``"text"`` (text-to-video) or ``"image"`` (image-to-video) —
    indicates which endpoint was actually hit, useful for diagnostics.
    """
    payload: Dict[str, Any] = {
        "success": True,
        "video": video,
        "model": model,
        "prompt": prompt,
        "modality": modality,
        "aspect_ratio": aspect_ratio,
        "duration": int(duration) if duration else 0,
        "provider": provider,
    }
    if extra:
        for k, v in extra.items():
            payload.setdefault(k, v)
    return payload


def error_response(
    *,
    error: str,
    error_type: str = "provider_error",
    provider: str = "",
    model: str = "",
    prompt: str = "",
    aspect_ratio: str = "",
) -> Dict[str, Any]:
    """Build a uniform error response dict."""
    return {
        "success": False,
        "video": None,
        "error": error,
        "error_type": error_type,
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": provider,
    }


# ─── OpenAI-Compatible video backends ────────────────────────────────────────


class OpenAICompatibleVideoGenProvider(VideoGenProvider):
    """Base for backends that speak the OpenAI ``videos`` API.

    Subclasses set ``_env_key`` (the env var carrying the API key) and
    ``_default_base_url``; everything else (client construction, the
    create→poll→download flow, t2v vs i2v routing) is handled here. The
    DeepInfra provider is the first consumer — it overrides only
    ``display_name``, ``list_models``, ``capabilities``, and ``get_setup_schema``.

    The flow:

      1. ``client.videos.create(model=..., seconds=str(duration), extra_body=...)``
         — ``seconds`` is passed as a string because the OpenAI videos schema
         accepts it that way; image-to-video inputs (``image_url``) and
         provider-specific knobs (``negative_prompt``) ride in ``extra_body``.
      2. If ``create`` returns a terminal status (``succeeded`` / ``completed``
         / ``failed``) ``_create_and_poll`` returns immediately — no retrieve()
         call, no sleep. A non-terminal status would poll ``retrieve(id)``
         until terminal (bounded).
      3. Delivery: a ``data[].url`` carries a download URL → ``save_url_video``
         (falls back to the URL itself if the local save raises). Otherwise
         (Sora-style) → ``download_content(id).read()`` bytes → ``save_bytes_video``.
      4. The SDK's structured ``error`` object is ``str()``-coerced so the
         response dict survives the tool layer's ``json.dumps``.
    """

    _env_key: str = ""
    _default_base_url: str = ""

    @property
    def display_name(self) -> str:  # type: ignore[override]
        return self.name.title()

    def _api_key(self) -> Optional[str]:
        import os

        return os.environ.get(self._env_key) if self._env_key else None

    def _base_url(self) -> str:
        return self._default_base_url

    def is_available(self) -> bool:
        return bool(self._api_key())

    def _client(self):
        """Build the OpenAI SDK client against this provider's base URL."""
        import openai

        return openai.OpenAI(api_key=self._api_key(), base_url=self._base_url())

    def _create_and_poll(self, client, **create_kwargs):
        """Create a videos job and poll until terminal.

        ``create`` may return a terminal status immediately (DeepInfra) or a
        pending one that must be polled via ``retrieve`` (Sora). Terminal
        statuses exit the loop without an extra call or sleep. The poll is
        bounded so a misbehaving backend cannot loop forever.
        """
        job = client.videos.create(**create_kwargs)
        terminal = {"succeeded", "completed", "failed"}
        # Defensive: some SDKs wrap status in a nested attribute; accept either.
        status = getattr(job, "status", None)
        if status in terminal:
            return job

        job_id = getattr(job, "id", None)
        # Bounded poll — a non-terminal create that never resolves must not hang.
        import time

        for _ in range(60):
            if job_id is None:
                break
            job = client.videos.retrieve(job_id)
            status = getattr(job, "status", None)
            if status in terminal or status is None:
                return job
            time.sleep(2)
        return job

    @staticmethod
    def _first_delivery_url(job) -> Optional[str]:
        """Extract the first ``data[].url`` from a completed job, if any."""
        data = getattr(job, "data", None) or []
        for entry in data:
            url = None
            if isinstance(entry, dict):
                url = entry.get("url")
            else:
                url = getattr(entry, "url", None)
            if url:
                return url
        return None

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.is_available():
            return error_response(
                error=f"{self._env_key} not set. Configure it via `hermes tools` → Video Generation.",
                error_type="missing_credentials",
                provider=self.name,
                model=model or "",
                prompt=prompt,
            )

        modality = "image" if image_url else "text"
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }
        if duration is not None:
            # OpenAI videos schema accepts ``seconds`` as a string.
            create_kwargs["seconds"] = str(duration)

        extra_body: Dict[str, Any] = {}
        if image_url:
            extra_body["image_url"] = image_url
        if negative_prompt:
            extra_body["negative_prompt"] = negative_prompt
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        # Forward-compat: ignore unknown kwargs (no TypeError to the caller).

        try:
            client = self._client()
            job = self._create_and_poll(client, **create_kwargs)
        except Exception as e:  # noqa: BLE001 — surface as provider error
            return error_response(
                error=str(e),
                error_type="provider_error",
                provider=self.name,
                model=model or "",
                prompt=prompt,
            )

        status = getattr(job, "status", None)
        if status == "failed":
            # SDK error objects (pydantic VideoCreateError) aren't JSON-serializable;
            # coerce so the tool layer's json.dumps survives.
            err = getattr(job, "error", None)
            return error_response(
                error=str(err) if err is not None else "video generation failed",
                error_type="job_failed",
                provider=self.name,
                model=model or "",
                prompt=prompt,
            )

        # Delivery path A: data[].url → download & save locally.
        url = self._first_delivery_url(job)
        if url:
            try:
                path = save_url_video(url, prefix=self.name)
                return success_response(
                    video=str(path),
                    model=model or "",
                    prompt=prompt,
                    modality=modality,
                    aspect_ratio=aspect_ratio,
                    duration=int(duration) if duration else 0,
                    provider=self.name,
                )
            except Exception:
                # Local save failed (network, disk) — fall back to the raw URL
                # so the tool layer still surfaces a usable result.
                return success_response(
                    video=url,
                    model=model or "",
                    prompt=prompt,
                    modality=modality,
                    aspect_ratio=aspect_ratio,
                    duration=int(duration) if duration else 0,
                    provider=self.name,
                )

        # Delivery path B: Sora-style download_content(id).read() bytes.
        job_id = getattr(job, "id", None)
        if job_id is not None:
            try:
                content_obj = client.videos.download_content(job_id)
                raw = content_obj.read() if hasattr(content_obj, "read") else bytes(content_obj)
                path = save_bytes_video(raw, prefix=self.name)
                return success_response(
                    video=str(path),
                    model=model or "",
                    prompt=prompt,
                    modality=modality,
                    aspect_ratio=aspect_ratio,
                    duration=int(duration) if duration else 0,
                    provider=self.name,
                )
            except Exception as e:  # noqa: BLE001
                return error_response(
                    error=str(e),
                    error_type="provider_error",
                    provider=self.name,
                    model=model or "",
                    prompt=prompt,
                )

        return error_response(
            error="job completed without a deliverable url or downloadable content",
            error_type="job_failed",
            provider=self.name,
            model=model or "",
            prompt=prompt,
        )
