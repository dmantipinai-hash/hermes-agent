"""Marker making ``locales/`` a setuptools package.

The catalogs are read as data files by :mod:`agent.i18n` (``_locales_dir``
resolves ``<repo-root>/locales`` for checkouts and ``site-packages/locales``
for wheel installs). Without this marker — and the matching
``packages.find`` / ``package-data`` entries in ``pyproject.toml`` — the
directory is invisible to wheel builds: ``_locales_dir()`` then points at a
missing path, ``t()`` degrades to returning raw dotted keys, and localized
runtime messages (e.g. the kanban notifier's wake notes) arrive as bare
``gateway.kanban.wake.message`` key strings with no payload
(BUG-kanban-wake-empty-payload, 2026-09-17). Nothing imports this module.
"""
