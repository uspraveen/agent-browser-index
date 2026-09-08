#!/usr/bin/env python3
"""
Per-run credential store for the Cosmic Browser Use Agent.

Holds vault credentials provisioned by the Cosmic orchestrator for this run.
Values live only in memory for the lifetime of a single run, are never logged,
never placed in screenshots, and never enter the LLM decision context — the
model only learns WHICH site domains have credentials (see
TaskConfig.credentials_available_for) and requests a deterministic fill via
the CredentialFill action.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional


def normalize_site_domain(value: str) -> str:
    """Reduce a URL or domain to a registrable-looking host for matching."""
    host = str(value or "").strip().lower()
    if not host:
        return ""
    if "://" in host:
        host = host.split("://", 1)[1]
    host = host.split("/", 1)[0]
    host = host.split(":", 1)[0]
    host = host.lstrip(".")
    # Strip common public suffixes one level for forgiving matching
    parts = host.split(".")
    if len(parts) > 2 and parts[-1] in {"com", "org", "net", "io", "ai", "co", "gov", "edu"}:
        parts = parts[1:]
        host = ".".join(parts)
    return host


class CredentialStore:
    """In-memory credential vault for one browser run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, Dict[str, str]] = {}

    def add(self, site_domain: str, username: str = "", password: str = "",
            totp_seed: str = "", notes: str = "", site_url: str = "") -> str:
        domain = normalize_site_domain(site_domain)
        if not domain:
            raise ValueError("site_domain is required for a credential entry")
        with self._lock:
            self._entries[domain] = {
                "site_domain": domain,
                "site_url": str(site_url or ""),
                "username": str(username or ""),
                "password": str(password or ""),
                "totp_seed": str(totp_seed or ""),
                "notes": str(notes or ""),
            }
        return domain

    def has_for(self, site_domain_or_url: str) -> bool:
        domain = normalize_site_domain(site_domain_or_url)
        with self._lock:
            if domain in self._entries:
                return True
            return any(domain.endswith("." + d) or d.endswith("." + domain) for d in self._entries)

    def get(self, site_domain_or_url: str) -> Optional[Dict[str, str]]:
        domain = normalize_site_domain(site_domain_or_url)
        with self._lock:
            entry = self._entries.get(domain)
            if entry:
                return dict(entry)
            for known_domain, known in self._entries.items():
                if domain.endswith("." + known_domain) or known_domain.endswith("." + domain):
                    return dict(known)
        return None

    def available_domains(self) -> List[str]:
        with self._lock:
            return sorted(self._entries.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - safety: never leak values
        return f"CredentialStore(domains={self.available_domains()!r})"
