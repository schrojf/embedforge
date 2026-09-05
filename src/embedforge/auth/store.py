"""File-backed token store.

A JSON file is the right storage here: tokens are few, change rarely, and a plain
file is trivial to back up, mount into a container, and edit under version control
of the operator's choosing. Writes are atomic (temp file + `os.replace`) and the
file is kept at mode 0600.

The server re-reads the file when its mtime changes, so `embedforge token create`
and `embedforge token revoke` take effect without a restart.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from embedforge.auth.models import (
    DEFAULT_SCOPES,
    Principal,
    Scope,
    TokenFile,
    TokenRecord,
    generate_token,
    hash_token,
    parse_token_id,
    utcnow,
)
from embedforge.logging import get_logger

log = get_logger(__name__)

_FILE_MODE = 0o600
_DIR_MODE = 0o700


class TokenStore:
    """Thread-safe view over the token file."""

    def __init__(self, path: Path, *, reload_interval: float = 5.0) -> None:
        self.path = Path(path)
        self.reload_interval = reload_interval
        self._lock = threading.RLock()
        self._data = TokenFile()
        self._signature: tuple[int, int] | None = None
        self._checked_at = 0.0
        self.reload(force=True)
        if self._signature is None:
            log.warning("token_file_missing", path=str(self.path))

    # ---- Loading and saving ----

    def _file_signature(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def reload(self, *, force: bool = False) -> None:
        """Re-read the file if it changed. Cheap enough to call per request."""
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._checked_at) < self.reload_interval:
                return
            self._checked_at = now
            signature = self._file_signature()
            if signature == self._signature and not force:
                return
            self._signature = signature
            if signature is None:
                self._data = TokenFile()
                return
            try:
                raw = json.loads(self.path.read_text("utf-8"))
                self._data = TokenFile.model_validate(raw)
            except Exception as exc:
                # Keep serving with the last good copy rather than locking everyone out.
                log.error("token_file_invalid", path=str(self.path), error=str(exc))

    def save(self) -> None:
        """Write the store atomically, creating parent directories as needed."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
            payload = self._data.model_dump_json(indent=2)
            fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix=".tokens-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(tmp_name, _FILE_MODE)
                os.replace(tmp_name, self.path)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise
            self._signature = self._file_signature()

    # ---- Queries ----

    def list_tokens(self) -> list[TokenRecord]:
        self.reload()
        with self._lock:
            return sorted(self._data.tokens.values(), key=lambda record: record.created_at)

    def get(self, token_id: str) -> TokenRecord | None:
        self.reload()
        with self._lock:
            return self._data.tokens.get(token_id)

    def authenticate(self, token: str) -> Principal | None:
        """Return the principal for a presented token, or None if it is not usable."""
        token_id = parse_token_id(token)
        if token_id is None:
            return None
        self.reload()
        with self._lock:
            record = self._data.tokens.get(token_id)
        if record is None:
            # Hash anyway so an unknown id is not measurably faster than a wrong secret.
            hash_token(token)
            return None
        if not record.verify(token):
            return None
        if not record.is_active():
            return None
        return record.to_principal()

    # ---- Mutations ----

    def create(
        self,
        name: str,
        *,
        scopes: list[Scope] | None = None,
        expires_at: datetime | None = None,
        note: str | None = None,
    ) -> tuple[TokenRecord, str]:
        """Create a token and return it with its plaintext form, shown only once."""
        with self._lock:
            self.reload(force=True)
            token_id, token = generate_token()
            while token_id in self._data.tokens:  # pragma: no cover - 48 bits of id
                token_id, token = generate_token()
            record = TokenRecord(
                id=token_id,
                name=name,
                token_hash=hash_token(token),
                scopes=list(scopes if scopes is not None else DEFAULT_SCOPES),
                created_at=utcnow(),
                expires_at=expires_at,
                note=note,
            )
            self._data.tokens[token_id] = record
            self.save()
        return record, token

    def set_disabled(self, token_id: str, disabled: bool) -> TokenRecord | None:
        with self._lock:
            self.reload(force=True)
            record = self._data.tokens.get(token_id)
            if record is None:
                return None
            record.disabled = disabled
            self.save()
            return record

    def delete(self, token_id: str) -> bool:
        with self._lock:
            self.reload(force=True)
            if self._data.tokens.pop(token_id, None) is None:
                return False
            self.save()
            return True
