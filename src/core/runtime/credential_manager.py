"""Runtime credential manager.

This module manages transient runtime credential state. Credentials tracked here
are execution-time handles and validation records; they are not persistent KG
facts.
"""

from __future__ import annotations

from src.core.models.runtime import (
    CredentialKind,
    CredentialRuntime,
    CredentialStatus,
    RuntimeState,
    utc_now,
)


class RuntimeCredentialManager:
    """Manage runtime credential lifecycle, bindings and validation status."""

    def upsert_credential(
        self,
        state: RuntimeState,
        credential_id: str,
        principal: str,
        *,
        kind: str = "password",
        secret_ref: str | None = None,
        source_session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CredentialRuntime:
        """Create or refresh one runtime credential entry."""

        now = utc_now()
        credential_kind = CredentialKind(kind)
        credential = state.credentials.get(credential_id)

        if credential is None:
            credential = CredentialRuntime(
                credential_id=credential_id,
                principal=principal,
                kind=credential_kind,
                secret_ref=secret_ref,
                source_session_id=source_session_id,
                metadata=dict(metadata or {}),
            )
            state.credentials[credential_id] = credential
        else:
            credential.principal = principal
            credential.kind = credential_kind
            credential.secret_ref = secret_ref
            credential.source_session_id = source_session_id
            if metadata:
                credential.metadata.update(dict(metadata))

        state.last_updated = now
        return credential

    def mark_valid(
        self,
        state: RuntimeState,
        credential_id: str,
        *,
        target_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CredentialRuntime:
        """Mark one credential as valid and optionally bind it to a target."""

        credential = self.get_credential(state, credential_id)
        credential.status = CredentialStatus.VALID
        credential.last_validated_at = utc_now()
        if target_id is not None:
            credential.bound_targets.add(target_id)
        if metadata:
            credential.metadata.update(dict(metadata))
        state.last_updated = credential.last_validated_at
        return credential

    def mark_invalid(
        self,
        state: RuntimeState,
        credential_id: str,
        *,
        reason: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CredentialRuntime:
        """Mark one credential as invalid and increment its failure count."""

        credential = self.get_credential(state, credential_id)
        now = utc_now()
        credential.status = CredentialStatus.INVALID
        credential.last_validated_at = now
        credential.failure_count += 1
        if reason is not None:
            credential.metadata["invalid_reason"] = reason
        if metadata:
            credential.metadata.update(dict(metadata))
        state.last_updated = now
        return credential

    def mark_expired(
        self,
        state: RuntimeState,
        credential_id: str,
        *,
        reason: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CredentialRuntime:
        """Mark one credential as expired."""

        credential = self.get_credential(state, credential_id)
        now = utc_now()
        credential.status = CredentialStatus.EXPIRED
        credential.last_validated_at = now
        if reason is not None:
            credential.metadata["expiry_reason"] = reason
        if metadata:
            credential.metadata.update(dict(metadata))
        state.last_updated = now
        return credential

    def mark_revoked(
        self,
        state: RuntimeState,
        credential_id: str,
        *,
        reason: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CredentialRuntime:
        """Mark one credential as revoked."""

        credential = self.get_credential(state, credential_id)
        now = utc_now()
        credential.status = CredentialStatus.REVOKED
        credential.last_validated_at = now
        if reason is not None:
            credential.metadata["revocation_reason"] = reason
        if metadata:
            credential.metadata.update(dict(metadata))
        state.last_updated = now
        return credential

    def bind_target(self, state: RuntimeState, credential_id: str, target_id: str) -> CredentialRuntime:
        """Attach one target binding to a tracked credential."""

        credential = self.get_credential(state, credential_id)
        credential.bound_targets.add(target_id)
        state.last_updated = utc_now()
        return credential

    def record_validation(
        self,
        state: RuntimeState,
        credential_id: str,
        *,
        status: str,
        target_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CredentialRuntime:
        """Record one validation result and update the credential state."""

        normalized = CredentialStatus(status)
        reason = None if metadata is None else str(metadata.get("reason")) if metadata.get("reason") is not None else None
        if normalized == CredentialStatus.VALID:
            return self.mark_valid(state, credential_id, target_id=target_id, metadata=metadata)
        if normalized == CredentialStatus.INVALID:
            return self.mark_invalid(state, credential_id, reason=reason, metadata=metadata)
        if normalized == CredentialStatus.EXPIRED:
            return self.mark_expired(state, credential_id, reason=reason, metadata=metadata)
        if normalized == CredentialStatus.REVOKED:
            return self.mark_revoked(state, credential_id, reason=reason, metadata=metadata)
        credential = self.get_credential(state, credential_id)
        credential.status = normalized
        credential.last_validated_at = utc_now()
        if target_id is not None:
            credential.bound_targets.add(target_id)
        if metadata:
            credential.metadata.update(dict(metadata))
        state.last_updated = credential.last_validated_at
        return credential

    def get_credential(self, state: RuntimeState, credential_id: str) -> CredentialRuntime:
        """Return one tracked runtime credential."""

        try:
            return state.credentials[credential_id]
        except KeyError as exc:
            raise ValueError(f"credential '{credential_id}' does not exist") from exc

    def is_credential_usable_for_target(
        self,
        state: RuntimeState,
        credential_id: str,
        *,
        target_id: str | None = None,
    ) -> bool:
        """Return True when the credential is valid for the optional target."""

        credential = self.get_credential(state, credential_id)
        if not credential.is_usable():
            return False
        if target_id is None or not credential.bound_targets:
            return True
        return target_id in credential.bound_targets

    def list_credentials_for_session(self, state: RuntimeState, session_id: str) -> list[CredentialRuntime]:
        """Return credentials discovered from the given source session."""

        return sorted(
            [credential for credential in state.credentials.values() if credential.source_session_id == session_id],
            key=lambda item: item.credential_id,
        )

    # 中文注释：
    # 会话失效后，由该会话导出的临时凭据也要同步过期，避免调度继续复用脏凭据。
    def expire_credentials_for_session(
        self,
        state: RuntimeState,
        session_id: str,
        *,
        reason: str | None = None,
    ) -> int:
        """Expire credentials sourced from the given session."""

        expired = 0
        for credential in self.list_credentials_for_session(state, session_id):
            self.mark_expired(state, credential.credential_id, reason=reason or "source_session_expired")
            expired += 1
        return expired


__all__ = ["RuntimeCredentialManager"]
