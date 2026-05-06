# Runbook: RBAC token expired

**Severity**: P1 if customer-facing; P2 otherwise.
**Owner**: Security / platform.
**Last reviewed**: 2026-05-06 (V130).

The crate's `access_control` module enforces RBAC over API/MCP
endpoints. Tokens are typically issued by the deployment's identity
layer (the crate validates and applies policy; it does not mint).
Expiry symptoms look identical to revocation, so distinguish those
two early.

## 1. Symptoms

* HTTP 401 (`token expired` / `invalid token`) on requests that
  worked previously.
* HTTP 403 (`insufficient scope`) — *different cause*, see "Likely
  causes" #4 below.
* Audit log: spike in `AuditEventType::AccessDenied`.
* Metric `auth_failures_total{reason="expired"}` increments.
* Customer reports "everyone in team X can't log in" right after a
  rotation event.

## 2. Likely causes

| # | Cause | Frequency |
|---|---|---|
| 1 | TTL elapsed normally; client didn't refresh | high |
| 2 | Deploy rotated the signing key, old tokens invalidated | medium |
| 3 | Clock skew on the validating host (token "not yet valid") | medium |
| 4 | Policy change tightened scopes (this is *403*, not *401*) | medium |
| 5 | Token revoked by an admin via `access_control revoke` | low |
| 6 | The whole identity provider is down | low |

## 3. Diagnose

Decode the token (do **not** paste a customer token into chat; use
a local terminal):

```bash
# Inspect a JWT-style token without validating signature.
echo "<token>" | cut -d. -f2 | base64 -d | jq .

# Crate-side: print policy decision for the principal.
ai_cli auth describe --token-file /tmp/token.txt
ai_cli auth check --principal alice@example.com --resource rag.query
```

Validate clock and signing-key state:

```bash
# Time on the host that's failing.
date -u

# Active signing keys (if your identity layer exposes a JWKS):
curl -s https://<idp>/.well-known/jwks.json | jq '.keys | map(.kid)'
# Compare with the token's `kid` header:
echo "<token>" | cut -d. -f1 | base64 -d | jq -r .kid
```

If `kid` is not in JWKS, that's cause #2 (rotation).

## 4. Mitigate

**A. Genuine TTL expiry:**
- This is not an outage; the client SDK should auto-refresh. If a
  *batch* of clients hit it simultaneously, suspect a thundering-
  herd refresh — see "Resolve" #1.

**B. Key rotation cut off live tokens:**
- Re-publish the previous signing key alongside the new one for at
  least one full TTL window (`overlap = max_ttl + slack`).
- If the rotation was unintentional or premature, redeploy with the
  prior `JWT_PUBLIC_KEYS` list. Tokens that were valid before the
  rotation should be valid again immediately.

**C. Clock skew:**
- Resync NTP (`systemctl restart systemd-timesyncd` /
  `w32tm /resync`).
- Increase the validation `leeway` in your config to ±60 seconds —
  the crate already supports it; check `[access_control].clock_skew_ms`.

**D. Policy tightening (403, not 401):**
- Check the most recent change to your policy file:
  `git log -p -- policies/access_control.yaml | head -80`
- Roll back if accidental; otherwise communicate to affected users.

**E. Identity provider down:**
- Out of scope for this crate. Follow your IdP runbook.

## 5. Resolve

* **Stagger refresh** — if many tokens expire at the same minute
  (e.g. issued during a deploy), add jitter to the refresh interval
  on the client. Server-side, you can also accept tokens for
  `original_ttl + grace` to absorb the herd.
* **Rotate keys with overlap** — never publish a new signing key
  *and* immediately remove the old one. The standard pattern: new
  key first, wait one full token TTL, then retire the old one.
* **Alert on `auth_failures_total{reason="expired"}` rate** — a
  steady low rate is normal (clients refreshing late). A sudden
  step is a deploy event; correlate with your release log.
* **Clock-monitoring** — alert if `|now - ntp_now| > 2s` on any
  validator. Expired-token errors caused by clock skew are the
  worst kind: confusing, intermittent, and customer-visible.

## 6. Postmortem

Log:

| Field | Value |
|---|---|
| Detection | metric / customer report / monitor |
| Affected principals | scope of impact |
| Cause | from §2 |
| Mitigation applied | from §4 |
| Customer impact | requests denied, duration |
| Action items | owner + due date |
