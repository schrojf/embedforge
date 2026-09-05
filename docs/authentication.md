# Authentication

Clients authenticate with a bearer token:

```
Authorization: Bearer ef_de5ba8d6d33e_7zHw2z...
```

## Token format and storage

A token is `ef_<id>_<secret>`: an `ef_` namespace marker so a leaked token is
recognizable and greppable, a 12-character id, and 256 bits of random secret.

The server stores **only a SHA-256 digest** of the whole token, plus the id in the
clear. Verification is a dict lookup and one constant-time comparison, which is why it
adds nothing measurable to request latency. A password hash such as argon2 would be the
wrong tool here: there is no low-entropy password to protect, and it would put tens of
milliseconds on every request.

Consequence: **a lost token cannot be recovered, only replaced.**

Tokens live in a JSON file (`EMBEDFORGE_TOKENS_FILE`, `/var/lib/embedforge/tokens.json`
in the container). It is written atomically at mode 0600. A file suits data that is
small, changes rarely, and needs to be trivially backed up.

## Scopes

| Scope | Grants |
| --- | --- |
| `embed` | `POST /v1/embed` |
| `query` | `POST /v1/query` |
| `models:read` | `GET /v1/models` |
| `admin` | Everything, including scopes added later. |

New tokens get `embed,query,models:read`. `GET /v1/user` works with any valid token.

Give each client the narrowest set that works — a search frontend usually needs only
`query`, while an indexing job needs only `embed`.

## Managing tokens

```bash
embedforge token create indexer --scopes embed --expires-in 90d
embedforge token list
embedforge token show de5ba8d6d33e
embedforge token disable de5ba8d6d33e   # reversible, keeps the audit record
embedforge token enable  de5ba8d6d33e
embedforge token revoke  de5ba8d6d33e   # permanent
```

`--expires-in` takes a duration (`30d`, `12h`, `90m`, `2w`) or an ISO date
(`2027-01-01`). Naive dates are read as UTC.

The plaintext token is printed **once**, at creation. See [cli.md](cli.md) for the full
command reference.

## Changes take effect without a restart

The server re-reads the token file when its modification time changes, at most once per
`EMBEDFORGE_TOKENS_RELOAD_INTERVAL` (5 seconds by default). So creating or revoking a
token from the CLI — or from another container writing to the same volume — applies
within a few seconds, with no restart and no dropped connections.

Two consequences worth knowing:

- A token created a moment ago may 401 for up to that interval. Set the interval to `0`
  to check on every request if that matters more than the file stats.
- **Revocation is not instant.** Budget the interval when you revoke a compromised
  token, and restart the process if you need it immediate.

If the file is missing, the server starts and refuses every request; if it becomes
corrupt, the server keeps serving with the last good copy and logs `token_file_invalid`,
because locking every client out is the worse failure.

## Running without authentication

`EMBEDFORGE_AUTH_ENABLED=false` accepts every request as an anonymous principal with
full scopes, and `/v1/user` reports `"authenticated": false`. This is for local
development only — anything reachable by anything else needs tokens.

## Operational notes

- **Back up the token file.** Tokens cannot be regenerated from the digests.
- **Terminate TLS in front of the server.** A bearer token over plain HTTP is a token in
  everyone's logs. See [deployment.md](deployment.md).
- Tokens do not appear in logs. Access logs record the token *id*, which is what you
  want when tracing a client's traffic.
- There is no per-token rate limiting yet. Overload protection is global and covered in
  [performance.md](performance.md); per-client quotas belong at the reverse proxy for now.
