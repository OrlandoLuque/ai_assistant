# Security Audit — ai_assistant v0.1.0

**Fecha**: 10 marzo 2026
**Alcance**: 203.427+ LOC, 220+ archivos .rs, 816 dependencias
**Metodología**: Análisis estático exhaustivo (5 dominios en paralelo) + `cargo audit`

---

## Resumen

| Severidad | Total | Resueltos |
|-----------|-------|-----------|
| CRITICAL  | 5     | 0         |
| HIGH      | 12    | 0         |
| MEDIUM    | 14    | 0         |

---

## CRITICAL

- [ ] **C1. Inyección de comandos — `run_command`** (`src/os_tools.rs:285-294`)
  El parámetro `cmd` se pasa directamente a `sh -c` / `cmd /C`. El `SandboxValidator`
  filtra patrones conocidos pero es bypassable con ofuscación o comandos alternativos.
  **Impacto**: Ejecución de código arbitrario en el host.
  **Remedio**: Eliminar el tool genérico `run_command` o usar whitelist estricta
  con argumentos separados (sin shell).

- [ ] **C2. Inyección de comandos — Docker exec** (`src/container_tools.rs:260`)
  `docker exec container_id sh -c <command>` con `command` sin sanitizar.
  **Impacto**: Ejecución arbitraria dentro de containers.
  **Remedio**: Pasar argumentos como array:
  `Command::new("docker").args(["exec", id, cmd_part1, ...])`.

- [ ] **C3. `AutoApproveAll` bypassa HITL** (`src/agent_policy.rs:102-108`)
  Siempre retorna `true`, anulando completamente la aprobación humana.
  Si se configura por error, el agente ejecuta acciones destructivas sin control.
  **Remedio**: Feature-gate con `#[cfg(test)]` y obligar handler explícito en prod.

- [ ] **C4. SSRF vía MCP tool calls** (`src/mcp_client.rs:440-444`)
  `call_tool()` pasa argumentos del usuario directamente al servidor MCP sin validar.
  Herramientas como `http_get` permiten escanear redes internas o exfiltrar datos.
  **Remedio**: Filtro anti-SSRF (rechazar IPs privadas/localhost). Whitelist de tools.

- [ ] **C5. Servicios escuchando en 0.0.0.0 por defecto**
  `src/bin/ai_proxy.rs:178`, `src/bin/ai_cluster_node.rs:136,171`,
  `src/distributed_network.rs:68`, `src/p2p.rs:301,420,534`.
  Servicios se vinculan a todas las interfaces de red, exponiéndose a la red.
  **Remedio**: Default a `127.0.0.1`. Requerir configuración explícita para 0.0.0.0.

---

## HIGH

- [ ] **H1. CORS wildcard por defecto** (`src/server.rs:148-165`)
  `allowed_origins: vec!["*"]` permite cualquier origen. La combinación
  `wildcard + allow_credentials` no está validada.
  **Remedio**: Default `vec![]` vacío. Rechazar wildcard+credentials.

- [ ] **H2. Rate limiter global, sin per-IP** (`src/server.rs:927-980`)
  Un solo contador atómico global. Un atacante agota el límite para todos.
  **Remedio**: Bucketing por IP con LRU eviction.

- [ ] **H3. Sin validación de X-Forwarded-For** (`src/server.rs:1728-1783`)
  Detrás de reverse proxy, no se extrae ni valida la IP del cliente.
  **Remedio**: Parsear `X-Forwarded-For` solo con `trust_proxy: true`.

- [ ] **H4. IP range permisivo si falta IP** (`src/access_control.rs:327-337`)
  Si `current_request_ip` es `None`, las condiciones `IpRange` se omiten (allow).
  **Remedio**: Deny-by-default: `return Some("IP not available")`.

- [ ] **H5. Path traversal — SharedFolder** (`src/shared_folder.rs:132-177`)
  `get_file()`, `put_file()`, `delete_file()` no validan `../` en paths relativos.
  **Remedio**: Rechazar paths con `..` o `/` absoluto. Canonicalizar.

- [ ] **H6. Path traversal — Agent policy** (`src/agent_policy.rs:237-259`)
  `can_access_path()` usa `starts_with()` sin canonicalizar.
  **Remedio**: `std::fs::canonicalize()` antes de comparar.

- [ ] **H7. ReDoS — Content moderation** (`src/content_moderation.rs:271`)
  `add_pattern()` acepta regex arbitrarios del usuario. Patrones con backtracking
  catastrófico causan DoS.
  **Remedio**: Timeout en compilación de regex o matching sin backtracking.

- [ ] **H8. Container bind mounts sin validar** (`src/container_executor.rs:221`)
  `bind_mounts` acepta paths arbitrarios incluyendo `/` del host.
  **Remedio**: Whitelist de paths permitidos para montar.

- [ ] **H9. WebSocket sin validación de Origin** (`src/server.rs:1273-1280`)
  El upgrade a WebSocket se acepta sin verificar el Origin contra `allowed_origins`.
  **Remedio**: Validar Origin antes del upgrade.

- [ ] **H10. API keys en config como String** (`src/config_file.rs`)
  Las API keys de proveedores son `String`, no `SecureString`. Se almacenan en
  texto plano en ficheros de configuración.
  **Remedio**: Migrar a `SecureString` o cifrar el fichero de config.

- [ ] **H11. Plugins interceptan/modifican mensajes sin restricción** (`src/plugins.rs:55-62`)
  `on_before_send` y `on_after_receive` permiten a un plugin malicioso alterar
  contenido sin restricciones ni capabilities.
  **Remedio**: Sistema de capabilities por plugin.

- [ ] **H12. PKCE débil — hash simple en vez de SHA-256** (`src/mcp_protocol/oauth.rs:178-193`)
  PKCE challenge usa `u64` multiply-add hash y devuelve method `"plain"` en vez de
  `"S256"`. No protege contra interceptación del code.
  **Remedio**: Reemplazar con SHA-256 real + method `"S256"`.

---

## MEDIUM

- [ ] **M1. Header injection (CRLF) en respuestas HTTP** (`src/server.rs:1756-1760`)
  Valores de cabecera CORS y `request_id` no se validan para `\r\n`.

- [ ] **M2. Auth deshabilitada por defecto** (`src/server.rs:111-119`)
  `AuthConfig { enabled: false, .. }`. Decisión de diseño, pero peligrosa si se
  olvida habilitar en producción.

- [ ] **M3. Sin session timeout/expiración** (`src/session.rs:50-100`)
  Las sesiones tienen timestamps pero no expiran nunca.
  **Remedio**: Añadir `.is_expired(max_age)` y enforcar en middleware.

- [ ] **M4. PII no redactada en mensajes de error al cliente** (`src/server.rs:1026,2071`)
  Si un error incluye datos del usuario, el PII se filtra al cliente.

- [ ] **M5. Error messages exponen detalles internos** (`src/server.rs:1026,1037,1063`)
  `format!("Invalid JSON: {}", e)` expone detalles de serde_json.
  **Remedio**: Mensajes genéricos al cliente, detalles solo en logs internos.

- [ ] **M6. SQLite con permisos por defecto (644)** (`src/persistence.rs`)
  Ficheros de base de datos legibles por cualquier usuario local.
  **Remedio**: `fs::set_permissions()` a 600 tras crear.

- [ ] **M7. P2P TCP sin cifrado fuera de QUIC** (`src/p2p.rs:301,420,534`)
  Conexiones TCP raw para P2P transport. STUN/SSDP/NAT-PMP en UDP sin cifrar.

- [ ] **M8. Container CPU quota default = 0 (ilimitado)** (`src/container_executor.rs:149`)
  Código malicioso en container puede consumir 100% CPU indefinidamente.

- [ ] **M9. Container NetworkMode::Host posible sin validar** (`src/container_executor.rs:152-169`)
  Un agente o plugin puede setear `NetworkMode::Host` sin restricción.

- [ ] **M10. Cost limits no atómicos** (`src/agent_sandbox.rs:193-208`)
  Race condition: llamadas concurrentes pueden exceder el presupuesto.

- [ ] **M11. Sandbox env permite PATH completo** (`src/code_sandbox.rs:337-339`)
  El PATH del sistema se añade incondicionalmente al sandbox.

- [ ] **M12. Limpieza de archivos temp ignora fallos** (`src/code_sandbox.rs:362-363`)
  `let _ = std::fs::remove_file(...)` ignora errores silenciosamente.

- [ ] **M13. Prompt injection no detectada por guardrails** (`src/guardrail_pipeline.rs`)
  Pipeline detecta contenido tóxico pero no intentos de jailbreak.

- [ ] **M14. PII detection: falsos negativos** (`src/pii_detection.rs:209`)
  Patrón SSN simplista. Faltan tipos: pasaportes, licencias conducir, tax IDs.

---

## Dependencias (cargo audit)

- [ ] **quinn-proto 0.11.13** — HIGH (8.7): DoS en endpoints (RUSTSEC-2026-0037).
  Solución: `cargo update -p quinn-proto` (>=0.11.14).

- [ ] **time 0.3.46** — MEDIUM (6.8): Stack exhaustion DoS (RUSTSEC-2026-0009).
  Solución: `cargo update -p time` (>=0.3.47).

- [ ] **lru 0.12.5** — Warning (unsound): Stacked Borrows violation (RUSTSEC-2026-0002).
  Transitiva vía tantivy. Sin acción directa posible.

- [ ] **bincode 1.3.3** — Warning: Unmaintained (RUSTSEC-2025-0141).
  Considerar migrar a bincode 2.x.

- [ ] **rustls-pemfile 2.2.0** — Warning: Unmaintained (RUSTSEC-2025-0134).
  Migrar a `rustls-pem`.

- [ ] **paste 1.0.15** — Warning: Unmaintained (RUSTSEC-2024-0436).
  Transitiva vía datafusion/lance. Sin acción directa.

---

## Puntos Fuertes

La auditoría confirmó implementaciones de seguridad excelentes en:

- **AES-256-GCM** con nonces OsRng (`encrypted_knowledge.rs`)
- **SecureString** con zeroización volatile + constant-time eq (`secure_credentials.rs`)
- **Log redaction** para API keys, Bearer tokens, passwords (`log_redaction.rs`)
- **Mutual TLS** Ed25519 para P2P/QUIC (`node_security.rs`)
- **Challenge-response** HMAC-SHA256 constant-time (`node_security.rs`)
- **RBAC** con condiciones granulares: TimeRange, IpRange, MaxUsage, MFA (`access_control.rs`)
- **SQL parametrizado** en todas las consultas rusqlite
- **Join tokens** con expiración temporal + límite de usos (`node_security.rs`)
- **Constant-time token comparison** en auth HTTP (`server.rs`)
- **Request body size limit** (1MB default) contra DoS
- **Zero secrets hardcoded** en 313 archivos .rs revisados
- **Zero unsafe** excepto `write_volatile` en zeroización (correcto y necesario)

---

## Plan de Remediación

### Fase 1 — Inmediato (antes de producción)
- C1-C5: Inyección de comandos, SSRF, AutoApproveAll, bind addresses
- Dependencias: quinn-proto, time

### Fase 2 — Corto plazo
- H1-H4: CORS, rate limiter per-IP, X-Forwarded-For, IP range fail-closed
- H5-H6: Path traversal (SharedFolder, agent policy)
- H9: WebSocket Origin validation

### Fase 3 — Medio plazo
- H7-H8, H10-H12: ReDoS, bind mounts, API keys encryption, plugins, PKCE
- M1-M14: Headers, auth default, sessions, PII, errors, SQLite perms, etc.
- Dependencias: bincode, rustls-pemfile
