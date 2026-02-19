# Análisis de Seguridad y Fugas de Datos de OpenClaw

> Documento generado el 2026-02-13.
> Caso de uso evaluado: **uso personal para búsqueda de empleo** (CVs, cartas de presentación,
> datos salariales, información personal).

---

## Resumen Ejecutivo

**Nivel de riesgo global: BAJO-MODERADO**

OpenClaw es un gateway **local-first** y **self-hosted**. Su arquitectura favorece la privacidad
por defecto. Sin embargo, hay consideraciones importantes cuando se manejan datos sensibles como
CVs, cartas de motivación e información salarial.

**Veredicto**: Es seguro para datos de búsqueda de empleo **si se configura correctamente**
(ver recomendaciones al final).

---

## 1. Transmisión de Datos

### 1.1 Arquitectura Local-First (POSITIVO)

- El gateway es un proceso **local** en `ws://127.0.0.1:18789`
- Los datos de conversación **NO salen del dispositivo** salvo que se configure un proveedor IA externo
- Sin componentes cloud obligatorios

### 1.2 Llamadas a APIs de Modelos IA (PUNTO CRITICO)

Cuando pides al asistente que revise tu CV, el texto se envía al proveedor de IA configurado:

| Proveedor | URL destino | Riesgo para CVs |
|-----------|------------|-----------------|
| **Ollama (local)** | `http://127.0.0.1:11434` | BAJO - nunca sale del PC |
| **Anthropic Claude** | `https://api.anthropic.com` | MEDIO - sale del PC pero Anthropic no entrena con datos de API |
| **OpenAI GPT** | `https://api.openai.com/v1` | ALTO - OpenAI puede logear/usar para mejoras |
| **Google Gemini** | APIs de Google | MEDIO - política de logging incierta |
| **OpenRouter** | `https://openrouter.ai/api/v1` | MEDIO - intermediario, múltiples destinos |
| **AWS Bedrock** | AWS regional endpoints | MEDIO-BAJO - dentro de tu cuenta AWS |

**Flujo con modelo en la nube**:
```
Tu CV → OpenClaw local → HTTPS → api.anthropic.com → Claude procesa → respuesta
         (tu PC)                  (servidores de Anthropic)
```

**Flujo con modelo local (Ollama)**:
```
Tu CV → OpenClaw local → HTTP local → Ollama → respuesta
         (tu PC)          (tu PC)      (tu PC)
         ✓ TODO en tu máquina
```

### 1.3 Consultas de Uso (analytics de proveedor)

OpenClaw consulta estadísticas de uso de los proveedores:
- `https://api.anthropic.com/api/oauth/usage`
- `https://claude.ai/api/organizations/{orgId}/usage`
- APIs de uso de OpenAI, Gemini, GitHub Copilot

**Solo envía tokens de autenticación**, no contenido de conversaciones.

### 1.4 Herramientas con Acceso a Red

| Herramienta | Qué envía | A dónde |
|-------------|-----------|---------|
| **web-fetch** | URLs solicitadas | Firecrawl (`api.firecrawl.dev`) o directo |
| **web-search** | Consultas de búsqueda | Perplexity / OpenRouter |
| **browser** | Navegación web | Sitios web visitados |
| **image-tool** | Imágenes para análisis | API de VLM (MiniMax, etc.) |

Si el agente usa estas herramientas con datos de tu CV, **esos datos podrían enviarse** a
servicios externos. Esto depende de lo que el agente decida hacer.

---

## 2. Almacenamiento de Datos

### 2.1 Sesiones (transcripciones)

**Ubicación**: `~/.openclaw/agents/<agentId>/sessions/`
**Formato**: `.jsonl` (JSON Lines, texto plano)
**Contenido**: historial completo de conversaciones incluyendo mensajes del usuario y respuestas IA

**Cifrado en disco: NO**

Los archivos de sesión se almacenan en **texto plano**. Si alguien accede a tu disco,
puede leer todas tus conversaciones (incluyendo CVs que hayas compartido).

**Mitigación**: Usar cifrado de disco completo (BitLocker en Windows).

### 2.2 Memoria Vectorial

**Ubicación**: Base de datos SQLite local (por workspace de agente)
**Contenido**: embeddings y fragmentos de texto indexados

Misma situación: **sin cifrado a nivel de aplicación**, depende del cifrado de disco.

### 2.3 Configuración y Credenciales

**Ubicación**: `~/.openclaw/config.json`, `~/.openclaw/credentials/`, `~/.openclaw/auth-profiles/`

Las API keys se almacenan en archivos de configuración. Aunque se redactan en los logs,
están **en texto plano en disco**.

### 2.4 Archivos Temporales

Media (imágenes, audio, vídeo) se almacenan temporalmente durante el procesamiento
y se eliminan después. Ciclo de vida gestionado por `src/memory/internal.js`.

---

## 3. Telemetría y Tracking

### 3.1 Telemetría Incorporada

- **OpenTelemetry**: Disponible como extensión (`extensions/diagnostics-otel/`)
- **Estado por defecto: DESACTIVADA**
- Si se activa, envía trazas/métricas a un endpoint OTLP **que tú configuras**
- No envía datos a terceros sin configuración explícita

### 3.2 Analytics de Terceros

- **NO hay** Mixpanel, Amplitude, Sentry, DataDog, PostHog ni similar
- **NO hay** tracking pixels
- **NO hay** reporte de errores a terceros
- **NO hay** estadísticas de uso enviadas "a casa"

**RESULTADO: EXCELENTE** - Sin telemetría oculta.

---

## 4. Sistema de Logging

### 4.1 Qué se logea

**Ubicación**: `~/.openclaw/openclaw.log`
**Rotación**: Logs de 24 horas con auto-eliminación

### 4.2 Redacción de Datos Sensibles

**Archivo**: `src/logging/redact.ts`

OpenClaw tiene un sistema **robusto** de redacción en logs que oculta:

- API keys (`API_KEY`, `TOKEN`, `SECRET`, `PASSWORD`)
- Bearer tokens
- Claves PEM privadas
- Tokens de GitHub (`ghp_*`, `github_pat_*`)
- Tokens de Slack (`xox[baprs]-*`)
- API keys de Google (`AIza*`)
- Keys de OpenAI (`sk-*`)
- Y muchos más patrones

**Lo que NO aparece en logs**:
- Contenido completo de conversaciones (redactado)
- PII de mensajes de usuario
- API keys (redactadas)

---

## 5. Seguridad de Red

### 5.1 Superficie de Ataque

El gateway expone:
- **WebSocket** en `127.0.0.1:18789` (solo localhost por defecto)
- **HTTP** para API, Control UI, Canvas

**Por defecto: NO accesible desde internet** (bind a localhost).

### 5.2 Autenticación del Gateway

- Token bearer o password para acceso al gateway
- Scopes de permisos (`operator.admin`, `operator.read`, `operator.write`, etc.)
- Pairing de dispositivos con clave pública Ed25519

### 5.3 Webhooks

Telegram y Google Chat requieren endpoints HTTPS públicos para webhooks.
**Solo el path específico del webhook se expone**, no todo el gateway.

---

## 6. Seguridad de Canales de Mensajería

### 6.1 Por Canal

| Canal | Cifrado en tránsito | Quién ve los mensajes |
|-------|--------------------|-----------------------|
| **Signal** | E2E | Solo tú y el destinatario |
| **WhatsApp** | E2E (pero OpenClaw ve texto plano local) | WhatsApp servers + tú |
| **iMessage** | E2E (via BlueBubbles) | Apple + tú |
| **Telegram** | TLS (no E2E por defecto) | Telegram servers + tú |
| **Discord** | TLS | Discord servers + tú |
| **Slack** | TLS | Slack + tu workspace admin |
| **WebChat** | Depende de config | Solo tú (si localhost) |

### 6.2 Política de DMs

- **Modo `pairing`** (por defecto): desconocidos reciben código de emparejamiento,
  el bot NO procesa su mensaje hasta aprobación
- **Modo `open`**: procesa DMs de cualquiera (requiere opt-in explícito)

---

## 7. Dependencias de Terceros

### 7.1 Paquetes Analizados

**Mensajería** (seguros, solo implementación de protocolos):
- `@whiskeysockets/baileys` (WhatsApp)
- `grammy` (Telegram)
- `@slack/bolt` (Slack)
- `discord-api-types` (Discord)

**Utilidades** (seguros):
- `sharp` (procesamiento de imágenes)
- `pdfjs-dist` (parsing de PDF)
- `zod` (validación de schemas)

**Paquetes sospechosos encontrados: NINGUNO**

---

## 8. Auditoría de Seguridad Integrada

OpenClaw incluye herramientas de auditoría:

```bash
openclaw security audit           # Comprobar problemas de seguridad
openclaw security audit --deep    # Prueba profunda (incluye conectividad gateway)
openclaw security audit --fix     # Auto-corregir problemas comunes (permisos)
openclaw doctor                   # Diagnóstico general del sistema
```

**Comprobaciones que realiza**:
- Permisos de archivos
- Secretos en archivos de configuración
- Seguridad de código de plugins
- Higiene de modelos (modelos pequeños con herramientas peligrosas)
- Resumen de superficie de ataque
- Políticas de DM
- Configuración de webhooks

---

## 9. Tabla de Riesgo por Escenario

| Escenario | Riesgo | Notas |
|-----------|--------|-------|
| Ollama local | BAJO | Todo en tu máquina |
| Anthropic Claude API | MEDIO | Sale del PC; Anthropic dice no entrenar con datos de API |
| OpenAI GPT-4 | ALTO | Puede logear/usar para mejoras de modelo |
| Google Gemini | MEDIO | Política de logging incierta |
| Almacenamiento local (sesiones) | MEDIO | Texto plano; necesita cifrado de disco |
| Webhooks (Telegram) | MEDIO | Requiere exposición HTTPS limitada |
| Slack/Discord | MEDIO | Plataformas ven mensajes (esperado) |
| Telemetría por defecto | BAJO | Desactivada |
| Dependencias npm | BAJO | Sin paquetes sospechosos |

---

## 10. Recomendaciones para Búsqueda de Empleo

### HACER

1. **Usar Ollama con modelo local** (la opción más segura)
   ```bash
   # Instalar: https://ollama.ai
   ollama pull llama3.1    # o mistral, qwen2.5, etc.
   # Configurar OpenClaw para usar localhost:11434
   ```

2. **Si necesitas modelo cloud, usar Anthropic Claude**
   - Política de privacidad favorable (no entrenan con datos de API)
   - Mejor calidad para texto largo (CVs, cartas)

3. **Activar cifrado de disco completo**
   - Windows: BitLocker
   - macOS: FileVault
   - Linux: LUKS

4. **Crear workspace aislado** para búsqueda de empleo
   ```
   ~/.openclaw/agents/
   ├── main/           # Uso general
   └── job-search/     # AISLADO para CVs
   ```

5. **API keys en variables de entorno** (no en config.json)
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   export OLLAMA_BASE_URL="http://localhost:11434"
   ```

6. **Ejecutar auditoría de seguridad** antes de manejar datos sensibles
   ```bash
   openclaw security audit --fix
   ```

7. **Verificar que el gateway es solo localhost**
   ```bash
   netstat -an | grep 18789  # Debe mostrar 127.0.0.1:18789
   ```

8. **Usar política de DM restrictiva** (pairing mode)

9. **Limpiar sesiones** después de terminar la búsqueda de empleo
   ```bash
   openclaw sessions delete <session-key>
   ```

### NO HACER

1. **NO usar OpenAI GPT** con datos de CV (pueden logear para mejoras de modelo)
2. **NO almacenar** CV en workspace compartido
3. **NO activar telemetría** sin entender a dónde van los datos
4. **NO exponer** el gateway a internet sin autenticación
5. **NO guardar** API keys hardcodeadas en config.json
6. **NO compartir** dispositivo sin cifrado de disco
7. **NO enviar** datos salariales por canales como Slack/Discord (los admins pueden verlos)

---

## 11. Modelo de Confianza

| Aspecto | Evaluación |
|---------|-----------|
| **Código abierto** | MIT - auditable |
| **Componentes cerrados** | Ninguno encontrado |
| **Ejecución local** | Tú controlas la infraestructura |
| **Privacidad por defecto** | Telemetría desactivada |
| **Configuración explícita** | Servicios externos requieren opt-in |
| **Redacción en logs** | Robusta y extensiva |
| **Cifrado en reposo** | No (depende del SO) |
| **Auditoría integrada** | Sí (`security audit`) |

---

## 12. Conclusión

**OpenClaw es seguro para búsqueda de empleo SI**:

- Usas **Ollama local** o **Anthropic Claude**
- Tu dispositivo tiene **cifrado de disco**
- El gateway está en **localhost**
- Las API keys están en **variables de entorno**
- Ejecutas en un **dispositivo personal** (no compartido)
- Usas **pairing mode** para DMs
- Ejecutas `openclaw security audit` periódicamente

**NO es seguro SI**:

- Usas OpenAI con logging habilitado
- Expones el gateway a internet
- No tienes cifrado de disco
- Compartes el dispositivo

El mayor riesgo no es OpenClaw en sí, sino **el proveedor de IA que elijas**.
OpenClaw es solo el intermediario; la decisión crítica es si tus datos salen de tu máquina o no.
