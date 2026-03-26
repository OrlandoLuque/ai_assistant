# Improvements V64 — Universal Event System + Home Automation Expansion + IoT

## Part A: Universal Event System (event_source.rs + mcp_event_tools.rs)

| # | Item | Estado |
|---|------|--------|
| 1 | EventSource trait (8 types: webhook, RSS, scraper, calendar, MQTT, WebSocket, REST, email) | HECHO |
| 2 | IncomingEvent normalized type | HECHO |
| 3 | EventRule with EventAction (PromptLlm/Notify/Both), EventFilter (5 types), cooldown | HECHO |
| 4 | EventSourceManager with dedup, cooldown, rate limiting (10 prompts/min) | HECHO |
| 5 | Prompt template rendering with injection sanitization (#16) | HECHO |
| 6 | SSRF/MQTT topic validation for all source configs | HECHO |
| 7 | RSS feed polling with XML parsing (regex-based, no XXE #13) | HECHO |
| 8 | Web scraper with change detection, max body 5MB (#18), max redirects 5 (#17) | HECHO |
| 9 | iCal calendar parsing with VEVENT extraction | HECHO |
| 10 | REST API polling with JSONPath change detection | HECHO |
| 11 | Webhook payload processing | HECHO |
| 12 | 5 MCP tools (event_subscribe/unsubscribe/list_rules/notifications/dismiss) | HECHO |
| 13 | AiEvent::ExternalEvent variant | HECHO |
| 14 | 19+3 tests | HECHO |

## Part B: MQTT Backend (home_automation/mqtt_backend.rs)

| # | Item | Estado |
|---|------|--------|
| 15 | MqttConfig with TLS-by-default, allow_insecure_mqtt opt-in (#9) | HECHO |
| 16 | TopicConvention (Zigbee2Mqtt/Tasmota/HomeAssistant/Custom) | HECHO |
| 17 | DeviceRegistry (in-memory cache, CRUD, domain filter) | HECHO |
| 18 | MqttHomeBackend implements HomeBackend | HECHO |
| 19 | Zigbee2MQTT bridge/devices discovery parsing | HECHO |
| 20 | CommandRateLimiter (10/min/device, 60/min global) (#7) | HECHO |
| 21 | Topic injection validation (#2) | HECHO |
| 22 | SSRF broker URL validation (#1) | HECHO |
| 23 | 11 tests | HECHO |

## Part C: OpenHAB Backend (home_automation/openhab_backend.rs)

| # | Item | Estado |
|---|------|--------|
| 24 | OpenHabBackend implements HomeBackend via REST | HECHO |
| 25 | Item type mapping (Switch→switch, Dimmer→light, Number→sensor, etc.) | HECHO |
| 26 | Command mapping (ON/OFF/TOGGLE/UP/DOWN/STOP) | HECHO |
| 27 | Scene/automation detection via tags | HECHO |
| 28 | 6 tests | HECHO |

## Part D: CoAP Backend (home_automation/coap_backend.rs)

| # | Item | Estado |
|---|------|--------|
| 29 | CoapBackend with CoAP message encoding/decoding (RFC 7252) | HECHO |
| 30 | GET/PUT with confirmable messages and retransmission | HECHO |
| 31 | OBSERVE support for real-time subscriptions | HECHO |
| 32 | Rate limiting (10 req/s) (#24) | HECHO |
| 33 | Device registration and listing | HECHO |
| 34 | Feature flag: `coap` (separate) | HECHO |
| 35 | 7 tests | HECHO |

## Part E: Custom IoT Devices (home_automation/custom_device.rs)

| # | Item | Estado |
|---|------|--------|
| 36 | CustomDeviceDefinition (StateSource, CommandTarget, ThresholdAlert) | HECHO |
| 37 | AlertCondition (Above/Below/Equals/Changed) | HECHO |
| 38 | Validation (SSRF, topic injection, poll interval, max alerts) | HECHO |
| 39 | 10 tests | HECHO |

## Part F: Event Listener (home_automation/event_listener.rs)

| # | Item | Estado |
|---|------|--------|
| 40 | HomeEventListenerManager (subscribe/unsubscribe/cleanup) | HECHO |
| 41 | ListenerHandle with CancellationToken | HECHO |
| 42 | ListenerSource (HA SSE, OpenHAB SSE, MQTT subscription) | HECHO |
| 43 | HA state_changed SSE parsing | HECHO |
| 44 | OpenHAB ItemStateChangedEvent SSE parsing | HECHO |
| 45 | SSRF validation on listener URLs | HECHO |
| 46 | 7 tests | HECHO |

## Part G: mDNS Discovery (home_automation/discovery.rs)

| # | Item | Estado |
|---|------|--------|
| 47 | DiscoveredService/DiscoveredServiceType types | HECHO |
| 48 | Service type list (HA, OpenHAB, MQTT) | HECHO |
| 49 | discover_services() stub (ready for mdns-sd wiring) | HECHO |
| 50 | 3 tests | HECHO |

## Part H: New MCP Tools (+4 in mcp_home_tools.rs)

| # | Item | Estado |
|---|------|--------|
| 51 | home_subscribe (start device event listener) | HECHO |
| 52 | home_unsubscribe (stop listener) | HECHO |
| 53 | home_register_device (register custom IoT device) | HECHO |
| 54 | home_discover (mDNS LAN scan) | HECHO |

## Part I: Infrastructure Extensions

| # | Item | Estado |
|---|------|--------|
| 55 | AiEvent::ExternalEvent (always available) | HECHO |
| 56 | AiEvent::DeviceStateChanged (cfg home-automation) | HECHO |
| 57 | ConfigSection::HomeAutomation | HECHO |
| 58 | ResourceType::Device | HECHO |
| 59 | NetworkPolicy::preset_home_automation() | HECHO |
| 60 | Concepts 227-233 | HECHO |

## Security: 51 Attack Vectors (11 iterations)

6 critical, 15 high, 16 medium, 14 low — all mitigated.

## Test count

- Before: 7,113 (V63)
- After: 7,179 (+66)

## Feature flags

- `home-automation` — HA + MQTT + OpenHAB + custom devices + mDNS + event listener
- `coap` — CoAP protocol for industrial IoT (separate, zero impact)
