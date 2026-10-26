//! V103: vLLM OpenAI-compatible server capability probe.
//!
//! vLLM (<https://github.com/vllm-project/vllm>) exposes a set of OpenAI-compatible
//! endpoints plus a small number of vLLM-specific ones that let us detect the
//! engine version, the models currently loaded, and whether LoRA hot-swap is
//! enabled:
//!
//! - `/v1/models` — OpenAI-compatible list of served models.
//! - `/version` — vLLM engine version string (vLLM-specific).
//! - `/health` — simple healthcheck (HTTP 200 when ready).
//! - `/v1/load_lora_adapter` — only present when vLLM was launched with
//!   `--enable-lora`. We don't call it; we just detect whether probing it
//!   returns something other than 404.
//!
//! The probe is intentionally *passive*: it never loads adapters, never
//! triggers inference, never changes server state.

use crate::http_client::{HttpClient, HttpMethod, HttpRequest, UreqClient};
use crate::retry::{retry_with_config, RetryConfig};
use serde::{Deserialize, Serialize};

/// A single model currently served by vLLM (as reported by `/v1/models`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VLlmServedModel {
    /// Model identifier (typically a HuggingFace repo ID, e.g.
    /// `Qwen/Qwen2.5-7B-Instruct`).
    pub id: String,
    /// Who owns/served this model (`vllm` for base models, user-chosen for
    /// LoRA adapters).
    pub owned_by: Option<String>,
    /// Maximum context length the engine was launched with (`--max-model-len`).
    /// vLLM reports this under the non-standard `max_model_len` field when
    /// available.
    pub max_model_len: Option<u32>,
    /// Root model that a LoRA adapter is attached to, when this entry is a
    /// LoRA adapter rather than a base model.
    pub parent: Option<String>,
}

/// What a vLLM probe tells us about the running server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VLlmCapability {
    /// vLLM engine version string reported by `/version`, if reachable.
    pub engine_version: Option<String>,
    /// Models currently loaded and served by the engine.
    pub served_models: Vec<VLlmServedModel>,
    /// Whether the `/health` endpoint returned success.
    pub healthy: bool,
    /// Whether vLLM was launched with LoRA support (`--enable-lora`).
    ///
    /// Detected by whether the `/v1/load_lora_adapter` endpoint responds
    /// with anything other than 404 to a HEAD/OPTIONS probe — we do not
    /// actually issue a load.
    pub supports_lora: bool,
}

impl VLlmCapability {
    /// Is the given model ID currently loaded in the engine?
    pub fn has_model(&self, id: &str) -> bool {
        self.served_models.iter().any(|m| m.id == id)
    }

    /// Return the maximum context length across all served base models.
    /// Useful for UI to cap the slider without querying per-model.
    pub fn max_context_length(&self) -> Option<u32> {
        self.served_models
            .iter()
            .filter_map(|m| m.max_model_len)
            .max()
    }
}

/// Probe a running vLLM server at `base_url` (e.g. `http://localhost:8000`).
///
/// Tries `/version`, `/v1/models`, `/health`, and `/v1/load_lora_adapter` in
/// sequence. Each probe is independent: a failed `/version` does not block
/// reading the model list, and a healthy server without LoRA support still
/// yields a valid capability. Returns `Err(String)` only if `/v1/models`
/// itself cannot be reached (the authoritative "is this a vLLM?" signal).
pub fn probe_vllm(base_url: &str) -> Result<VLlmCapability, String> {
    probe_vllm_with(&UreqClient, base_url)
}

/// Igual que [`probe_vllm`], con el cliente HTTP puesto desde fuera.
///
/// El sufijo `_with` es la convencion que ya usa `fetch_ollama_models_with` y
/// compania: la version corta se queda como la puerta de entrada comoda y esta
/// hace el trabajo.
///
/// Existe porque las cuatro reglas de este sondeo --- que un 404 en
/// `/v1/load_lora_adapter` significa «LoRA no se habilito» y cualquier otro
/// estado significa «la ruta existe», que la ausencia de `/version` no es un
/// fallo, que `/health` manda sobre `healthy`, y que `/v1/models` si es
/// autoritativo --- **no se podian probar sin un vLLM en marcha**. Un puerto que
/// solo se puede usar con su unica implementacion concreta es un puerto de
/// forma y no de funcion (N166).
pub fn probe_vllm_with(client: &dyn HttpClient, base_url: &str) -> Result<VLlmCapability, String> {
    let base = base_url.trim_end_matches('/');

    // `/v1/models` is the authoritative check — if this fails, the server is
    // either down or not OpenAI-compatible.
    let models_url = format!("{}/v1/models", base);
    let models_body: serde_json::Value = retry_with_config(RetryConfig::fast(), || {
        let resp = client
            .send(&HttpRequest::get(&models_url).timeout(std::time::Duration::from_secs(5)))?;
        // `error_for_status` conserva el comportamiento de `.call()`, que ante
        // un no-2xx daba `Err` y hacia reintentar --- pero ahora el mensaje
        // lleva el cuerpo, que es donde el servidor dice por que.
        resp.error_for_status()?;
        let v: serde_json::Value = resp.json()?;
        Ok(v)
    })
    .map_err(|e| format!("GET {}: {}", models_url, e))?;
    let served_models = parse_models_response(&models_body);

    // `/version` is vLLM-specific; absence just means we don't know the
    // version. Don't retry hard — a single try is enough.
    let engine_version = {
        let url = format!("{}/version", base);
        client
            .send(&HttpRequest::get(&url).timeout(std::time::Duration::from_secs(3)))
            .ok()
            .filter(|resp| resp.is_success())
            .and_then(|resp| resp.json().ok())
            .and_then(|v| parse_version_response(&v))
    };

    // `/health` is OpenAI-compatible-ish — 200 means ready.
    let healthy = {
        let url = format!("{}/health", base);
        client
            .send(&HttpRequest::get(&url).timeout(std::time::Duration::from_secs(3)))
            .is_ok_and(|resp| resp.is_success())
    };

    // LoRA detection: HEAD `/v1/load_lora_adapter`. A 404 means LoRA wasn't
    // enabled at launch; anything else (405, 400, 200) means the route
    // exists. We use OPTIONS because HEAD isn't guaranteed to work on POST
    // endpoints.
    let supports_lora = {
        let url = format!("{}/v1/load_lora_adapter", base);
        // El puerto separa las dos cosas que `.call()` mezclaba: un 404 y un
        // fallo de red llegaban los dos como `Err`, y aqui habia que
        // distinguirlos por la variante. Ahora el estado es un estado y el
        // error es un error.
        let probe =
            HttpRequest::new(HttpMethod::Options, &url).timeout(std::time::Duration::from_secs(3));
        match client.send(&probe) {
            // 404 = LoRA no se habilito al arrancar. Cualquier otro estado
            // (405, 400, 200) significa que la ruta existe.
            Ok(resp) => resp.status != 404,
            // Fallo de red: no se sabe, se asume que no esta.
            Err(_) => false,
        }
    };

    Ok(VLlmCapability {
        engine_version,
        served_models,
        healthy,
        supports_lora,
    })
}

/// Parse a `/v1/models` response body into a list of served models.
///
/// OpenAI's contract is `{ "object": "list", "data": [{ "id": "...", ... }] }`.
/// vLLM augments each entry with `max_model_len` and, for LoRA adapters, a
/// `parent` field pointing at the base model.
pub fn parse_models_response(body: &serde_json::Value) -> Vec<VLlmServedModel> {
    body.get("data")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let id = m.get("id").and_then(|v| v.as_str()).map(String::from)?;
                    Some(VLlmServedModel {
                        id,
                        owned_by: m.get("owned_by").and_then(|v| v.as_str()).map(String::from),
                        max_model_len: m
                            .get("max_model_len")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32),
                        parent: m.get("parent").and_then(|v| v.as_str()).map(String::from),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Block until a vLLM server at `base_url` is ready or `timeout` elapses.
///
/// Polls `/health` every `interval` with a short per-request timeout. Returns
/// the capability probe on first success, or `Err` with an elapsed-timeout
/// message. Never hits the network after success.
///
/// Typical use after `vllm serve …`: give vLLM 30–120 s to load weights into
/// VRAM before the CLI returns control. No-op when vLLM is already up.
pub fn vllm_wait_until_ready(
    base_url: &str,
    timeout: std::time::Duration,
    interval: std::time::Duration,
) -> Result<VLlmCapability, String> {
    vllm_wait_until_ready_with(&UreqClient, base_url, timeout, interval)
}

/// Igual que [`vllm_wait_until_ready`], con el cliente HTTP puesto desde fuera.
///
/// Lo que esto hace probable es el camino de EXITO: que un `/health` que
/// contesta mal y luego bien devuelve la capacidad, en vez de agotar el plazo.
/// Antes el unico test posible era el del plazo agotado contra un puerto
/// cerrado --- o sea, la mitad de la funcion.
pub fn vllm_wait_until_ready_with(
    client: &dyn HttpClient,
    base_url: &str,
    timeout: std::time::Duration,
    interval: std::time::Duration,
) -> Result<VLlmCapability, String> {
    let base = base_url.trim_end_matches('/');
    let health_url = format!("{}/health", base);
    let deadline = std::time::Instant::now() + timeout;

    loop {
        let healthy = client
            .send(&HttpRequest::get(&health_url).timeout(std::time::Duration::from_secs(2)))
            .is_ok_and(|resp| resp.is_success());
        if healthy {
            return probe_vllm_with(client, base);
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "vLLM at {} not ready after {:.1}s",
                base_url,
                timeout.as_secs_f32()
            ));
        }
        std::thread::sleep(interval);
    }
}

/// Parse a `/version` response into a version string.
///
/// vLLM returns `{ "version": "0.6.3" }`. Some older builds return the raw
/// string or nothing at all. Handles both.
pub fn parse_version_response(body: &serde_json::Value) -> Option<String> {
    if let Some(s) = body.as_str() {
        return Some(s.to_string());
    }
    body.get("version")
        .and_then(|v| v.as_str())
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_minimal_models_list() {
        let body = json!({
            "object": "list",
            "data": [
                { "id": "Qwen/Qwen2.5-7B-Instruct", "owned_by": "vllm" }
            ]
        });
        let models = parse_models_response(&body);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(models[0].owned_by.as_deref(), Some("vllm"));
        assert!(models[0].max_model_len.is_none());
        assert!(models[0].parent.is_none());
    }

    #[test]
    fn parses_vllm_specific_max_model_len() {
        let body = json!({
            "data": [
                {
                    "id": "meta-llama/Llama-3.1-8B-Instruct",
                    "owned_by": "vllm",
                    "max_model_len": 8192
                }
            ]
        });
        let models = parse_models_response(&body);
        assert_eq!(models[0].max_model_len, Some(8192));
    }

    #[test]
    fn parses_lora_adapter_with_parent() {
        let body = json!({
            "data": [
                { "id": "base-model", "owned_by": "vllm", "max_model_len": 4096 },
                {
                    "id": "my-lora",
                    "owned_by": "user",
                    "parent": "base-model"
                }
            ]
        });
        let models = parse_models_response(&body);
        assert_eq!(models.len(), 2);
        assert_eq!(models[1].parent.as_deref(), Some("base-model"));
    }

    #[test]
    fn parses_empty_models_list() {
        let body = json!({ "object": "list", "data": [] });
        assert!(parse_models_response(&body).is_empty());
    }

    #[test]
    fn parses_missing_data_field_as_empty() {
        let body = json!({ "object": "list" });
        assert!(parse_models_response(&body).is_empty());
    }

    #[test]
    fn skips_entries_without_id() {
        let body = json!({ "data": [ { "owned_by": "vllm" }, { "id": "ok" } ] });
        let models = parse_models_response(&body);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "ok");
    }

    #[test]
    fn parses_version_object() {
        let body = json!({ "version": "0.6.3" });
        assert_eq!(parse_version_response(&body).as_deref(), Some("0.6.3"));
    }

    #[test]
    fn parses_version_as_bare_string() {
        let body = json!("0.5.0");
        assert_eq!(parse_version_response(&body).as_deref(), Some("0.5.0"));
    }

    #[test]
    fn returns_none_when_no_version() {
        let body = json!({ "other": "field" });
        assert!(parse_version_response(&body).is_none());
    }

    #[test]
    fn capability_has_model_lookup() {
        let cap = VLlmCapability {
            engine_version: Some("0.6.3".into()),
            served_models: vec![VLlmServedModel {
                id: "Qwen/Qwen2.5-7B-Instruct".into(),
                owned_by: Some("vllm".into()),
                max_model_len: Some(32768),
                parent: None,
            }],
            healthy: true,
            supports_lora: false,
        };
        assert!(cap.has_model("Qwen/Qwen2.5-7B-Instruct"));
        assert!(!cap.has_model("not-loaded"));
    }

    #[test]
    fn capability_max_context_length_picks_largest() {
        let cap = VLlmCapability {
            engine_version: None,
            served_models: vec![
                VLlmServedModel {
                    id: "a".into(),
                    owned_by: None,
                    max_model_len: Some(4096),
                    parent: None,
                },
                VLlmServedModel {
                    id: "b".into(),
                    owned_by: None,
                    max_model_len: Some(32768),
                    parent: None,
                },
                VLlmServedModel {
                    id: "c".into(),
                    owned_by: None,
                    max_model_len: None,
                    parent: None,
                },
            ],
            healthy: true,
            supports_lora: true,
        };
        assert_eq!(cap.max_context_length(), Some(32768));
    }

    #[test]
    fn wait_until_ready_times_out_fast_when_unreachable() {
        let start = std::time::Instant::now();
        let res = vllm_wait_until_ready(
            "http://127.0.0.1:1", // unused port — connection will fail fast
            std::time::Duration::from_millis(100),
            std::time::Duration::from_millis(50),
        );
        assert!(res.is_err());
        assert!(start.elapsed() < std::time::Duration::from_secs(5));
    }

    #[test]
    fn capability_max_context_none_when_no_data() {
        let cap = VLlmCapability {
            engine_version: None,
            served_models: vec![],
            healthy: false,
            supports_lora: false,
        };
        assert_eq!(cap.max_context_length(), None);
    }
    // ========================================================================
    // Inyeccion del cliente HTTP (N166)
    // ========================================================================

    /// Un cliente que contesta POR RUTA.
    ///
    /// Hace falta asi y no con una cola de respuestas porque `probe_vllm` toca
    /// cuatro rutas y lo que se prueba es que cada una manda sobre su campo:
    /// una cola acoplaria el test al ORDEN de las llamadas, que no es parte del
    /// contrato.
    struct ByPath {
        routes: Vec<(&'static str, u16, serde_json::Value)>,
    }

    impl crate::http_client::HttpClient for ByPath {
        fn send(
            &self,
            request: &crate::http_client::HttpRequest,
        ) -> anyhow::Result<crate::http_client::HttpResponse> {
            for (suffix, status, body) in &self.routes {
                if request.url.ends_with(suffix) {
                    return Ok(crate::http_client::HttpResponse {
                        status: *status,
                        headers: Vec::new(),
                        body: serde_json::to_vec(body)?,
                    });
                }
            }
            anyhow::bail!("no route configured for {}", request.url)
        }

        fn send_streaming(
            &self,
            _request: &crate::http_client::HttpRequest,
        ) -> anyhow::Result<crate::http_client::HttpStream> {
            anyhow::bail!("not used by the probe")
        }
    }

    fn models_ok() -> serde_json::Value {
        serde_json::json!({
            "object": "list",
            "data": [{"id": "Qwen/Qwen2.5-7B-Instruct", "max_model_len": 32768}]
        })
    }

    /// La regla que no se podia probar sin un vLLM en marcha: en
    /// `/v1/load_lora_adapter` un **404 significa que LoRA no se habilito**, y
    /// cualquier otro estado significa que la ruta existe.
    #[test]
    fn test_a_404_on_the_lora_route_means_lora_is_off() {
        let client = ByPath {
            routes: vec![
                ("/v1/models", 200, models_ok()),
                ("/version", 200, serde_json::json!({"version": "0.6.3"})),
                ("/health", 200, serde_json::json!({})),
                ("/v1/load_lora_adapter", 404, serde_json::json!({})),
            ],
        };
        let cap = probe_vllm_with(&client, "http://vllm.test").expect("el sondeo sale bien");
        assert!(!cap.supports_lora, "un 404 es LoRA apagado");
        assert!(cap.healthy);
        assert_eq!(cap.engine_version.as_deref(), Some("0.6.3"));
        assert!(cap.has_model("Qwen/Qwen2.5-7B-Instruct"));
    }

    /// El caso separador: con 405 --- metodo no permitido --- la ruta EXISTE, asi
    /// que LoRA esta. Sin este test, «siempre false» pasaria el de arriba.
    #[test]
    fn test_a_405_on_the_lora_route_means_the_route_exists() {
        let client = ByPath {
            routes: vec![
                ("/v1/models", 200, models_ok()),
                ("/version", 200, serde_json::json!({"version": "0.6.3"})),
                ("/health", 200, serde_json::json!({})),
                ("/v1/load_lora_adapter", 405, serde_json::json!({})),
            ],
        };
        let cap = probe_vllm_with(&client, "http://vllm.test").expect("el sondeo sale bien");
        assert!(
            cap.supports_lora,
            "405 es la ruta contestando que no es ese metodo"
        );
    }

    /// `/version` es propio de vLLM: que no este solo significa que no sabemos
    /// la version, **no** que el sondeo falle.
    #[test]
    fn test_a_missing_version_route_is_not_a_failure() {
        let client = ByPath {
            routes: vec![
                ("/v1/models", 200, models_ok()),
                ("/version", 404, serde_json::json!({})),
                ("/health", 200, serde_json::json!({})),
                ("/v1/load_lora_adapter", 404, serde_json::json!({})),
            ],
        };
        let cap = probe_vllm_with(&client, "http://vllm.test")
            .expect("sin /version el sondeo sigue valiendo");
        assert_eq!(cap.engine_version, None);
        assert!(cap.healthy, "la salud la decide /health, no /version");
    }

    /// Y `/health` manda sobre `healthy`, independientemente del resto.
    #[test]
    fn test_an_unhealthy_health_route_is_reported_without_failing_the_probe() {
        let client = ByPath {
            routes: vec![
                ("/v1/models", 200, models_ok()),
                ("/version", 200, serde_json::json!({"version": "0.6.3"})),
                ("/health", 503, serde_json::json!({})),
                ("/v1/load_lora_adapter", 404, serde_json::json!({})),
            ],
        };
        let cap = probe_vllm_with(&client, "http://vllm.test").expect("el sondeo sale bien");
        assert!(!cap.healthy, "un 503 en /health no es estar sano");
        // Pero el resto se sigue leyendo: un servidor cargando pesos contesta
        // /v1/models y todavia no esta listo.
        assert!(cap.has_model("Qwen/Qwen2.5-7B-Instruct"));
    }

    /// `/v1/models` SI es autoritativo: si falla, no hay capacidad que devolver.
    #[test]
    fn test_a_failing_models_route_fails_the_whole_probe() {
        let client = ByPath {
            routes: vec![("/v1/models", 500, serde_json::json!({"error": "boom"}))],
        };
        let err = probe_vllm_with(&client, "http://vllm.test")
            .expect_err("sin /v1/models no se sabe nada del servidor");
        assert!(err.contains("/v1/models"), "err = {}", err);
        assert!(err.contains("500"), "err = {}", err);
    }
    /// El camino de EXITO de la espera, que era inalcanzable: `/health`
    /// contesta mal las dos primeras veces y bien la tercera.
    ///
    /// Lo unico que se podia probar antes era el plazo agotado contra un puerto
    /// cerrado, o sea la mitad de la funcion --- y justo la mitad que no es la
    /// razon de existir de la otra.
    #[test]
    fn test_the_wait_returns_as_soon_as_health_turns_green() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct HealthyOnThirdTry {
            calls: AtomicUsize,
        }

        impl crate::http_client::HttpClient for HealthyOnThirdTry {
            fn send(
                &self,
                request: &crate::http_client::HttpRequest,
            ) -> anyhow::Result<crate::http_client::HttpResponse> {
                if request.url.ends_with("/health") {
                    let n = self.calls.fetch_add(1, Ordering::Relaxed);
                    // Cargando pesos: contesta, pero con 503.
                    let status = if n < 2 { 503 } else { 200 };
                    return Ok(crate::http_client::HttpResponse {
                        status,
                        headers: Vec::new(),
                        body: Vec::new(),
                    });
                }
                let body = if request.url.ends_with("/v1/models") {
                    models_ok()
                } else {
                    serde_json::json!({})
                };
                // `/version` y la ruta de LoRA contestan 404: ni version ni LoRA.
                let status = if request.url.ends_with("/v1/models") {
                    200
                } else {
                    404
                };
                Ok(crate::http_client::HttpResponse {
                    status,
                    headers: Vec::new(),
                    body: serde_json::to_vec(&body)?,
                })
            }

            fn send_streaming(
                &self,
                _request: &crate::http_client::HttpRequest,
            ) -> anyhow::Result<crate::http_client::HttpStream> {
                anyhow::bail!("not used by the wait")
            }
        }

        let client = HealthyOnThirdTry {
            calls: AtomicUsize::new(0),
        };
        let cap = vllm_wait_until_ready_with(
            &client,
            "http://vllm.test",
            std::time::Duration::from_secs(5),
            std::time::Duration::from_millis(10),
        )
        .expect("en la tercera vuelta /health contesta 200");
        assert!(cap.has_model("Qwen/Qwen2.5-7B-Instruct"));
        assert!(cap.healthy);
        // CUATRO llamadas a `/health`, y el numero dice algo: tres del bucle de
        // espera --- si saliera a la primera, el 503 no contaria como «todavia
        // no» y la espera no serviria de nada --- mas una cuarta del sondeo, que
        // vuelve a preguntar para rellenar su propio campo `healthy`.
        //
        // Esa cuarta es redundante: la espera acaba de comprobarlo. Es una
        // peticion de mas, no un fallo, y se deja escrita aqui en vez de
        // ajustar el numero sin mirar --- que es como un test deja de describir
        // lo que el codigo hace.
        assert_eq!(client.calls.load(Ordering::Relaxed), 4);
    }
}
