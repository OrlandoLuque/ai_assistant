//! MCP tools for speaker enrollment, identification, and voice cloning.
//!
//! Provides tools:
//! - voice_enroll_speaker — enroll a speaker into the SpeakerGate
//! - voice_identify_speaker — identify a speaker from an audio sample
//! - voice_list_speakers — list enrolled speaker profiles
//! - voice_remove_speaker — remove a speaker profile
//! - voice_gate_config — configure SpeakerGate settings
//! - voice_clone_create — create a cloned voice (requires `audio` feature)
//! - voice_clone_synthesize — synthesize speech with a cloned voice
//! - voice_clone_list — list cloned voice profiles

use crate::audio_filter::SpeakerGate;
use crate::mcp_protocol::server::McpServer;
use crate::mcp_protocol::types::{McpTool, McpToolAnnotation};
use std::sync::{Arc, Mutex};

// ============================================================================
// Registration
// ============================================================================

/// Register voice speaker enrollment and identification tools.
pub fn register_voice_tools(server: &mut McpServer, speaker_gate: Arc<Mutex<SpeakerGate>>) {
    let ann_ro = McpToolAnnotation {
        title: None,
        read_only_hint: Some(true),
        destructive_hint: Some(false),
        idempotent_hint: Some(true),
        open_world_hint: Some(false),
    };
    let ann_action = McpToolAnnotation {
        title: None,
        read_only_hint: Some(false),
        destructive_hint: Some(false),
        idempotent_hint: Some(false),
        open_world_hint: Some(false),
    };
    let ann_delete = McpToolAnnotation {
        title: None,
        read_only_hint: Some(false),
        destructive_hint: Some(true),
        idempotent_hint: Some(true),
        open_world_hint: Some(false),
    };

    // --- voice_enroll_speaker ---
    {
        let gate = speaker_gate.clone();
        server.register_tool(
            McpTool::new(
                "voice_enroll_speaker",
                "Enroll a speaker by providing PCM16 audio bytes (base64-encoded). Requires at least 3 seconds of clear speech.",
            )
            .with_property("audio_base64", "string", "Base64-encoded PCM16 audio at 16kHz", true)
            .with_property("name", "string", "Speaker name", true)
            .with_property("is_owner", "boolean", "Whether this is the device owner", false)
            .with_annotations(ann_action.clone()),
            move |args| {
                let audio_b64 = args.get("audio_base64").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: audio_base64")?;
                let name = args.get("name").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: name")?;
                let is_owner = args.get("is_owner").and_then(|v| v.as_bool()).unwrap_or(false);

                let audio = simple_base64_decode(audio_b64)
                    .map_err(|e| format!("Invalid base64: {}", e))?;

                if audio.len() % 2 != 0 {
                    return Err("Audio data must be even-length PCM16 bytes".to_string());
                }
                if audio.len() > 16000 * 2 * 300 {
                    return Err("Audio too large (max 5 minutes)".to_string());
                }

                let samples: Vec<i16> = audio
                    .chunks(2)
                    .filter(|pair| pair.len() == 2)
                    .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                    .collect();

                let mut gate = gate.lock().map_err(|e| format!("Lock error: {}", e))?;
                if gate.profiles().len() >= 100 {
                    return Err("Speaker limit reached (max 100 profiles)".to_string());
                }
                let speaker_id = gate
                    .enroll(&samples, 16000, name, is_owner)
                    .map_err(|e| format!("Enrollment failed: {}", e))?;

                Ok(serde_json::json!({
                    "status": "enrolled",
                    "speaker_id": speaker_id,
                    "name": name,
                    "is_owner": is_owner,
                }))
            },
        );
    }

    // --- voice_identify_speaker ---
    {
        let gate = speaker_gate.clone();
        server.register_tool(
            McpTool::new(
                "voice_identify_speaker",
                "Identify a speaker from a PCM16 audio sample (base64-encoded).",
            )
            .with_property(
                "audio_base64",
                "string",
                "Base64-encoded PCM16 audio at 16kHz",
                true,
            )
            .with_annotations(ann_ro.clone()),
            move |args| {
                let audio_b64 = args
                    .get("audio_base64")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: audio_base64")?;

                let audio = simple_base64_decode(audio_b64)
                    .map_err(|e| format!("Invalid base64: {}", e))?;

                if audio.len() % 2 != 0 {
                    return Err("Audio data must be even-length PCM16 bytes".to_string());
                }

                let samples: Vec<i16> = audio
                    .chunks(2)
                    .filter(|pair| pair.len() == 2)
                    .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                    .collect();

                let mut gate = gate.lock().map_err(|e| format!("Lock error: {}", e))?;
                let identification = gate.identify(&samples, 16000);

                Ok(serde_json::json!({
                    "identification": format!("{:?}", identification),
                }))
            },
        );
    }

    // --- voice_list_speakers ---
    {
        let gate = speaker_gate.clone();
        server.register_tool(
            McpTool::new("voice_list_speakers", "List all enrolled speaker profiles.")
                .with_annotations(ann_ro.clone()),
            move |_args| {
                let gate = gate.lock().map_err(|e| format!("Lock error: {}", e))?;
                let profiles = gate.profiles();
                let list: Vec<serde_json::Value> = profiles
                    .iter()
                    .map(|p| {
                        serde_json::json!({
                            "name": p.name,
                            "is_owner": p.is_owner,
                            "embeddings_count": p.embeddings.len(),
                        })
                    })
                    .collect();
                Ok(serde_json::json!({ "speakers": list, "count": list.len() }))
            },
        );
    }

    // --- voice_remove_speaker ---
    {
        let gate = speaker_gate.clone();
        server.register_tool(
            McpTool::new("voice_remove_speaker", "Remove a speaker profile by name.")
                .with_property("name", "string", "Speaker name to remove", true)
                .with_annotations(ann_delete),
            move |args| {
                let name = args
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: name")?;
                let mut gate = gate.lock().map_err(|e| format!("Lock error: {}", e))?;
                let removed = gate.remove_profile(name);
                Ok(serde_json::json!({ "removed": removed, "name": name }))
            },
        );
    }

    // --- voice_gate_config ---
    {
        let gate = speaker_gate;
        server.register_tool(
            McpTool::new(
                "voice_gate_config",
                "Configure the speaker gate: only_owner mode and allow_unknown mode.",
            )
            .with_property(
                "only_owner",
                "boolean",
                "Only allow the owner to pass the gate",
                false,
            )
            .with_property(
                "allow_unknown",
                "boolean",
                "Allow unrecognized speakers",
                false,
            )
            .with_annotations(ann_action),
            move |args| {
                let mut gate = gate.lock().map_err(|e| format!("Lock error: {}", e))?;
                if let Some(val) = args.get("only_owner").and_then(|v| v.as_bool()) {
                    gate.set_only_owner(val);
                }
                if let Some(val) = args.get("allow_unknown").and_then(|v| v.as_bool()) {
                    gate.set_allow_unknown(val);
                }
                Ok(serde_json::json!({ "status": "updated" }))
            },
        );
    }
}

/// Register voice cloning tools (requires `audio` feature for VoiceCloneProvider).
#[cfg(feature = "audio")]
pub fn register_voice_clone_tools(
    server: &mut McpServer,
    clone_provider: Arc<Mutex<dyn crate::speech::VoiceCloneProvider>>,
) {
    let ann_action = McpToolAnnotation {
        title: None,
        read_only_hint: Some(false),
        destructive_hint: Some(false),
        idempotent_hint: Some(false),
        open_world_hint: Some(true),
    };
    let ann_ro = McpToolAnnotation {
        title: None,
        read_only_hint: Some(true),
        destructive_hint: Some(false),
        idempotent_hint: Some(true),
        open_world_hint: Some(true),
    };

    // --- voice_clone_create ---
    {
        let provider = clone_provider.clone();
        server.register_tool(
            McpTool::new(
                "voice_clone_create",
                "Create a cloned voice from audio samples. Base64-encoded PCM16 at 16kHz, min 3 seconds.",
            )
            .with_property("audio_base64", "string", "Base64-encoded PCM16 audio", true)
            .with_property("name", "string", "Name for the cloned voice", true)
            .with_annotations(ann_action.clone()),
            move |args| {
                let audio_b64 = args.get("audio_base64").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: audio_base64")?;
                let name = args.get("name").and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: name")?;

                let audio = simple_base64_decode(audio_b64)
                    .map_err(|e| format!("Invalid base64: {}", e))?;

                let provider = provider.lock().map_err(|e| format!("Lock error: {}", e))?;
                let voice_id = provider
                    .enroll(&audio, crate::speech::AudioFormat::Pcm, name, 16000)
                    .map_err(|e| format!("Voice cloning failed: {}", e))?;

                Ok(serde_json::json!({
                    "status": "created",
                    "voice_id": voice_id,
                    "name": name,
                }))
            },
        );
    }

    // --- voice_clone_synthesize ---
    {
        let provider = clone_provider.clone();
        server.register_tool(
            McpTool::new(
                "voice_clone_synthesize",
                "Synthesize speech using a cloned voice. Returns base64-encoded audio.",
            )
            .with_property("text", "string", "Text to synthesize", true)
            .with_property("voice_id", "string", "Cloned voice ID", true)
            .with_annotations(ann_action),
            move |args| {
                let text = args
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: text")?;
                let voice_id = args
                    .get("voice_id")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing required parameter: voice_id")?;

                let provider = provider.lock().map_err(|e| format!("Lock error: {}", e))?;
                let options = crate::speech::SynthesisOptions::default();
                let result = provider
                    .synthesize_cloned(text, &voice_id.to_string(), &options)
                    .map_err(|e| format!("Synthesis failed: {}", e))?;

                let audio_b64 = simple_base64_encode(&result.audio);
                Ok(serde_json::json!({
                    "audio_base64": audio_b64,
                    "format": format!("{:?}", result.format),
                    "duration_secs": result.duration_secs,
                    "sample_rate": result.sample_rate,
                }))
            },
        );
    }

    // --- voice_clone_list ---
    {
        server.register_tool(
            McpTool::new("voice_clone_list", "List all cloned voice profiles.")
                .with_annotations(ann_ro),
            move |_args| {
                let provider = clone_provider
                    .lock()
                    .map_err(|e| format!("Lock error: {}", e))?;
                let voices = provider
                    .list_cloned_voices()
                    .map_err(|e| format!("List failed: {}", e))?;

                let list: Vec<serde_json::Value> = voices
                    .iter()
                    .map(|v| {
                        serde_json::json!({
                            "voice_id": v.voice_id,
                            "name": v.name,
                            "provider": v.provider,
                            "quality_score": v.quality_score,
                        })
                    })
                    .collect();
                Ok(serde_json::json!({ "voices": list, "count": list.len() }))
            },
        );
    }
}

// ============================================================================
// Base64 helpers
// ============================================================================

fn simple_base64_decode(input: &str) -> Result<Vec<u8>, String> {
    const TABLE: [u8; 256] = {
        let mut t = [255u8; 256];
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            t[chars[i] as usize] = i as u8;
            i += 1;
        }
        t
    };

    let input = input.trim().as_bytes();
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;

    for &b in input {
        if b == b'=' || b == b'\n' || b == b'\r' {
            continue;
        }
        let val = TABLE[b as usize];
        if val == 255 {
            return Err(format!("Invalid base64 character: {}", b as char));
        }
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(output)
}

fn simple_base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((n >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(n & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_roundtrip() {
        let data = b"Hello, Voice Cloning!";
        let encoded = simple_base64_encode(data);
        let decoded = simple_base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64_decode_known() {
        let decoded = simple_base64_decode("SGVsbG8=").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_base64_encode_known() {
        assert_eq!(simple_base64_encode(b"Hello"), "SGVsbG8=");
        assert_eq!(simple_base64_encode(b""), "");
    }

    #[test]
    fn test_base64_decode_invalid() {
        assert!(simple_base64_decode("!!!").is_err());
    }

    #[test]
    fn test_speaker_gate_tools_registration() {
        use crate::audio_filter::MfccSpeakerVerifier;
        let verifier = Box::new(MfccSpeakerVerifier::new());
        let gate = Arc::new(Mutex::new(SpeakerGate::new(verifier, 0.7)));
        let mut server = McpServer::new("test", "0.1.0");
        // Should not panic
        register_voice_tools(&mut server, gate);
    }
}
