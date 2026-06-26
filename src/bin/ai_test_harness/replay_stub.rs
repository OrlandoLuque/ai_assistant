use super::*;

pub struct ReplayConfig {
    pub session_file: String,
    pub provider: Option<String>,
    pub url: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub compare: bool,
    pub session_index: Option<usize>,
}

pub fn run_replay(_config: ReplayConfig) -> Result<(), String> {
    Err("Replay mode requires the 'rag' feature to be enabled".to_string())
}
