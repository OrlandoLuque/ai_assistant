//! V128 (C.7) — `ai_backup` CLI.
//!
//! Subcommands:
//!   create  --source <path> [--source <path>]... --output <base> [--passphrase-env VAR] [--sign-key <path>] [--label <text>]
//!   verify  --input <archive> [--passphrase-env VAR] [--verify-key <path>]
//!   restore --input <archive> --output <dir> [--passphrase-env VAR] [--verify-key <path>]
//!
//! Passphrase is read from an environment variable (never the command
//! line) to avoid leaking it to shell history. The signing/verifying
//! keys are 32-byte raw Ed25519 secret/public keys (matches what
//! ed25519-dalek emits via SigningKey::to_bytes / VerifyingKey::to_bytes).

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ai_assistant::secure_backup::{
    create_backup, derive_key, restore_backup, salt_from_envelope, verify_backup, BackupConfig,
    BackupReport, EncryptionMaterial,
};
use ed25519_dalek::{SigningKey, VerifyingKey};

const USAGE: &str = "\
ai_backup — encrypted, signed archive of one or more source paths.

USAGE:
    ai_backup create  --source <path> [--source <path>]... --output <base> [--passphrase-env VAR] [--sign-key <path>] [--label <text>]
    ai_backup verify  --input <archive> [--passphrase-env VAR] [--verify-key <path>]
    ai_backup restore --input <archive> --output <dir> [--passphrase-env VAR] [--verify-key <path>]
    ai_backup --help

Notes:
  * --passphrase-env names an environment variable (e.g. AI_BACKUP_PASS).
    The CLI never reads passphrases from argv.
  * --sign-key / --verify-key are 32-byte raw Ed25519 keys on disk.
  * Without --passphrase-env the archive is plain (still SHA-256-summed).
  * On `create`, the output suffix is appended automatically:
        <base>.zip      (plain) or <base>.zip.enc (encrypted)
        <base>.zip.sha256       (always)
        <base>.zip.sig          (when --sign-key supplied)
";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    match args[1].as_str() {
        "-h" | "--help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        "create" => run(cmd_create(&args[2..])),
        "verify" => run(cmd_verify(&args[2..])),
        "restore" => run(cmd_restore(&args[2..])),
        other => {
            eprintln!("ai_backup: unknown subcommand '{other}'");
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run<F>(f: F) -> ExitCode
where
    F: FnOnce() -> Result<(), String>,
{
    match f() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("ai_backup: {msg}");
            ExitCode::from(1)
        }
    }
}

#[derive(Default)]
struct Flags {
    sources: Vec<PathBuf>,
    output: Option<PathBuf>,
    input: Option<PathBuf>,
    passphrase_env: Option<String>,
    sign_key: Option<PathBuf>,
    verify_key: Option<PathBuf>,
    label: String,
}

fn parse_flags(args: &[String]) -> Result<Flags, String> {
    let mut f = Flags::default();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let v = || -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("flag '{a}' needs a value"))
        };
        match a.as_str() {
            "--source" => {
                f.sources.push(PathBuf::from(v()?));
                i += 2;
            }
            "--output" => {
                f.output = Some(PathBuf::from(v()?));
                i += 2;
            }
            "--input" => {
                f.input = Some(PathBuf::from(v()?));
                i += 2;
            }
            "--passphrase-env" => {
                f.passphrase_env = Some(v()?);
                i += 2;
            }
            "--sign-key" => {
                f.sign_key = Some(PathBuf::from(v()?));
                i += 2;
            }
            "--verify-key" => {
                f.verify_key = Some(PathBuf::from(v()?));
                i += 2;
            }
            "--label" => {
                f.label = v()?;
                i += 2;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    Ok(f)
}

fn read_passphrase(env_name: &str) -> Result<String, String> {
    env::var(env_name).map_err(|_| format!("env var '{env_name}' not set"))
}

fn read_sign_key(path: &Path) -> Result<SigningKey, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("read sign-key {}: {e}", path.display()))?;
    if bytes.len() != 32 {
        return Err(format!(
            "sign-key must be 32 raw bytes, got {} bytes",
            bytes.len()
        ));
    }
    let mut k = [0u8; 32];
    k.copy_from_slice(&bytes);
    Ok(SigningKey::from_bytes(&k))
}

fn read_verify_key(path: &Path) -> Result<VerifyingKey, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("read verify-key {}: {e}", path.display()))?;
    if bytes.len() != 32 {
        return Err(format!(
            "verify-key must be 32 raw bytes, got {} bytes",
            bytes.len()
        ));
    }
    let mut k = [0u8; 32];
    k.copy_from_slice(&bytes);
    VerifyingKey::from_bytes(&k).map_err(|e| format!("verify-key invalid: {e}"))
}

fn derive_key_for_open(archive_path: &Path, passphrase: &str) -> Result<[u8; 32], String> {
    let envelope = std::fs::read(archive_path)
        .map_err(|e| format!("read archive {}: {e}", archive_path.display()))?;
    let salt = salt_from_envelope(&envelope).map_err(|e| format!("read salt: {e}"))?;
    Ok(derive_key(passphrase, &salt))
}

fn cmd_create(args: &[String]) -> impl FnOnce() -> Result<(), String> {
    let args = args.to_vec();
    move || {
        let f = parse_flags(&args)?;
        if f.sources.is_empty() {
            return Err("at least one --source is required".into());
        }
        let output = f.output.ok_or("--output is required")?;

        let encryption = if let Some(env_name) = &f.passphrase_env {
            let pass = read_passphrase(env_name)?;
            Some(EncryptionMaterial::Passphrase(pass))
        } else {
            None
        };

        let signing_key = if let Some(p) = &f.sign_key {
            Some(read_sign_key(p)?)
        } else {
            None
        };

        let cfg = BackupConfig {
            sources: f.sources,
            output,
            encryption,
            signing_key: signing_key.as_ref(),
            source_label: if f.label.is_empty() {
                "ai_assistant backup".into()
            } else {
                f.label
            },
        };
        let report = create_backup(&cfg).map_err(|e| format!("create_backup: {e}"))?;
        print_report(&report);
        Ok(())
    }
}

fn cmd_verify(args: &[String]) -> impl FnOnce() -> Result<(), String> {
    let args = args.to_vec();
    move || {
        let f = parse_flags(&args)?;
        let input = f.input.ok_or("--input is required")?;

        let key_bytes = if let Some(env_name) = &f.passphrase_env {
            let pass = read_passphrase(env_name)?;
            Some(derive_key_for_open(&input, &pass)?)
        } else {
            None
        };

        let verify_key = if let Some(p) = &f.verify_key {
            Some(read_verify_key(p)?)
        } else {
            None
        };

        let report = verify_backup(&input, key_bytes.as_ref(), verify_key.as_ref())
            .map_err(|e| format!("verify_backup: {e}"))?;
        print_report(&report);
        println!("VERIFY OK");
        Ok(())
    }
}

fn cmd_restore(args: &[String]) -> impl FnOnce() -> Result<(), String> {
    let args = args.to_vec();
    move || {
        let f = parse_flags(&args)?;
        let input = f.input.ok_or("--input is required")?;
        let output = f.output.ok_or("--output (target dir) is required")?;

        let key_bytes = if let Some(env_name) = &f.passphrase_env {
            let pass = read_passphrase(env_name)?;
            Some(derive_key_for_open(&input, &pass)?)
        } else {
            None
        };

        let verify_key = if let Some(p) = &f.verify_key {
            Some(read_verify_key(p)?)
        } else {
            None
        };

        let report = restore_backup(&input, &output, key_bytes.as_ref(), verify_key.as_ref())
            .map_err(|e| format!("restore_backup: {e}"))?;
        print_report(&report);
        println!("RESTORE OK -> {}", output.display());
        Ok(())
    }
}

fn print_report(r: &BackupReport) {
    println!("archive : {}", r.archive_path.display());
    println!("encrypted: {}", r.encrypted);
    println!("signed   : {}", r.signed);
    println!("label    : {}", r.manifest.source_label);
    println!("created  : {}", r.manifest.created_at);
    println!("entries  : {}", r.manifest.entries.len());
    println!("bytes    : {}", r.manifest.total_bytes);
}
