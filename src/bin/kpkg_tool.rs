//! CLI tool for creating and inspecting encrypted knowledge packages (.kpkg)
//!
//! Usage:
//!   kpkg_tool create --input <folder> --output <file.kpkg> [--name "Name"]
//!   kpkg_tool list <file.kpkg>
//!   kpkg_tool extract --input <file.kpkg> --output <folder>

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ai_assistant::{
    AppKeyProvider, CustomKeyProvider, KpkgBuilder, KpkgReader, KeyProvider,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return ExitCode::FAILURE;
    }

    match args[1].as_str() {
        "create" => match run_create(&args[2..]) {
            Ok(_) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("Error: {}", e);
                ExitCode::FAILURE
            }
        },
        "list" => match run_list(&args[2..]) {
            Ok(_) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("Error: {}", e);
                ExitCode::FAILURE
            }
        },
        "extract" => match run_extract(&args[2..]) {
            Ok(_) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("Error: {}", e);
                ExitCode::FAILURE
            }
        },
        "--help" | "-h" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        cmd => {
            eprintln!("Unknown command: {}", cmd);
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!(
        r#"kpkg_tool - Encrypted Knowledge Package Tool

USAGE:
    kpkg_tool <COMMAND> [OPTIONS]

COMMANDS:
    create      Create a new .kpkg file from a folder
    list        List contents of a .kpkg file
    extract     Extract contents of a .kpkg file (for debugging)
    help        Show this help message

CREATE OPTIONS:
    --input, -i <folder>     Source folder containing .md and .txt files
    --output, -o <file>      Output .kpkg file
    --name, -n <name>        Package name (optional)
    --description, -d <desc> Package description (optional)
    --passphrase, -p <pass>  Custom passphrase (default: app key)

LIST OPTIONS:
    <file.kpkg>              The package to list
    --passphrase, -p <pass>  Custom passphrase if not using app key

EXTRACT OPTIONS:
    --input, -i <file>       The .kpkg file to extract
    --output, -o <folder>    Output folder
    --passphrase, -p <pass>  Custom passphrase if not using app key

EXAMPLES:
    kpkg_tool create -i ./knowledge -o knowledge.kpkg -n "My Knowledge Base"
    kpkg_tool list knowledge.kpkg
    kpkg_tool extract -i knowledge.kpkg -o ./extracted
"#
    );
}

fn run_create(args: &[String]) -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut passphrase: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input = Some(PathBuf::from(args.get(i).ok_or("Missing input path")?));
            }
            "--output" | "-o" => {
                i += 1;
                output = Some(PathBuf::from(args.get(i).ok_or("Missing output path")?));
            }
            "--name" | "-n" => {
                i += 1;
                name = Some(args.get(i).ok_or("Missing name")?.clone());
            }
            "--description" | "-d" => {
                i += 1;
                description = Some(args.get(i).ok_or("Missing description")?.clone());
            }
            "--passphrase" | "-p" => {
                i += 1;
                passphrase = Some(args.get(i).ok_or("Missing passphrase")?.clone());
            }
            arg => {
                return Err(format!("Unknown argument: {}", arg));
            }
        }
        i += 1;
    }

    let input = input.ok_or("Missing required --input")?;
    let output = output.ok_or("Missing required --output")?;

    if !input.exists() {
        return Err(format!("Input folder does not exist: {}", input.display()));
    }

    if !input.is_dir() {
        return Err(format!("Input is not a directory: {}", input.display()));
    }

    // Collect all documents from the folder
    let docs = collect_documents(&input)?;

    if docs.is_empty() {
        return Err("No .md or .txt files found in input folder".into());
    }

    println!("Found {} documents", docs.len());

    // Build the package
    let encrypted = if let Some(pass) = passphrase {
        build_package_with_key(
            CustomKeyProvider::new(pass),
            &docs,
            name,
            description,
        )?
    } else {
        build_package_with_key(AppKeyProvider, &docs, name, description)?
    };

    // Write to output
    fs::write(&output, &encrypted)
        .map_err(|e| format!("Failed to write output file: {}", e))?;

    println!("Created package: {} ({} bytes)", output.display(), encrypted.len());

    Ok(())
}

fn build_package_with_key<K: KeyProvider>(
    key_provider: K,
    docs: &[(String, String)],
    name: Option<String>,
    description: Option<String>,
) -> Result<Vec<u8>, String> {
    let mut builder = KpkgBuilder::with_key_provider(key_provider);

    if let Some(n) = name {
        builder = builder.name(n);
    }

    if let Some(d) = description {
        builder = builder.description(d);
    }

    for (path, content) in docs {
        builder = builder.add_document(path.clone(), content.clone(), None);
    }

    builder.build().map_err(|e| format!("Failed to build package: {}", e))
}

fn collect_documents(folder: &Path) -> Result<Vec<(String, String)>, String> {
    let mut docs = Vec::new();

    collect_documents_recursive(folder, folder, &mut docs)?;

    Ok(docs)
}

fn collect_documents_recursive(
    base: &Path,
    current: &Path,
    docs: &mut Vec<(String, String)>,
) -> Result<(), String> {
    let entries = fs::read_dir(current)
        .map_err(|e| format!("Failed to read directory {}: {}", current.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let path = entry.path();

        if path.is_dir() {
            // Skip hidden directories
            if path
                .file_name()
                .map(|n| n.to_string_lossy().starts_with('.'))
                .unwrap_or(false)
            {
                continue;
            }
            collect_documents_recursive(base, &path, docs)?;
        } else if path.is_file() {
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();

            if ext == "md" || ext == "txt" {
                let content = fs::read_to_string(&path)
                    .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

                // Get relative path from base
                let rel_path = path
                    .strip_prefix(base)
                    .map_err(|_| "Failed to get relative path")?
                    .to_string_lossy()
                    .replace('\\', "/"); // Normalize path separators

                docs.push((rel_path, content));
            }
        }
    }

    Ok(())
}

fn run_list(args: &[String]) -> Result<(), String> {
    let mut file: Option<PathBuf> = None;
    let mut passphrase: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--passphrase" | "-p" => {
                i += 1;
                passphrase = Some(args.get(i).ok_or("Missing passphrase")?.clone());
            }
            arg if !arg.starts_with('-') && file.is_none() => {
                file = Some(PathBuf::from(arg));
            }
            arg => {
                return Err(format!("Unknown argument: {}", arg));
            }
        }
        i += 1;
    }

    let file = file.ok_or("Missing .kpkg file path")?;

    if !file.exists() {
        return Err(format!("File does not exist: {}", file.display()));
    }

    let data = fs::read(&file).map_err(|e| format!("Failed to read file: {}", e))?;

    let docs = if let Some(pass) = passphrase {
        let reader = KpkgReader::with_key_provider(CustomKeyProvider::new(pass));
        reader.read(&data).map_err(|e| format!("Failed to read package: {}", e))?
    } else {
        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        reader.read(&data).map_err(|e| format!("Failed to read package: {}", e))?
    };

    println!("Package: {} ({} bytes encrypted)", file.display(), data.len());
    println!("Documents: {}", docs.len());
    println!();

    let mut total_size = 0usize;

    for doc in &docs {
        let size = doc.content.len();
        total_size += size;
        println!(
            "  {:40} {:>8} bytes  (priority: {})",
            doc.path, size, doc.priority
        );
    }

    println!();
    println!("Total content size: {} bytes", total_size);

    Ok(())
}

fn run_extract(args: &[String]) -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut passphrase: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                i += 1;
                input = Some(PathBuf::from(args.get(i).ok_or("Missing input path")?));
            }
            "--output" | "-o" => {
                i += 1;
                output = Some(PathBuf::from(args.get(i).ok_or("Missing output path")?));
            }
            "--passphrase" | "-p" => {
                i += 1;
                passphrase = Some(args.get(i).ok_or("Missing passphrase")?.clone());
            }
            arg => {
                return Err(format!("Unknown argument: {}", arg));
            }
        }
        i += 1;
    }

    let input = input.ok_or("Missing required --input")?;
    let output = output.ok_or("Missing required --output")?;

    if !input.exists() {
        return Err(format!("Input file does not exist: {}", input.display()));
    }

    let data = fs::read(&input).map_err(|e| format!("Failed to read file: {}", e))?;

    let docs = if let Some(pass) = passphrase {
        let reader = KpkgReader::with_key_provider(CustomKeyProvider::new(pass));
        reader.read(&data).map_err(|e| format!("Failed to read package: {}", e))?
    } else {
        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        reader.read(&data).map_err(|e| format!("Failed to read package: {}", e))?
    };

    // Create output directory
    fs::create_dir_all(&output)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    println!("Extracting {} documents to {}", docs.len(), output.display());

    for doc in &docs {
        let doc_path = output.join(&doc.path);

        // Create parent directories
        if let Some(parent) = doc_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        fs::write(&doc_path, &doc.content)
            .map_err(|e| format!("Failed to write {}: {}", doc_path.display(), e))?;

        println!("  Extracted: {}", doc.path);
    }

    println!("Done!");

    Ok(())
}
