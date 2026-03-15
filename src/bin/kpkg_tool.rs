//! CLI tool for creating and inspecting encrypted knowledge packages (.kpkg)
//!
//! Usage:
//!   kpkg_tool create --input <folder> --output <file.kpkg> [--name "Name"]
//!   kpkg_tool list <file.kpkg>
//!   kpkg_tool inspect <file.kpkg>
//!   kpkg_tool extract --input <file.kpkg> --output <folder>

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ai_assistant::{
    AppKeyProvider, CustomKeyProvider, ExamplePair, KeyProvider, KpkgBuilder, KpkgManifest,
    KpkgReader,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return ExitCode::FAILURE;
    }

    // Background update check
    let update_rx = ai_assistant::update_checker::check_for_update_bg(env!("CARGO_PKG_VERSION"));

    let result = match args[1].as_str() {
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
        "inspect" => match run_inspect(&args[2..]) {
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
    };

    // Check for updates before exit
    if let Ok(info) = update_rx.try_recv() {
        eprintln!();
        eprintln!("  Update available: v{} \u{2192} v{}", info.current, info.latest);
        eprintln!("  Download: {}", info.url);
        eprintln!();
    }

    result
}

fn print_usage() {
    println!(
        r#"kpkg_tool - Encrypted Knowledge Package Tool

USAGE:
    kpkg_tool <COMMAND> [OPTIONS]

COMMANDS:
    create      Create a new .kpkg file from a folder
    list        List contents of a .kpkg file
    inspect     Show full manifest details of a .kpkg file
    extract     Extract contents of a .kpkg file (for debugging)
    help        Show this help message

CREATE OPTIONS:
    --input, -i <folder>         Source folder containing .md and .txt files
    --output, -o <file>          Output .kpkg file
    --name, -n <name>            Package name (optional)
    --description, -d <desc>     Package description (optional)
    --passphrase, -p <pass>      Custom passphrase (default: app key)

    Professional Package Options:
    --system-prompt <text>       System prompt for AI assistants
    --system-prompt-file <file>  Read system prompt from file
    --persona <text>             Persona description for the AI
    --examples <file>            JSON file with ExamplePair array
    --author, -a <name>          Package author
    --language, -l <code>        Language code (e.g., "en", "es")
    --license <id>               License identifier (e.g., "MIT", "CC-BY-4.0")
    --tag, -t <tag>              Add a tag (can be repeated)
    --url <url>                  URL for more information

    RAG Configuration Options:
    --chunk-size <tokens>        Preferred chunk size for RAG
    --chunk-overlap <tokens>     Overlap between chunks
    --top-k <number>             Number of results to retrieve
    --min-relevance <score>      Minimum relevance score (0.0-1.0)
    --priority-boost <number>    Priority boost for all documents

LIST OPTIONS:
    <file.kpkg>                  The package to list
    --passphrase, -p <pass>      Custom passphrase if not using app key

INSPECT OPTIONS:
    <file.kpkg>                  The package to inspect
    --passphrase, -p <pass>      Custom passphrase if not using app key
    --json                       Output as JSON instead of formatted text

EXTRACT OPTIONS:
    --input, -i <file>           The .kpkg file to extract
    --output, -o <folder>        Output folder
    --passphrase, -p <pass>      Custom passphrase if not using app key

EXAMPLES:
    # Basic package creation
    kpkg_tool create -i ./knowledge -o knowledge.kpkg -n "My Knowledge Base"

    # Professional package with all options
    kpkg_tool create -i ./docs -o docs.kpkg \
        -n "Star Citizen Guide" \
        -d "Comprehensive guide for Star Citizen" \
        -a "GameDev Team" \
        -l "en" \
        --license "CC-BY-4.0" \
        --system-prompt "You are a helpful Star Citizen expert." \
        --persona "Expert pilot with years of experience" \
        --examples examples.json \
        -t "gaming" -t "guide" -t "star-citizen" \
        --chunk-size 512 \
        --top-k 5

    # List package contents
    kpkg_tool list knowledge.kpkg

    # Inspect manifest details
    kpkg_tool inspect knowledge.kpkg
    kpkg_tool inspect knowledge.kpkg --json

    # Extract contents
    kpkg_tool extract -i knowledge.kpkg -o ./extracted
"#
    );
}

/// Options collected from command-line arguments for create command
struct CreateOptions {
    input: PathBuf,
    output: PathBuf,
    name: Option<String>,
    description: Option<String>,
    passphrase: Option<String>,
    // Professional options
    system_prompt: Option<String>,
    persona: Option<String>,
    examples: Vec<ExamplePair>,
    author: Option<String>,
    language: Option<String>,
    license: Option<String>,
    tags: Vec<String>,
    url: Option<String>,
    // RAG config
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    top_k: Option<usize>,
    min_relevance: Option<f32>,
    priority_boost: Option<i32>,
}

fn run_create(args: &[String]) -> Result<(), String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut passphrase: Option<String> = None;
    // Professional options
    let mut system_prompt: Option<String> = None;
    let mut persona: Option<String> = None;
    let mut examples: Vec<ExamplePair> = Vec::new();
    let mut author: Option<String> = None;
    let mut language: Option<String> = None;
    let mut license: Option<String> = None;
    let mut tags: Vec<String> = Vec::new();
    let mut url: Option<String> = None;
    // RAG config
    let mut chunk_size: Option<usize> = None;
    let mut chunk_overlap: Option<usize> = None;
    let mut top_k: Option<usize> = None;
    let mut min_relevance: Option<f32> = None;
    let mut priority_boost: Option<i32> = None;

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
            "--system-prompt" => {
                i += 1;
                system_prompt = Some(args.get(i).ok_or("Missing system prompt")?.clone());
            }
            "--system-prompt-file" => {
                i += 1;
                let path = args.get(i).ok_or("Missing system prompt file path")?;
                system_prompt = Some(
                    fs::read_to_string(path)
                        .map_err(|e| format!("Failed to read system prompt file: {}", e))?,
                );
            }
            "--persona" => {
                i += 1;
                persona = Some(args.get(i).ok_or("Missing persona")?.clone());
            }
            "--examples" => {
                i += 1;
                let path = args.get(i).ok_or("Missing examples file path")?;
                let content = fs::read_to_string(path)
                    .map_err(|e| format!("Failed to read examples file: {}", e))?;
                examples = serde_json::from_str(&content)
                    .map_err(|e| format!("Failed to parse examples JSON: {}", e))?;
            }
            "--author" | "-a" => {
                i += 1;
                author = Some(args.get(i).ok_or("Missing author")?.clone());
            }
            "--language" | "-l" => {
                i += 1;
                language = Some(args.get(i).ok_or("Missing language")?.clone());
            }
            "--license" => {
                i += 1;
                license = Some(args.get(i).ok_or("Missing license")?.clone());
            }
            "--tag" | "-t" => {
                i += 1;
                tags.push(args.get(i).ok_or("Missing tag")?.clone());
            }
            "--url" => {
                i += 1;
                url = Some(args.get(i).ok_or("Missing URL")?.clone());
            }
            "--chunk-size" => {
                i += 1;
                chunk_size = Some(
                    args.get(i)
                        .ok_or("Missing chunk size")?
                        .parse()
                        .map_err(|_| "Invalid chunk size")?,
                );
            }
            "--chunk-overlap" => {
                i += 1;
                chunk_overlap = Some(
                    args.get(i)
                        .ok_or("Missing chunk overlap")?
                        .parse()
                        .map_err(|_| "Invalid chunk overlap")?,
                );
            }
            "--top-k" => {
                i += 1;
                top_k = Some(
                    args.get(i)
                        .ok_or("Missing top-k")?
                        .parse()
                        .map_err(|_| "Invalid top-k")?,
                );
            }
            "--min-relevance" => {
                i += 1;
                min_relevance = Some(
                    args.get(i)
                        .ok_or("Missing min-relevance")?
                        .parse()
                        .map_err(|_| "Invalid min-relevance")?,
                );
            }
            "--priority-boost" => {
                i += 1;
                priority_boost = Some(
                    args.get(i)
                        .ok_or("Missing priority-boost")?
                        .parse()
                        .map_err(|_| "Invalid priority-boost")?,
                );
            }
            arg => {
                return Err(format!("Unknown argument: {}", arg));
            }
        }
        i += 1;
    }

    let options = CreateOptions {
        input: input.ok_or("Missing required --input")?,
        output: output.ok_or("Missing required --output")?,
        name,
        description,
        passphrase,
        system_prompt,
        persona,
        examples,
        author,
        language,
        license,
        tags,
        url,
        chunk_size,
        chunk_overlap,
        top_k,
        min_relevance,
        priority_boost,
    };

    if !options.input.exists() {
        return Err(format!(
            "Input folder does not exist: {}",
            options.input.display()
        ));
    }

    if !options.input.is_dir() {
        return Err(format!(
            "Input is not a directory: {}",
            options.input.display()
        ));
    }

    // Collect all documents from the folder
    let docs = collect_documents(&options.input)?;

    if docs.is_empty() {
        return Err("No .md or .txt files found in input folder".into());
    }

    println!("Found {} documents", docs.len());

    // Build the package
    let encrypted = if let Some(ref pass) = options.passphrase {
        build_package_with_key(CustomKeyProvider::new(pass.clone()), &docs, &options)?
    } else {
        build_package_with_key(AppKeyProvider, &docs, &options)?
    };

    // Write to output
    fs::write(&options.output, &encrypted)
        .map_err(|e| format!("Failed to write output file: {}", e))?;

    println!(
        "Created package: {} ({} bytes)",
        options.output.display(),
        encrypted.len()
    );

    // Print summary of professional options used
    let mut features: Vec<String> = Vec::new();
    if options.system_prompt.is_some() {
        features.push("system_prompt".to_string());
    }
    if options.persona.is_some() {
        features.push("persona".to_string());
    }
    if !options.examples.is_empty() {
        features.push(format!("{} examples", options.examples.len()));
    }
    if options.author.is_some() || options.license.is_some() || !options.tags.is_empty() {
        features.push("metadata".to_string());
    }
    if options.chunk_size.is_some() || options.top_k.is_some() {
        features.push("rag_config".to_string());
    }

    if !features.is_empty() {
        println!("Professional features: {}", features.join(", "));
    }

    Ok(())
}

fn build_package_with_key<K: KeyProvider>(
    key_provider: K,
    docs: &[(String, String)],
    options: &CreateOptions,
) -> Result<Vec<u8>, String> {
    let mut builder = KpkgBuilder::with_key_provider(key_provider);

    if let Some(ref n) = options.name {
        builder = builder.name(n.clone());
    }

    if let Some(ref d) = options.description {
        builder = builder.description(d.clone());
    }

    // Professional options
    if let Some(ref sp) = options.system_prompt {
        builder = builder.system_prompt(sp.clone());
    }

    if let Some(ref p) = options.persona {
        builder = builder.persona(p.clone());
    }

    for example in &options.examples {
        if let Some(ref cat) = example.category {
            builder = builder.add_example_with_category(
                example.input.clone(),
                example.output.clone(),
                cat.clone(),
            );
        } else {
            builder = builder.add_example(example.input.clone(), example.output.clone());
        }
    }

    // Metadata
    if let Some(ref a) = options.author {
        builder = builder.author(a.clone());
    }

    if let Some(ref l) = options.language {
        builder = builder.language(l.clone());
    }

    if let Some(ref lic) = options.license {
        builder = builder.license(lic.clone());
    }

    for tag in &options.tags {
        builder = builder.add_tag(tag.clone());
    }

    if let Some(ref u) = options.url {
        builder = builder.url(u.clone());
    }

    // RAG config
    if let Some(cs) = options.chunk_size {
        builder = builder.chunk_size(cs);
    }

    if let Some(co) = options.chunk_overlap {
        builder = builder.chunk_overlap(co);
    }

    if let Some(tk) = options.top_k {
        builder = builder.top_k(tk);
    }

    if let Some(mr) = options.min_relevance {
        builder = builder.min_relevance(mr);
    }

    if let Some(pb) = options.priority_boost {
        builder = builder.priority_boost(pb);
    }

    // Add timestamps
    builder = builder.with_current_timestamps();

    // Add documents
    for (path, content) in docs {
        builder = builder.add_document(path.clone(), content.clone(), None);
    }

    builder
        .build()
        .map_err(|e| format!("Failed to build package: {}", e))
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
        reader
            .read(&data)
            .map_err(|e| format!("Failed to read package: {}", e))?
    } else {
        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        reader
            .read(&data)
            .map_err(|e| format!("Failed to read package: {}", e))?
    };

    println!(
        "Package: {} ({} bytes encrypted)",
        file.display(),
        data.len()
    );
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

fn run_inspect(args: &[String]) -> Result<(), String> {
    let mut file: Option<PathBuf> = None;
    let mut passphrase: Option<String> = None;
    let mut json_output = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--passphrase" | "-p" => {
                i += 1;
                passphrase = Some(args.get(i).ok_or("Missing passphrase")?.clone());
            }
            "--json" => {
                json_output = true;
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

    let manifest = if let Some(pass) = passphrase {
        let reader = KpkgReader::with_key_provider(CustomKeyProvider::new(pass));
        reader
            .read_manifest_only(&data)
            .map_err(|e| format!("Failed to read package: {}", e))?
    } else {
        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        reader
            .read_manifest_only(&data)
            .map_err(|e| format!("Failed to read package: {}", e))?
    };

    if json_output {
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("Failed to serialize manifest: {}", e))?;
        println!("{}", json);
    } else {
        print_manifest_formatted(&manifest, &file, data.len());
    }

    Ok(())
}

fn print_manifest_formatted(manifest: &KpkgManifest, file: &Path, file_size: usize) {
    println!("=== KPKG Manifest ===");
    println!("File: {} ({} bytes)", file.display(), file_size);
    println!();

    // Basic info
    println!("--- Basic Information ---");
    println!(
        "Name:        {}",
        if manifest.name.is_empty() {
            "(not set)"
        } else {
            &manifest.name
        }
    );
    println!(
        "Description: {}",
        if manifest.description.is_empty() {
            "(not set)"
        } else {
            &manifest.description
        }
    );
    println!(
        "Version:     {}",
        if manifest.version.is_empty() {
            "(not set)"
        } else {
            &manifest.version
        }
    );
    println!();

    // Priorities
    if !manifest.priorities.is_empty() || manifest.default_priority != 0 {
        println!("--- Document Priorities ---");
        println!("Default priority: {}", manifest.default_priority);
        if !manifest.priorities.is_empty() {
            println!("Custom priorities:");
            for (path, priority) in &manifest.priorities {
                println!("  {} -> {}", path, priority);
            }
        }
        println!();
    }

    // System prompt and persona
    if manifest.system_prompt.is_some() || manifest.persona.is_some() {
        println!("--- AI Configuration ---");
        if let Some(ref sp) = manifest.system_prompt {
            println!("System Prompt:");
            for line in sp.lines().take(5) {
                println!("  {}", line);
            }
            if sp.lines().count() > 5 {
                println!("  ... ({} more lines)", sp.lines().count() - 5);
            }
        }
        if let Some(ref p) = manifest.persona {
            println!("Persona: {}", p);
        }
        println!();
    }

    // Examples
    if !manifest.examples.is_empty() {
        println!("--- Few-Shot Examples ({}) ---", manifest.examples.len());
        for (i, ex) in manifest.examples.iter().enumerate().take(3) {
            println!("Example {}:", i + 1);
            if let Some(ref cat) = ex.category {
                println!("  Category: {}", cat);
            }
            println!("  Input:  {}", truncate_string(&ex.input, 60));
            println!("  Output: {}", truncate_string(&ex.output, 60));
        }
        if manifest.examples.len() > 3 {
            println!("  ... ({} more examples)", manifest.examples.len() - 3);
        }
        println!();
    }

    // RAG config
    if let Some(ref rag) = manifest.rag_config {
        if !rag.is_empty() {
            println!("--- RAG Configuration ---");
            if let Some(cs) = rag.chunk_size {
                println!("Chunk size:      {} tokens", cs);
            }
            if let Some(co) = rag.chunk_overlap {
                println!("Chunk overlap:   {} tokens", co);
            }
            if let Some(tk) = rag.top_k {
                println!("Top-K:           {}", tk);
            }
            if let Some(mct) = rag.max_context_tokens {
                println!("Max context:     {} tokens", mct);
            }
            if let Some(mr) = rag.min_relevance {
                println!("Min relevance:   {:.2}", mr);
            }
            if let Some(ref cs) = rag.chunking_strategy {
                println!("Chunking:        {}", cs);
            }
            if let Some(hs) = rag.use_hybrid_search {
                println!(
                    "Hybrid search:   {}",
                    if hs { "enabled" } else { "disabled" }
                );
            }
            if let Some(pb) = rag.priority_boost {
                println!("Priority boost:  {}", pb);
            }
            println!();
        }
    }

    // Metadata
    if let Some(ref meta) = manifest.metadata {
        if !meta.is_empty() {
            println!("--- Package Metadata ---");
            if let Some(ref a) = meta.author {
                println!("Author:     {}", a);
            }
            if let Some(ref l) = meta.language {
                println!("Language:   {}", l);
            }
            if let Some(ref lic) = meta.license {
                println!("License:    {}", lic);
            }
            if let Some(ref u) = meta.url {
                println!("URL:        {}", u);
            }
            if let Some(ref ca) = meta.created_at {
                println!("Created:    {}", ca);
            }
            if let Some(ref ua) = meta.updated_at {
                println!("Updated:    {}", ua);
            }
            if !meta.tags.is_empty() {
                println!("Tags:       {}", meta.tags.join(", "));
            }
            if !meta.custom.is_empty() {
                println!("Custom fields:");
                for (k, v) in &meta.custom {
                    println!("  {}: {}", k, v);
                }
            }
            println!();
        }
    }
}

fn truncate_string(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ").replace('\r', "");
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s
    }
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
        reader
            .read(&data)
            .map_err(|e| format!("Failed to read package: {}", e))?
    } else {
        let reader = KpkgReader::<AppKeyProvider>::with_app_key();
        reader
            .read(&data)
            .map_err(|e| format!("Failed to read package: {}", e))?
    };

    // Create output directory
    fs::create_dir_all(&output).map_err(|e| format!("Failed to create output directory: {}", e))?;

    println!(
        "Extracting {} documents to {}",
        docs.len(),
        output.display()
    );

    for doc in &docs {
        let doc_path = output.join(&doc.path);

        // Create parent directories
        if let Some(parent) = doc_path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        fs::write(&doc_path, &doc.content)
            .map_err(|e| format!("Failed to write {}: {}", doc_path.display(), e))?;

        println!("  Extracted: {}", doc.path);
    }

    println!("Done!");

    Ok(())
}
