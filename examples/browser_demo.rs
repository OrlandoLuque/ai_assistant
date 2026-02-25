//! Browser automation demo — Chrome DevTools Protocol types.
//!
//! Demonstrates the browser session API, error types, and Chrome binary
//! detection. This example does NOT launch a real browser; it shows the
//! configuration and type system.
//!
//! Run with: cargo run --example browser_demo --features "browser"

use ai_assistant::{
    find_chrome_binary, BrowserError, BrowserPageContent, BrowserSession,
};

fn main() {
    println!("=== Browser Automation Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Detect Chrome binary
    // -----------------------------------------------------------------------
    println!("--- Chrome binary detection ---");
    match find_chrome_binary() {
        Some(path) => println!("Chrome found at: {}", path.display()),
        None => println!("Chrome not found on this system"),
    }

    // -----------------------------------------------------------------------
    // 2. Create a BrowserSession (not connected yet)
    // -----------------------------------------------------------------------
    println!("\n--- BrowserSession lifecycle ---");
    let session = BrowserSession::new();
    println!("Session created (connected: {})", session.is_connected());
    println!("Current URL: {:?}", session.current_url());
    println!("Current title: {:?}", session.current_title());

    // -----------------------------------------------------------------------
    // 3. Demonstrate BrowserError variants
    // -----------------------------------------------------------------------
    println!("\n--- BrowserError variants ---");
    let errors: Vec<BrowserError> = vec![
        BrowserError::ChromeNotFound,
        BrowserError::LaunchFailed("Port already in use".into()),
        BrowserError::NotConnected,
        BrowserError::WebSocketError("Connection reset".into()),
        BrowserError::CdpError("DOM.querySelector returned null".into()),
        BrowserError::Timeout("Navigation took >30s".into()),
        BrowserError::ElementNotFound("#missing-button".into()),
    ];
    for err in &errors {
        println!("  {}", err);
    }

    // -----------------------------------------------------------------------
    // 4. Construct a PageContent value manually
    // -----------------------------------------------------------------------
    println!("\n--- BrowserPageContent (alias for PageContent) ---");
    let page = BrowserPageContent {
        url: "https://example.com".to_string(),
        title: "Example Domain".to_string(),
        html: "<html><body><h1>Example</h1></body></html>".to_string(),
        text: "Example Domain\nThis domain is for illustrative examples.".to_string(),
        screenshot: None,
    };
    println!("  URL: {}", page.url);
    println!("  Title: {}", page.title);
    println!("  HTML length: {} bytes", page.html.len());
    println!("  Text length: {} bytes", page.text.len());
    println!("  Screenshot: {:?}", page.screenshot);

    // -----------------------------------------------------------------------
    // 5. Show what a real workflow would look like (without actually connecting)
    // -----------------------------------------------------------------------
    println!("\n--- Typical workflow (pseudocode) ---");
    println!("  1. session.launch(9222)        // launch headless Chrome on port 9222");
    println!("  2. session.navigate(url)       // navigate and get PageContent");
    println!("  3. session.click(selector)     // click an element");
    println!("  4. session.type_text(sel, txt) // type into an input");
    println!("  5. session.evaluate(js)        // run arbitrary JavaScript");
    println!("  6. session.screenshot()        // capture base64 screenshot");
    println!("  7. session.close()             // close the browser");

    println!("\nBrowser demo complete.");
}
