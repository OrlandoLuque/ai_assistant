//! Pre-built egui widgets for AI chat interfaces
//!
//! This module provides reusable UI components for building chat interfaces
//! with egui. Enable with the `egui-widgets` feature.

use crate::context::ContextUsage;
use crate::messages::ChatMessage;
use crate::models::ModelInfo;
use crate::session::ChatSession;
use egui::{Color32, RichText, Ui, Vec2};

/// Colors for chat UI
pub struct ChatColors {
    pub user_bubble: Color32,
    pub assistant_bubble: Color32,
    pub system_bubble: Color32,
    pub user_text: Color32,
    pub assistant_text: Color32,
    pub error_text: Color32,
    pub muted_text: Color32,
}

impl Default for ChatColors {
    fn default() -> Self {
        Self {
            user_bubble: Color32::from_rgb(40, 60, 80),
            assistant_bubble: Color32::from_rgb(50, 50, 60),
            system_bubble: Color32::from_rgb(60, 50, 50),
            user_text: Color32::WHITE,
            assistant_text: Color32::WHITE,
            error_text: Color32::from_rgb(255, 100, 100),
            muted_text: Color32::GRAY,
        }
    }
}

/// Render a single chat message bubble
pub fn chat_message(ui: &mut Ui, msg: &ChatMessage, colors: &ChatColors, max_width: f32) {
    let is_user = msg.is_user();
    let is_system = msg.is_system();

    let frame_color = if is_system {
        colors.system_bubble
    } else if is_user {
        colors.user_bubble
    } else {
        colors.assistant_bubble
    };

    let (icon, role_name) = if is_system {
        ("💾", "System")
    } else if is_user {
        ("👤", "You")
    } else {
        ("🤖", "Assistant")
    };

    if is_user {
        // User messages aligned right
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            ui.add_space(10.0);
            egui::Frame::none()
                .fill(frame_color)
                .rounding(8.0)
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.set_max_width(max_width);

                    ui.horizontal(|ui| {
                        ui.label(RichText::new(icon).size(14.0));
                        ui.label(RichText::new(role_name).size(12.0).color(colors.muted_text));
                    });

                    ui.add_space(4.0);
                    ui.add(egui::Label::new(&msg.content).wrap(true));
                });
        });
    } else {
        // Assistant/system messages aligned left
        ui.horizontal(|ui| {
            ui.add_space(10.0);
            egui::Frame::none()
                .fill(frame_color)
                .rounding(8.0)
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.set_max_width(max_width);

                    ui.horizontal(|ui| {
                        ui.label(RichText::new(icon).size(14.0));
                        ui.label(RichText::new(role_name).size(12.0).color(colors.muted_text));
                    });

                    ui.add_space(4.0);
                    ui.add(egui::Label::new(&msg.content).wrap(true));
                });
        });
    }

    ui.add_space(8.0);
}

/// Render a streaming response bubble (while generating)
pub fn streaming_response(ui: &mut Ui, current_text: &str, colors: &ChatColors, max_width: f32) {
    ui.horizontal(|ui| {
        ui.add_space(10.0);
        egui::Frame::none()
            .fill(colors.assistant_bubble)
            .rounding(8.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                ui.set_max_width(max_width);

                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        RichText::new("Assistant")
                            .size(12.0)
                            .color(colors.muted_text),
                    );
                });

                ui.add_space(4.0);

                if current_text.is_empty() {
                    ui.label(
                        RichText::new("Thinking...")
                            .italics()
                            .color(colors.muted_text),
                    );
                } else {
                    ui.add(egui::Label::new(current_text).wrap(true));
                }
            });
    });
}

/// Render a "thinking" indicator
pub fn thinking_indicator(ui: &mut Ui, colors: &ChatColors) {
    ui.horizontal(|ui| {
        ui.add_space(10.0);
        egui::Frame::none()
            .fill(colors.assistant_bubble)
            .rounding(8.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        RichText::new("Thinking...")
                            .italics()
                            .color(colors.muted_text),
                    );
                });
            });
    });
}

/// Render an error message
pub fn error_message(ui: &mut Ui, error: &str, colors: &ChatColors) {
    ui.horizontal(|ui| {
        ui.add_space(10.0);
        egui::Frame::none()
            .fill(Color32::from_rgb(60, 30, 30))
            .rounding(8.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("⚠").size(14.0).color(colors.error_text));
                    ui.label(RichText::new("Error").size(12.0).color(colors.error_text));
                });

                ui.add_space(4.0);
                ui.label(RichText::new(error).color(colors.error_text));
            });
    });
}

/// Model selector dropdown widget
///
/// Returns the selected model if changed
pub fn model_selector(
    ui: &mut Ui,
    selected_model: &mut String,
    models: &[ModelInfo],
) -> Option<ModelInfo> {
    let mut selected: Option<ModelInfo> = None;

    let display_text = if selected_model.is_empty() {
        "Select a model...".to_string()
    } else {
        selected_model.clone()
    };

    egui::ComboBox::from_id_source("ai_model_selector")
        .selected_text(&display_text)
        .width(250.0)
        .show_ui(ui, |ui| {
            for model in models {
                let label = model.display_name_with_icon();

                if ui
                    .selectable_label(*selected_model == model.name, &label)
                    .clicked()
                {
                    *selected_model = model.name.clone();
                    selected = Some(model.clone());
                }
            }
        });

    selected
}

/// Context usage progress bar
pub fn context_usage_bar(ui: &mut Ui, usage: &ContextUsage, width: f32) {
    let (bar_color, text_color) = if usage.is_critical {
        (
            Color32::from_rgb(200, 60, 60),
            Color32::from_rgb(255, 150, 150),
        )
    } else if usage.is_warning {
        (
            Color32::from_rgb(200, 150, 60),
            Color32::from_rgb(255, 220, 150),
        )
    } else {
        (
            Color32::from_rgb(60, 150, 60),
            Color32::from_rgb(150, 255, 150),
        )
    };

    let progress = usage.usage_percent / 100.0;

    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 14.0), egui::Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 2.0, Color32::from_rgb(40, 40, 50));

        // Fill
        let fill_width = rect.width() * progress.min(1.0);
        if fill_width > 0.0 {
            painter.rect_filled(
                egui::Rect::from_min_size(rect.min, Vec2::new(fill_width, rect.height())),
                2.0,
                bar_color,
            );
        }

        // Text
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{:.0}%", usage.usage_percent),
            egui::FontId::proportional(10.0),
            Color32::WHITE,
        );
    }

    // Tooltip
    response.on_hover_text(format!(
        "Context Usage: {:.0}% ({} / {} tokens)\n\n\
         System: {} tokens\n\
         Knowledge: {} tokens\n\
         Conversation: {} tokens\n\n\
         {}",
        usage.usage_percent,
        usage.total_tokens,
        usage.max_tokens,
        usage.system_tokens,
        usage.knowledge_tokens,
        usage.conversation_tokens,
        if usage.is_critical {
            "Critical! Consider clearing chat."
        } else if usage.is_warning {
            "High usage. Old messages may be summarized."
        } else {
            "Plenty of context available."
        }
    ));

    ui.label(RichText::new("Context").size(10.0).color(text_color));

    if usage.is_critical {
        ui.label(RichText::new("⚠").size(12.0).color(Color32::RED));
    }
}

/// Session list sidebar widget
pub struct SessionListResponse {
    pub session_to_load: Option<String>,
    pub session_to_delete: Option<String>,
}

/// Render a session list sidebar
pub fn session_list(
    ui: &mut Ui,
    sessions: &[ChatSession],
    current_session_id: Option<&str>,
    max_height: f32,
) -> SessionListResponse {
    let mut response = SessionListResponse {
        session_to_load: None,
        session_to_delete: None,
    };

    egui::ScrollArea::vertical()
        .max_height(max_height)
        .show(ui, |ui| {
            for session in sessions.iter().rev() {
                let is_current = current_session_id == Some(&session.id);

                ui.horizontal(|ui| {
                    let frame_color = if is_current {
                        Color32::from_rgb(50, 60, 80)
                    } else {
                        Color32::TRANSPARENT
                    };

                    egui::Frame::none()
                        .fill(frame_color)
                        .rounding(4.0)
                        .inner_margin(4.0)
                        .show(ui, |ui| {
                            ui.set_min_width(150.0);

                            if ui.selectable_label(is_current, &session.name).clicked()
                                && !is_current
                            {
                                response.session_to_load = Some(session.id.clone());
                            }

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .small_button("🗑")
                                        .on_hover_text("Delete session")
                                        .clicked()
                                    {
                                        response.session_to_delete = Some(session.id.clone());
                                    }
                                },
                            );
                        });
                });
            }

            if sessions.is_empty() {
                ui.label(
                    RichText::new("No sessions yet")
                        .color(Color32::GRAY)
                        .size(11.0),
                );
            }
        });

    response
}

/// Input text field with send button
///
/// Returns Some(message) when the user submits
pub fn chat_input(
    ui: &mut Ui,
    input_text: &mut String,
    is_generating: bool,
    placeholder: &str,
) -> Option<String> {
    let mut submitted = None;

    ui.horizontal(|ui| {
        let response = ui.add_sized(
            Vec2::new(ui.available_width() - 70.0, 28.0),
            egui::TextEdit::singleline(input_text)
                .hint_text(placeholder)
                .interactive(!is_generating),
        );

        // Submit on Enter
        if response.lost_focus()
            && ui.input(|i| i.key_pressed(egui::Key::Enter))
            && !input_text.trim().is_empty()
        {
            submitted = Some(input_text.trim().to_string());
            input_text.clear();
        }

        ui.add_enabled_ui(!is_generating && !input_text.trim().is_empty(), |ui| {
            if ui.button("Send").clicked() {
                submitted = Some(input_text.trim().to_string());
                input_text.clear();
            }
        });
    });

    submitted
}

/// Multi-line chat input with send button.
///
/// `enter_sends`: if true, Enter sends and Ctrl+Enter inserts newline (default);
///                if false, Ctrl+Enter sends and Enter inserts newline.
/// Shift+Enter and Alt+Enter always insert a newline regardless of config.
///
/// Returns Some(message) when the user submits.
pub fn chat_input_multiline(
    ui: &mut Ui,
    input_text: &mut String,
    is_generating: bool,
    placeholder: &str,
    max_height: f32,
    enter_sends: bool,
) -> Option<String> {
    let mut submitted = None;

    let text_lines = input_text.lines().count().max(1).min(3);
    let input_height = (28.0 + (text_lines.saturating_sub(1) as f32 * 14.0)).min(max_height);

    ui.horizontal(|ui| {
        let response = ui.add_sized(
            Vec2::new(ui.available_width() - 70.0, input_height),
            egui::TextEdit::multiline(input_text)
                .hint_text(placeholder)
                .interactive(!is_generating)
                .desired_rows(1),
        );

        if response.has_focus() && !input_text.trim().is_empty() {
            let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
            let ctrl = ui.input(|i| i.modifiers.ctrl);
            let shift = ui.input(|i| i.modifiers.shift);
            let alt = ui.input(|i| i.modifiers.alt);

            if enter {
                if shift || alt {
                    // Shift+Enter / Alt+Enter → always insert newline
                    // (egui TextEdit::multiline handles the newline insertion)
                } else if enter_sends && !ctrl {
                    // Default mode: bare Enter → send
                    submitted = Some(input_text.trim().to_string());
                    input_text.clear();
                } else if !enter_sends && ctrl {
                    // Swapped mode: Ctrl+Enter → send
                    submitted = Some(input_text.trim().to_string());
                    input_text.clear();
                }
                // Otherwise the key combo inserts a newline via TextEdit
            }
        }

        ui.vertical(|ui| {
            ui.add_enabled_ui(!is_generating && !input_text.trim().is_empty(), |ui| {
                if ui.button("Send").clicked() {
                    submitted = Some(input_text.trim().to_string());
                    input_text.clear();
                }
            });

            let hint = if enter_sends { "Enter" } else { "Ctrl+Enter" };
            ui.label(
                RichText::new(hint)
                    .size(9.0)
                    .color(Color32::DARK_GRAY),
            );
        });
    });

    submitted
}

/// Quick suggestion buttons
pub fn suggestions(ui: &mut Ui, suggestions: &[&str]) -> Option<String> {
    let mut selected = None;

    ui.horizontal_wrapped(|ui| {
        for suggestion in suggestions {
            if ui.small_button(*suggestion).clicked() {
                selected = Some(suggestion.to_string());
            }
        }
    });

    selected
}

/// Welcome screen with suggestions
pub fn welcome_screen(
    ui: &mut Ui,
    title: &str,
    subtitle: &str,
    suggestions: &[&str],
) -> Option<String> {
    let mut selected = None;

    ui.vertical_centered(|ui| {
        ui.add_space(50.0);
        ui.label(RichText::new(title).size(24.0).color(Color32::WHITE));
        ui.add_space(10.0);
        ui.label(RichText::new(subtitle).size(14.0).color(Color32::GRAY));
        ui.add_space(20.0);

        ui.label(RichText::new("Try asking:").color(Color32::LIGHT_GRAY));
        ui.add_space(5.0);

        for suggestion in suggestions {
            if ui.button(*suggestion).clicked() {
                selected = Some(suggestion.to_string());
            }
        }
    });

    selected
}

/// Connection status indicator
pub fn connection_status(ui: &mut Ui, is_fetching: bool, model_count: usize) {
    if is_fetching {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(
                RichText::new("Searching...")
                    .color(Color32::YELLOW)
                    .size(11.0),
            );
        });
    } else if model_count == 0 {
        ui.label(
            RichText::new("⚪ No models found")
                .color(Color32::GRAY)
                .size(11.0),
        );
    } else {
        ui.label(
            RichText::new(format!("🟢 {} models", model_count))
                .color(Color32::LIGHT_GREEN)
                .size(11.0),
        );
    }
}

// === RAG Widgets ===

/// Configuration for RAG control widgets
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RagWidgetConfig {
    /// Whether the user can toggle knowledge RAG
    pub knowledge_rag_editable: bool,
    /// Whether the user can toggle conversation RAG
    pub conversation_rag_editable: bool,
    /// Show hint when context is full and RAG can help
    pub show_context_full_hint: bool,
    /// Label for knowledge RAG checkbox
    pub knowledge_label: String,
    /// Label for conversation RAG checkbox
    pub conversation_label: String,
    /// Tooltip for knowledge RAG
    pub knowledge_tooltip: String,
    /// Tooltip for conversation RAG
    pub conversation_tooltip: String,
}

impl Default for RagWidgetConfig {
    fn default() -> Self {
        Self {
            knowledge_rag_editable: true,
            conversation_rag_editable: true,
            show_context_full_hint: true,
            knowledge_label: "Smart Knowledge".to_string(),
            conversation_label: "Extended Memory".to_string(),
            knowledge_tooltip: "Search and retrieve only relevant parts of the knowledge base instead of loading everything".to_string(),
            conversation_tooltip: "Store conversation history in database for virtually unlimited context".to_string(),
        }
    }
}

impl RagWidgetConfig {
    /// Create a config where both RAG options are enabled and not editable
    pub fn locked_enabled() -> Self {
        Self {
            knowledge_rag_editable: false,
            conversation_rag_editable: false,
            show_context_full_hint: false,
            ..Default::default()
        }
    }

    /// Create a config where both RAG options are disabled and not editable
    pub fn locked_disabled() -> Self {
        Self {
            knowledge_rag_editable: false,
            conversation_rag_editable: false,
            show_context_full_hint: false,
            ..Default::default()
        }
    }

    /// Create a config where options are editable
    pub fn editable() -> Self {
        Self::default()
    }
}

/// Response from RAG controls widget
#[derive(Debug, Clone, Default)]
pub struct RagControlsResponse {
    /// Whether knowledge RAG was toggled
    pub knowledge_toggled: bool,
    /// New value for knowledge RAG (if toggled)
    pub knowledge_enabled: bool,
    /// Whether conversation RAG was toggled
    pub conversation_toggled: bool,
    /// New value for conversation RAG (if toggled)
    pub conversation_enabled: bool,
}

/// RAG status information for display
#[derive(Debug, Clone, Default)]
pub struct RagStatus {
    /// Whether RAG database is initialized
    pub rag_available: bool,
    /// Whether knowledge RAG is enabled
    pub knowledge_enabled: bool,
    /// Whether conversation RAG is enabled
    pub conversation_enabled: bool,
    /// Number of knowledge chunks indexed
    pub knowledge_chunks: usize,
    /// Total tokens in knowledge base
    pub knowledge_tokens: usize,
    /// Number of archived conversation messages
    pub archived_messages: usize,
    /// Tokens in archived messages
    pub archived_tokens: usize,
}

/// Render RAG control checkboxes
///
/// Returns information about any toggles that occurred
pub fn rag_controls(
    ui: &mut Ui,
    status: &RagStatus,
    config: &RagWidgetConfig,
) -> RagControlsResponse {
    let mut response = RagControlsResponse::default();

    if !status.rag_available {
        return response;
    }

    ui.horizontal(|ui| {
        // Knowledge RAG checkbox
        let mut knowledge = status.knowledge_enabled;
        let knowledge_response = if config.knowledge_rag_editable {
            ui.checkbox(&mut knowledge, &config.knowledge_label)
        } else {
            // Show as disabled checkbox
            ui.add_enabled(
                false,
                egui::Checkbox::new(&mut knowledge, &config.knowledge_label),
            )
        };

        if !config.knowledge_tooltip.is_empty() {
            knowledge_response.on_hover_text(&config.knowledge_tooltip);
        }

        if knowledge != status.knowledge_enabled {
            response.knowledge_toggled = true;
            response.knowledge_enabled = knowledge;
        }

        ui.add_space(10.0);

        // Conversation RAG checkbox
        let mut conversation = status.conversation_enabled;
        let conversation_response = if config.conversation_rag_editable {
            ui.checkbox(&mut conversation, &config.conversation_label)
        } else {
            ui.add_enabled(
                false,
                egui::Checkbox::new(&mut conversation, &config.conversation_label),
            )
        };

        if !config.conversation_tooltip.is_empty() {
            conversation_response.on_hover_text(&config.conversation_tooltip);
        }

        if conversation != status.conversation_enabled {
            response.conversation_toggled = true;
            response.conversation_enabled = conversation;
        }
    });

    response
}

/// Render RAG status information (compact version)
pub fn rag_status_compact(ui: &mut Ui, status: &RagStatus) {
    if !status.rag_available {
        return;
    }

    ui.horizontal(|ui| {
        if status.knowledge_enabled {
            ui.label(
                RichText::new(format!("📚 {} chunks", status.knowledge_chunks))
                    .size(10.0)
                    .color(Color32::LIGHT_BLUE),
            )
            .on_hover_text(format!("{} tokens indexed", status.knowledge_tokens));
        }

        if status.conversation_enabled && status.archived_messages > 0 {
            ui.label(
                RichText::new(format!("💬 {} archived", status.archived_messages))
                    .size(10.0)
                    .color(Color32::LIGHT_GREEN),
            )
            .on_hover_text(format!("{} tokens in archive", status.archived_tokens));
        }
    });
}

/// Render a hint when context is full and RAG can help
///
/// Returns (enable_knowledge, enable_conversation) if user clicks the enable button
pub fn context_full_hint(
    ui: &mut Ui,
    can_help_knowledge: bool,
    can_help_conversation: bool,
    estimated_savings: usize,
    config: &RagWidgetConfig,
) -> (bool, bool) {
    // Don't show if hints are disabled or nothing can help
    if !config.show_context_full_hint || (!can_help_knowledge && !can_help_conversation) {
        return (false, false);
    }

    // Only show if at least one option is editable
    let any_editable = (can_help_knowledge && config.knowledge_rag_editable)
        || (can_help_conversation && config.conversation_rag_editable);

    if !any_editable {
        return (false, false);
    }

    let mut enable_knowledge = false;
    let mut enable_conversation = false;

    egui::Frame::none()
        .fill(Color32::from_rgb(60, 50, 30))
        .rounding(6.0)
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("💡").size(14.0));
                ui.label(
                    RichText::new("Context is getting full!")
                        .color(Color32::from_rgb(255, 220, 100))
                        .size(12.0),
                );
            });

            ui.add_space(4.0);

            let mut hint_parts = Vec::new();
            if can_help_knowledge && config.knowledge_rag_editable {
                hint_parts.push("Smart Knowledge");
            }
            if can_help_conversation && config.conversation_rag_editable {
                hint_parts.push("Extended Memory");
            }

            ui.label(
                RichText::new(format!(
                    "Enable {} to save ~{} tokens and continue chatting.",
                    hint_parts.join(" and/or "),
                    estimated_savings
                ))
                .size(11.0)
                .color(Color32::LIGHT_GRAY),
            );

            ui.add_space(4.0);

            ui.horizontal(|ui| {
                if can_help_knowledge && config.knowledge_rag_editable {
                    if ui.small_button("Enable Knowledge RAG").clicked() {
                        enable_knowledge = true;
                    }
                }

                if can_help_conversation && config.conversation_rag_editable {
                    if ui.small_button("Enable Memory RAG").clicked() {
                        enable_conversation = true;
                    }
                }

                if can_help_knowledge
                    && can_help_conversation
                    && config.knowledge_rag_editable
                    && config.conversation_rag_editable
                {
                    if ui.small_button("Enable Both").clicked() {
                        enable_knowledge = true;
                        enable_conversation = true;
                    }
                }
            });
        });

    (enable_knowledge, enable_conversation)
}

/// Render RAG detailed status panel
pub fn rag_status_panel(ui: &mut Ui, status: &RagStatus) {
    if !status.rag_available {
        ui.label(
            RichText::new("RAG not initialized")
                .color(Color32::GRAY)
                .size(11.0),
        );
        return;
    }

    egui::Grid::new("rag_status_grid")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            // Knowledge status
            ui.label(RichText::new("Knowledge RAG:").size(11.0));
            if status.knowledge_enabled {
                ui.label(
                    RichText::new(format!(
                        "✓ {} chunks ({} tokens)",
                        status.knowledge_chunks, status.knowledge_tokens
                    ))
                    .color(Color32::LIGHT_GREEN)
                    .size(11.0),
                );
            } else {
                ui.label(RichText::new("Disabled").color(Color32::GRAY).size(11.0));
            }
            ui.end_row();

            // Conversation status
            ui.label(RichText::new("Conversation RAG:").size(11.0));
            if status.conversation_enabled {
                ui.label(
                    RichText::new(format!(
                        "✓ {} archived ({} tokens)",
                        status.archived_messages, status.archived_tokens
                    ))
                    .color(Color32::LIGHT_GREEN)
                    .size(11.0),
                );
            } else {
                ui.label(RichText::new("Disabled").color(Color32::GRAY).size(11.0));
            }
            ui.end_row();
        });
}

// === Notes Widgets ===

/// Complete Notes Manager widget that handles all notes types
/// This widget manages session notes, global notes, and knowledge notes
/// as a cohesive unit with all UI and state management built-in.
pub struct NotesManager {
    /// Internal state
    state: NotesManagerState,
    /// Configuration
    config: NotesManagerConfig,
}

/// Internal state for NotesManager
#[derive(Debug, Clone, Default)]
struct NotesManagerState {
    /// Which editor is currently open
    active_editor: Option<NotesEditorType>,
    /// Temporary editing buffer
    edit_buffer: String,
    /// Selected knowledge source (for knowledge notes)
    selected_source: Option<String>,
    /// Whether there are unsaved changes
    has_changes: bool,
}

/// Type of notes editor currently active
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NotesEditorType {
    Session,
    Global,
    Knowledge,
}

/// Configuration for NotesManager widget
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NotesManagerConfig {
    /// Enable session notes
    pub session_enabled: bool,
    /// Enable global notes
    pub global_enabled: bool,
    /// Enable knowledge notes
    pub knowledge_enabled: bool,
    /// Session notes button label
    pub session_label: String,
    /// Global notes button label
    pub global_label: String,
    /// Knowledge notes button label
    pub knowledge_label: String,
    /// Session notes tooltip
    pub session_tooltip: String,
    /// Global notes tooltip
    pub global_tooltip: String,
    /// Knowledge notes tooltip
    pub knowledge_tooltip: String,
    /// Session notes editor title
    pub session_title: String,
    /// Global notes editor title
    pub global_title: String,
    /// Knowledge notes editor title
    pub knowledge_title: String,
}

impl Default for NotesManagerConfig {
    fn default() -> Self {
        Self {
            session_enabled: true,
            global_enabled: true,
            knowledge_enabled: true,
            session_label: "Session Notes".to_string(),
            global_label: "Global Notes".to_string(),
            knowledge_label: "Guide Notes".to_string(),
            session_tooltip: "Notes specific to this conversation".to_string(),
            global_tooltip: "Notes that apply to all conversations".to_string(),
            knowledge_tooltip: "Notes for specific knowledge guides".to_string(),
            session_title: "Session Notes".to_string(),
            global_title: "Global Notes".to_string(),
            knowledge_title: "Knowledge Notes".to_string(),
        }
    }
}

/// Response from NotesManager widget
#[derive(Debug, Clone, Default)]
pub struct NotesManagerResponse {
    /// Session notes were saved - contains new value
    pub session_saved: Option<String>,
    /// Global notes were saved - contains new value
    pub global_saved: Option<String>,
    /// Knowledge notes were saved - contains (source, notes)
    pub knowledge_saved: Option<(String, String)>,
}

impl NotesManager {
    /// Create a new NotesManager with default config
    pub fn new() -> Self {
        Self {
            state: NotesManagerState::default(),
            config: NotesManagerConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: NotesManagerConfig) -> Self {
        Self {
            state: NotesManagerState::default(),
            config,
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &NotesManagerConfig {
        &self.config
    }

    /// Get mutable access to configuration
    pub fn config_mut(&mut self) -> &mut NotesManagerConfig {
        &mut self.config
    }

    /// Check if any editor is currently open
    pub fn is_editor_open(&self) -> bool {
        self.state.active_editor.is_some()
    }

    /// Open session notes editor
    pub fn open_session_notes(&mut self, current_value: &str) {
        self.state.active_editor = Some(NotesEditorType::Session);
        self.state.edit_buffer = current_value.to_string();
        self.state.has_changes = false;
    }

    /// Open global notes editor
    pub fn open_global_notes(&mut self, current_value: &str) {
        self.state.active_editor = Some(NotesEditorType::Global);
        self.state.edit_buffer = current_value.to_string();
        self.state.has_changes = false;
    }

    /// Open knowledge notes editor for a specific source
    pub fn open_knowledge_notes(&mut self, source: &str, current_value: &str) {
        self.state.active_editor = Some(NotesEditorType::Knowledge);
        self.state.selected_source = Some(source.to_string());
        self.state.edit_buffer = current_value.to_string();
        self.state.has_changes = false;
    }

    /// Close any open editor without saving
    pub fn close_editor(&mut self) {
        self.state.active_editor = None;
        self.state.edit_buffer.clear();
        self.state.selected_source = None;
        self.state.has_changes = false;
    }

    /// Render the notes buttons toolbar
    /// Returns which button was clicked (if any)
    pub fn render_buttons(
        &mut self,
        ui: &mut Ui,
        session_notes: &str,
        global_notes: &str,
    ) -> (bool, bool, bool) {
        let mut open_session = false;
        let mut open_global = false;
        let mut open_knowledge = false;

        ui.horizontal(|ui| {
            // Session notes button
            if self.config.session_enabled {
                let has_notes = !session_notes.is_empty();
                let icon = "📝";
                let label = format!("{} {}", icon, self.config.session_label);

                let button = if has_notes {
                    ui.button(RichText::new(&label).color(Color32::LIGHT_BLUE))
                } else {
                    ui.button(&label)
                };

                if button.on_hover_text(&self.config.session_tooltip).clicked() {
                    open_session = true;
                }
            }

            // Global notes button
            if self.config.global_enabled {
                let has_notes = !global_notes.is_empty();
                let icon = "📋";
                let label = format!("{} {}", icon, self.config.global_label);

                let button = if has_notes {
                    ui.button(RichText::new(&label).color(Color32::LIGHT_GREEN))
                } else {
                    ui.button(&label)
                };

                if button.on_hover_text(&self.config.global_tooltip).clicked() {
                    open_global = true;
                }
            }

            // Knowledge notes button
            if self.config.knowledge_enabled {
                let label = format!("📚 {}", self.config.knowledge_label);
                if ui
                    .button(&label)
                    .on_hover_text(&self.config.knowledge_tooltip)
                    .clicked()
                {
                    open_knowledge = true;
                }
            }
        });

        (open_session, open_global, open_knowledge)
    }

    /// Render the currently active editor window
    ///
    /// # Arguments
    /// * `ctx` - egui context
    /// * `available_sources` - list of knowledge sources (only needed for knowledge notes)
    /// * `get_knowledge_notes` - function to get notes for a knowledge source
    pub fn render_editor<F>(
        &mut self,
        ctx: &egui::Context,
        available_sources: &[String],
        get_knowledge_notes: F,
    ) -> NotesManagerResponse
    where
        F: FnMut(&str) -> String,
    {
        let mut response = NotesManagerResponse::default();

        let editor_type = match &self.state.active_editor {
            Some(t) => t.clone(),
            None => return response,
        };

        match editor_type {
            NotesEditorType::Session => {
                if let Some(notes) = self.render_session_editor(ctx) {
                    response.session_saved = Some(notes);
                }
            }
            NotesEditorType::Global => {
                if let Some(notes) = self.render_global_editor(ctx) {
                    response.global_saved = Some(notes);
                }
            }
            NotesEditorType::Knowledge => {
                if let Some((source, notes)) =
                    self.render_knowledge_editor(ctx, available_sources, get_knowledge_notes)
                {
                    response.knowledge_saved = Some((source, notes));
                }
            }
        }

        response
    }

    fn render_session_editor(&mut self, ctx: &egui::Context) -> Option<String> {
        let mut result = None;
        let mut close = false;

        egui::Window::new(&self.config.session_title)
            .collapsible(false)
            .resizable(true)
            .default_size([400.0, 300.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(
                    RichText::new(
                        "Add notes to help the AI understand context for this conversation:",
                    )
                    .size(11.0)
                    .color(Color32::GRAY),
                );
                ui.add_space(8.0);

                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        let response = ui.add(
                            egui::TextEdit::multiline(&mut self.state.edit_buffer)
                                .desired_width(f32::INFINITY)
                                .desired_rows(8)
                                .hint_text(
                                    "e.g., 'I prefer concise answers' or 'Focus on combat ships'",
                                ),
                        );
                        if response.changed() {
                            self.state.has_changes = true;
                        }
                    });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        result = Some(self.state.edit_buffer.clone());
                        close = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("Clear").clicked() {
                            self.state.edit_buffer.clear();
                            self.state.has_changes = true;
                        }
                    });
                });
            });

        if close {
            self.close_editor();
        }

        result
    }

    fn render_global_editor(&mut self, ctx: &egui::Context) -> Option<String> {
        let mut result = None;
        let mut close = false;

        egui::Window::new(&self.config.global_title)
            .collapsible(false)
            .resizable(true)
            .default_size([400.0, 300.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(
                    RichText::new("Global notes apply to ALL conversations:")
                        .size(11.0)
                        .color(Color32::GRAY),
                );
                ui.add_space(8.0);

                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        let response = ui.add(
                            egui::TextEdit::multiline(&mut self.state.edit_buffer)
                                .desired_width(f32::INFINITY)
                                .desired_rows(8)
                                .hint_text("e.g., 'My budget is $500' or 'I own a Cutlass Black'"),
                        );
                        if response.changed() {
                            self.state.has_changes = true;
                        }
                    });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        result = Some(self.state.edit_buffer.clone());
                        close = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("Clear").clicked() {
                            self.state.edit_buffer.clear();
                            self.state.has_changes = true;
                        }
                    });
                });
            });

        if close {
            self.close_editor();
        }

        result
    }

    fn render_knowledge_editor<F>(
        &mut self,
        ctx: &egui::Context,
        available_sources: &[String],
        mut get_notes: F,
    ) -> Option<(String, String)>
    where
        F: FnMut(&str) -> String,
    {
        let mut result = None;
        let mut close = false;

        egui::Window::new(&self.config.knowledge_title)
            .collapsible(false)
            .resizable(true)
            .default_size([450.0, 350.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(
                    RichText::new("Add notes for specific knowledge guides:")
                        .size(11.0)
                        .color(Color32::GRAY),
                );
                ui.add_space(8.0);

                // Source selector
                ui.horizontal(|ui| {
                    ui.label("Select guide:");

                    let current_text = self
                        .state
                        .selected_source
                        .as_deref()
                        .unwrap_or("Choose a guide...");

                    egui::ComboBox::from_id_source("notes_knowledge_source_selector")
                        .selected_text(current_text)
                        .width(250.0)
                        .show_ui(ui, |ui| {
                            for source in available_sources {
                                let was_selected =
                                    self.state.selected_source.as_deref() == Some(source);
                                if ui.selectable_label(was_selected, source).clicked() {
                                    let notes = get_notes(source);
                                    self.state.selected_source = Some(source.clone());
                                    self.state.edit_buffer = notes;
                                    self.state.has_changes = false;
                                }
                            }
                        });
                });

                ui.add_space(8.0);

                if self.state.selected_source.is_some() {
                    egui::ScrollArea::vertical()
                        .max_height(180.0)
                        .show(ui, |ui| {
                            let response = ui.add(
                                egui::TextEdit::multiline(&mut self.state.edit_buffer)
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(6)
                                    .hint_text("Add notes about this guide"),
                            );
                            if response.changed() {
                                self.state.has_changes = true;
                            }
                        });
                } else {
                    ui.label(
                        RichText::new("Select a guide from the dropdown above")
                            .color(Color32::GRAY)
                            .italics(),
                    );
                }

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(self.state.selected_source.is_some(), |ui| {
                        if ui.button("Save").clicked() {
                            if let Some(ref source) = self.state.selected_source {
                                result = Some((source.clone(), self.state.edit_buffer.clone()));
                            }
                            close = true;
                        }
                    });
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.state.selected_source.is_some()
                            && ui.small_button("Clear").clicked()
                        {
                            self.state.edit_buffer.clear();
                            self.state.has_changes = true;
                        }
                    });
                });
            });

        if close {
            self.close_editor();
        }

        result
    }
}

impl Default for NotesManager {
    fn default() -> Self {
        Self::new()
    }
}

// === Legacy Notes Widgets (for backwards compatibility) ===

/// State for managing notes editor popup
///
/// @deprecated Use `NotesManager` instead for a more complete widget experience
#[derive(Debug, Clone, Default)]
pub struct NotesEditorState {
    /// Whether the session notes editor is open
    pub session_notes_open: bool,
    /// Whether the global notes editor is open
    pub global_notes_open: bool,
    /// Whether the knowledge notes selector is open
    pub knowledge_notes_open: bool,
    /// Selected knowledge source for editing
    pub selected_knowledge_source: Option<String>,
    /// Temporary editing buffer for session notes
    pub session_notes_buffer: String,
    /// Temporary editing buffer for global notes
    pub global_notes_buffer: String,
    /// Temporary editing buffer for knowledge notes
    pub knowledge_notes_buffer: String,
}

impl NotesEditorState {
    /// Initialize session notes buffer from current value
    pub fn open_session_notes(&mut self, current_value: &str) {
        self.session_notes_buffer = current_value.to_string();
        self.session_notes_open = true;
    }

    /// Initialize global notes buffer from current value
    pub fn open_global_notes(&mut self, current_value: &str) {
        self.global_notes_buffer = current_value.to_string();
        self.global_notes_open = true;
    }

    /// Initialize knowledge notes buffer for a specific source
    pub fn open_knowledge_notes(&mut self, source: &str, current_value: &str) {
        self.selected_knowledge_source = Some(source.to_string());
        self.knowledge_notes_buffer = current_value.to_string();
        self.knowledge_notes_open = true;
    }
}

/// Response from notes editor widgets
#[derive(Debug, Clone, Default)]
pub struct NotesEditorResponse {
    /// Session notes changed - contains new value
    pub session_notes_changed: Option<String>,
    /// Global notes changed - contains new value
    pub global_notes_changed: Option<String>,
    /// Knowledge notes changed - contains (source, new_value)
    pub knowledge_notes_changed: Option<(String, String)>,
}

/// Configuration for notes editor widgets
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NotesEditorConfig {
    /// Label for session notes button
    pub session_button_label: String,
    /// Label for global notes button
    pub global_button_label: String,
    /// Label for knowledge notes button
    pub knowledge_button_label: String,
    /// Tooltip for session notes
    pub session_tooltip: String,
    /// Tooltip for global notes
    pub global_tooltip: String,
    /// Tooltip for knowledge notes
    pub knowledge_tooltip: String,
    /// Whether session notes are enabled
    pub session_notes_enabled: bool,
    /// Whether global notes are enabled
    pub global_notes_enabled: bool,
    /// Whether knowledge notes are enabled
    pub knowledge_notes_enabled: bool,
}

impl Default for NotesEditorConfig {
    fn default() -> Self {
        Self {
            session_button_label: "Session Notes".to_string(),
            global_button_label: "Global Notes".to_string(),
            knowledge_button_label: "Guide Notes".to_string(),
            session_tooltip: "Notes specific to this conversation. These help the AI remember important context.".to_string(),
            global_tooltip: "Notes that apply to all conversations. Use for persistent preferences and important info.".to_string(),
            knowledge_tooltip: "Notes for specific knowledge guides. Add clarifications or priorities for each guide.".to_string(),
            session_notes_enabled: true,
            global_notes_enabled: true,
            knowledge_notes_enabled: true,
        }
    }
}

/// Render notes control buttons
///
/// Returns which button was clicked (if any)
pub fn notes_buttons(
    ui: &mut Ui,
    session_notes: &str,
    global_notes: &str,
    config: &NotesEditorConfig,
) -> (bool, bool, bool) {
    let mut open_session = false;
    let mut open_global = false;
    let mut open_knowledge = false;

    ui.horizontal(|ui| {
        // Session notes button
        if config.session_notes_enabled {
            let has_notes = !session_notes.is_empty();
            let icon = if has_notes { "📝" } else { "📝" };
            let label = format!("{} {}", icon, config.session_button_label);

            let button = if has_notes {
                ui.button(RichText::new(&label).color(Color32::LIGHT_BLUE))
            } else {
                ui.button(&label)
            };

            if button.on_hover_text(&config.session_tooltip).clicked() {
                open_session = true;
            }
        }

        // Global notes button
        if config.global_notes_enabled {
            let has_notes = !global_notes.is_empty();
            let icon = if has_notes { "📋" } else { "📋" };
            let label = format!("{} {}", icon, config.global_button_label);

            let button = if has_notes {
                ui.button(RichText::new(&label).color(Color32::LIGHT_GREEN))
            } else {
                ui.button(&label)
            };

            if button.on_hover_text(&config.global_tooltip).clicked() {
                open_global = true;
            }
        }

        // Knowledge notes button
        if config.knowledge_notes_enabled {
            let label = format!("📚 {}", config.knowledge_button_label);
            if ui
                .button(&label)
                .on_hover_text(&config.knowledge_tooltip)
                .clicked()
            {
                open_knowledge = true;
            }
        }
    });

    (open_session, open_global, open_knowledge)
}

/// Render session notes editor popup
///
/// Returns the new notes value if saved, or None if cancelled/unchanged
pub fn session_notes_editor(
    ctx: &egui::Context,
    state: &mut NotesEditorState,
    title: &str,
) -> Option<String> {
    let mut result = None;

    if !state.session_notes_open {
        return None;
    }

    egui::Window::new(title)
        .collapsible(false)
        .resizable(true)
        .default_size([400.0, 300.0])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label(
                RichText::new("Add notes to help the AI understand context for this conversation:")
                    .size(11.0)
                    .color(Color32::GRAY),
            );
            ui.add_space(8.0);

            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut state.session_notes_buffer)
                            .desired_width(f32::INFINITY)
                            .desired_rows(8)
                            .hint_text(
                                "e.g., 'I prefer concise answers' or 'Focus on combat ships'",
                            ),
                    );
                });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    result = Some(state.session_notes_buffer.clone());
                    state.session_notes_open = false;
                }
                if ui.button("Cancel").clicked() {
                    state.session_notes_open = false;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Clear").clicked() {
                        state.session_notes_buffer.clear();
                    }
                });
            });
        });

    result
}

/// Render global notes editor popup
///
/// Returns the new notes value if saved, or None if cancelled/unchanged
pub fn global_notes_editor(
    ctx: &egui::Context,
    state: &mut NotesEditorState,
    title: &str,
) -> Option<String> {
    let mut result = None;

    if !state.global_notes_open {
        return None;
    }

    egui::Window::new(title)
        .collapsible(false)
        .resizable(true)
        .default_size([400.0, 300.0])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label(
                RichText::new("Global notes apply to ALL conversations:")
                    .size(11.0)
                    .color(Color32::GRAY)
            );
            ui.add_space(8.0);

            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut state.global_notes_buffer)
                            .desired_width(f32::INFINITY)
                            .desired_rows(8)
                            .hint_text("e.g., 'My budget is $500' or 'I own a Cutlass Black and Prospector'")
                    );
                });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    result = Some(state.global_notes_buffer.clone());
                    state.global_notes_open = false;
                }
                if ui.button("Cancel").clicked() {
                    state.global_notes_open = false;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Clear").clicked() {
                        state.global_notes_buffer.clear();
                    }
                });
            });
        });

    result
}

/// Knowledge source selector and notes editor
///
/// Returns (source, notes) if notes were changed for a source
pub fn knowledge_notes_editor(
    ctx: &egui::Context,
    state: &mut NotesEditorState,
    available_sources: &[String],
    mut get_notes: impl FnMut(&str) -> String,
    title: &str,
) -> Option<(String, String)> {
    let mut result = None;

    if !state.knowledge_notes_open {
        return None;
    }

    egui::Window::new(title)
        .collapsible(false)
        .resizable(true)
        .default_size([450.0, 350.0])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label(
                RichText::new("Add notes for specific knowledge guides:")
                    .size(11.0)
                    .color(Color32::GRAY),
            );
            ui.add_space(8.0);

            // Source selector
            ui.horizontal(|ui| {
                ui.label("Select guide:");

                let current_text = state
                    .selected_knowledge_source
                    .as_deref()
                    .unwrap_or("Choose a guide...");

                egui::ComboBox::from_id_source("knowledge_source_selector")
                    .selected_text(current_text)
                    .width(250.0)
                    .show_ui(ui, |ui| {
                        for source in available_sources {
                            let was_selected =
                                state.selected_knowledge_source.as_deref() == Some(source);
                            if ui.selectable_label(was_selected, source).clicked() {
                                // Load notes for this source
                                let notes = get_notes(source);
                                state.selected_knowledge_source = Some(source.clone());
                                state.knowledge_notes_buffer = notes;
                            }
                        }
                    });
            });

            ui.add_space(8.0);

            // Notes editor (only if source selected)
            if state.selected_knowledge_source.is_some() {
                egui::ScrollArea::vertical()
                    .max_height(180.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut state.knowledge_notes_buffer)
                                .desired_width(f32::INFINITY)
                                .desired_rows(6)
                                .hint_text(
                                    "Add notes about this guide (priorities, clarifications, etc.)",
                                ),
                        );
                    });
            } else {
                ui.label(
                    RichText::new("Select a guide from the dropdown above")
                        .color(Color32::GRAY)
                        .italics(),
                );
            }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_enabled_ui(state.selected_knowledge_source.is_some(), |ui| {
                    if ui.button("Save").clicked() {
                        if let Some(ref source) = state.selected_knowledge_source {
                            result = Some((source.clone(), state.knowledge_notes_buffer.clone()));
                        }
                        state.knowledge_notes_open = false;
                    }
                });
                if ui.button("Close").clicked() {
                    state.knowledge_notes_open = false;
                    state.selected_knowledge_source = None;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if state.selected_knowledge_source.is_some() {
                        if ui.small_button("Clear").clicked() {
                            state.knowledge_notes_buffer.clear();
                        }
                    }
                });
            });
        });

    result
}

// === Document Indexing Widgets ===

/// Widget showing the number of documents pending indexing
///
/// Returns true if the user clicked to start indexing
pub fn pending_documents_indicator(ui: &mut Ui, pending_count: usize, is_indexing: bool) -> bool {
    let mut start_indexing = false;

    if pending_count == 0 && !is_indexing {
        return false;
    }

    egui::Frame::none()
        .fill(Color32::from_rgb(50, 50, 60))
        .rounding(4.0)
        .inner_margin(6.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if is_indexing {
                    ui.spinner();
                    ui.label(
                        RichText::new("Indexing...")
                            .color(Color32::LIGHT_BLUE)
                            .size(11.0),
                    );
                } else if pending_count > 0 {
                    ui.label(
                        RichText::new(format!("📄 {} pending", pending_count))
                            .color(Color32::YELLOW)
                            .size(11.0),
                    );
                    if ui.small_button("Index Now").clicked() {
                        start_indexing = true;
                    }
                }
            });
        });

    start_indexing
}

/// Widget showing indexing progress
///
/// Call this each frame while `is_indexing` is true
pub fn indexing_progress(ui: &mut Ui, current_document: &str, current: usize, total: usize) {
    let progress = if total > 0 {
        current as f32 / total as f32
    } else {
        0.0
    };

    egui::Frame::none()
        .fill(Color32::from_rgb(40, 50, 60))
        .rounding(4.0)
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new(format!(
                        "Indexing {}/{}: {}",
                        current, total, current_document
                    ))
                    .color(Color32::LIGHT_BLUE)
                    .size(11.0),
                );
            });

            ui.add_space(4.0);

            // Progress bar
            let (rect, _response) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 6.0), egui::Sense::hover());

            if ui.is_rect_visible(rect) {
                let painter = ui.painter();
                painter.rect_filled(rect, 2.0, Color32::from_rgb(30, 30, 40));

                let fill_width = rect.width() * progress;
                if fill_width > 0.0 {
                    painter.rect_filled(
                        egui::Rect::from_min_size(rect.min, Vec2::new(fill_width, rect.height())),
                        2.0,
                        Color32::LIGHT_BLUE,
                    );
                }
            }
        });
}

/// Document statistics display
///
/// Shows a list of indexed documents with their stats
pub fn document_stats_list(ui: &mut Ui, stats: &[DocumentStatsDisplay], max_height: f32) {
    if stats.is_empty() {
        ui.label(
            RichText::new("No documents indexed")
                .color(Color32::GRAY)
                .size(11.0),
        );
        return;
    }

    egui::ScrollArea::vertical()
        .max_height(max_height)
        .show(ui, |ui| {
            egui::Grid::new("document_stats_grid")
                .num_columns(4)
                .spacing([10.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    // Header
                    ui.label(RichText::new("Document").size(11.0).strong());
                    ui.label(RichText::new("Chunks").size(11.0).strong());
                    ui.label(RichText::new("Tokens").size(11.0).strong());
                    ui.label(RichText::new("Status").size(11.0).strong());
                    ui.end_row();

                    // Rows
                    for stat in stats {
                        ui.label(RichText::new(&stat.source).size(10.0));
                        ui.label(RichText::new(stat.chunk_count.to_string()).size(10.0));
                        ui.label(RichText::new(stat.total_tokens.to_string()).size(10.0));

                        let status = if stat.is_pending {
                            RichText::new("Pending").color(Color32::YELLOW).size(10.0)
                        } else {
                            RichText::new("Indexed")
                                .color(Color32::LIGHT_GREEN)
                                .size(10.0)
                        };
                        ui.label(status);
                        ui.end_row();
                    }
                });
        });
}

/// Compact document stats for display in widgets
#[derive(Debug, Clone)]
pub struct DocumentStatsDisplay {
    /// Source name
    pub source: String,
    /// Number of chunks
    pub chunk_count: usize,
    /// Total tokens
    pub total_tokens: usize,
    /// Whether it's pending indexing
    pub is_pending: bool,
}

/// Document statistics panel with collapsible details
pub fn document_stats_panel(ui: &mut Ui, stats: &[DocumentStatsDisplay], title: &str) {
    let total_chunks: usize = stats.iter().map(|s| s.chunk_count).sum();
    let total_tokens: usize = stats.iter().map(|s| s.total_tokens).sum();
    let pending_count = stats.iter().filter(|s| s.is_pending).count();

    egui::CollapsingHeader::new(
        RichText::new(format!("{} ({} docs)", title, stats.len())).size(12.0),
    )
    .default_open(false)
    .show(ui, |ui| {
        // Summary
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "📊 {} chunks, {} tokens",
                    total_chunks, total_tokens
                ))
                .size(10.0),
            );
            if pending_count > 0 {
                ui.label(
                    RichText::new(format!("({} pending)", pending_count))
                        .color(Color32::YELLOW)
                        .size(10.0),
                );
            }
        });

        ui.add_space(4.0);

        // List
        document_stats_list(ui, stats, 150.0);
    });
}

/// Combined indexing status widget
///
/// Shows pending count, indexing progress, or nothing if all is indexed
pub fn indexing_status(
    ui: &mut Ui,
    pending_count: usize,
    is_indexing: bool,
    current_document: Option<&str>,
    current: usize,
    total: usize,
) -> bool {
    if is_indexing {
        if let Some(doc) = current_document {
            indexing_progress(ui, doc, current, total);
        } else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    RichText::new("Indexing documents...")
                        .color(Color32::LIGHT_BLUE)
                        .size(11.0),
                );
            });
        }
        false
    } else {
        pending_documents_indicator(ui, pending_count, is_indexing)
    }
}

/// Export/Import buttons for knowledge base
///
/// Returns (export_clicked, import_clicked)
pub fn knowledge_export_import_buttons(ui: &mut Ui, has_knowledge: bool) -> (bool, bool) {
    let mut export = false;
    let mut import = false;

    ui.horizontal(|ui| {
        ui.add_enabled_ui(has_knowledge, |ui| {
            if ui
                .button("📤 Export Knowledge")
                .on_hover_text("Export knowledge base to a file")
                .clicked()
            {
                export = true;
            }
        });

        if ui
            .button("📥 Import Knowledge")
            .on_hover_text("Import knowledge from a file")
            .clicked()
        {
            import = true;
        }
    });

    (export, import)
}

// === Knowledge Search Widget ===

/// State for standalone knowledge search widget
#[derive(Debug, Clone, Default)]
pub struct KnowledgeSearchState {
    /// Current search query
    pub query: String,
    /// Search results
    pub results: Vec<SearchResultDisplay>,
    /// Whether a search is in progress
    pub is_searching: bool,
    /// Total tokens in results
    pub total_tokens: usize,
}

/// Display format for a search result
#[derive(Debug, Clone)]
pub struct SearchResultDisplay {
    /// Source document name
    pub source: String,
    /// Section within document
    pub section: String,
    /// Content snippet (may be truncated)
    pub content: String,
    /// Token count
    pub token_count: usize,
    /// Relevance score (0.0 to 1.0)
    pub relevance: f32,
}

/// Response from knowledge search widget
#[derive(Debug, Clone, Default)]
pub struct KnowledgeSearchResponse {
    /// User triggered a search
    pub search_triggered: bool,
    /// Query to search for
    pub query: String,
    /// User clicked on a result (index)
    pub result_clicked: Option<usize>,
    /// User wants to use result in chat
    pub use_in_chat: Option<usize>,
}

/// Render a standalone knowledge search widget
pub fn knowledge_search(
    ui: &mut Ui,
    state: &mut KnowledgeSearchState,
    max_height: f32,
) -> KnowledgeSearchResponse {
    let mut response = KnowledgeSearchResponse::default();

    // Search input
    ui.horizontal(|ui| {
        let text_response = ui.add(
            egui::TextEdit::singleline(&mut state.query)
                .hint_text("Search knowledge base...")
                .desired_width(ui.available_width() - 70.0),
        );

        let search_enabled = !state.query.trim().is_empty() && !state.is_searching;

        if ui
            .add_enabled(search_enabled, egui::Button::new("🔍 Search"))
            .clicked()
            || (text_response.lost_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Enter))
                && search_enabled)
        {
            response.search_triggered = true;
            response.query = state.query.trim().to_string();
        }

        if state.is_searching {
            ui.spinner();
        }
    });

    // Results summary
    if !state.results.is_empty() {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "Found {} results ({} tokens)",
                    state.results.len(),
                    state.total_tokens
                ))
                .size(10.0)
                .color(Color32::GRAY),
            );
        });
    }

    // Results list
    if !state.results.is_empty() {
        ui.add_space(4.0);
        egui::ScrollArea::vertical()
            .max_height(max_height)
            .show(ui, |ui| {
                for (idx, result) in state.results.iter().enumerate() {
                    egui::Frame::none()
                        .fill(Color32::from_rgb(40, 45, 55))
                        .rounding(4.0)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            // Header with source and section
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(&result.source)
                                        .size(11.0)
                                        .color(Color32::LIGHT_BLUE)
                                        .strong(),
                                );
                                if !result.section.is_empty() {
                                    ui.label(
                                        RichText::new(format!("› {}", result.section))
                                            .size(10.0)
                                            .color(Color32::GRAY),
                                    );
                                }
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            RichText::new(format!("{} tokens", result.token_count))
                                                .size(9.0)
                                                .color(Color32::DARK_GRAY),
                                        );
                                    },
                                );
                            });

                            // Content snippet
                            ui.add_space(2.0);
                            let content_preview: String =
                                result.content.chars().take(200).collect();
                            let display_text = if result.content.len() > 200 {
                                format!("{}...", content_preview)
                            } else {
                                content_preview
                            };

                            if ui
                                .add(
                                    egui::Label::new(RichText::new(&display_text).size(10.0))
                                        .sense(egui::Sense::click()),
                                )
                                .clicked()
                            {
                                response.result_clicked = Some(idx);
                            }

                            // Actions
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                if ui.small_button("📋 Copy").clicked() {
                                    ui.ctx().copy_text(result.content.clone());
                                }
                                if ui.small_button("💬 Use in chat").clicked() {
                                    response.use_in_chat = Some(idx);
                                }
                            });
                        });
                    ui.add_space(4.0);
                }
            });
    }

    response
}

// === Retrieved Chunks Visualization ===

/// Display information about retrieved chunks used in a response
#[derive(Debug, Clone)]
pub struct RetrievedChunkInfo {
    /// Source document
    pub source: String,
    /// Section
    pub section: String,
    /// Token count
    pub tokens: usize,
    /// Whether it was actually used (fit in context)
    pub used: bool,
}

/// Render a collapsible panel showing retrieved chunks
pub fn retrieved_chunks_panel(ui: &mut Ui, chunks: &[RetrievedChunkInfo], title: &str) {
    if chunks.is_empty() {
        return;
    }

    let used_count = chunks.iter().filter(|c| c.used).count();
    let total_tokens: usize = chunks.iter().filter(|c| c.used).map(|c| c.tokens).sum();

    egui::CollapsingHeader::new(
        RichText::new(format!(
            "{} ({}/{} chunks, {} tokens)",
            title,
            used_count,
            chunks.len(),
            total_tokens
        ))
        .size(11.0),
    )
    .default_open(false)
    .show(ui, |ui| {
        for chunk in chunks {
            ui.horizontal(|ui| {
                let icon = if chunk.used { "✓" } else { "○" };
                let color = if chunk.used {
                    Color32::LIGHT_GREEN
                } else {
                    Color32::GRAY
                };

                ui.label(RichText::new(icon).color(color).size(10.0));
                ui.label(
                    RichText::new(&chunk.source)
                        .size(10.0)
                        .color(if chunk.used {
                            Color32::LIGHT_BLUE
                        } else {
                            Color32::DARK_GRAY
                        }),
                );
                if !chunk.section.is_empty() {
                    ui.label(
                        RichText::new(format!("› {}", chunk.section))
                            .size(9.0)
                            .color(Color32::DARK_GRAY),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("{} tok", chunk.tokens))
                            .size(9.0)
                            .color(Color32::DARK_GRAY),
                    );
                });
            });
        }
    });
}

// === LLM Provider Configuration Widget ===

/// Configuration for an LLM provider
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ProviderConfig {
    /// Provider name (Ollama, LM Studio, etc.)
    pub name: String,
    /// Base URL
    pub url: String,
    /// Whether this provider is enabled
    pub enabled: bool,
    /// Connection status
    pub status: ProviderStatus,
    /// Number of models found
    pub model_count: usize,
}

/// Connection status for a provider
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ProviderStatus {
    Unknown,
    Checking,
    Connected,
    Error(String),
}

impl Default for ProviderStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Response from provider configuration widget
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ProviderConfigResponse {
    /// Provider URL was changed (provider_name, new_url)
    pub url_changed: Option<(String, String)>,
    /// Provider was toggled (provider_name, enabled)
    pub toggled: Option<(String, bool)>,
    /// Test connection requested for provider
    pub test_connection: Option<String>,
    /// Refresh models requested for provider
    pub refresh_models: Option<String>,
}

/// Render LLM provider configuration panel
pub fn provider_config_panel(
    ui: &mut Ui,
    providers: &mut [ProviderConfig],
) -> ProviderConfigResponse {
    let mut response = ProviderConfigResponse::default();

    for provider in providers.iter_mut() {
        egui::CollapsingHeader::new(RichText::new(&provider.name).size(12.0))
            .default_open(provider.enabled)
            .show(ui, |ui| {
                // Enabled toggle
                ui.horizontal(|ui| {
                    let mut enabled = provider.enabled;
                    if ui.checkbox(&mut enabled, "Enabled").changed() {
                        response.toggled = Some((provider.name.clone(), enabled));
                        provider.enabled = enabled;
                    }

                    // Status indicator
                    match &provider.status {
                        ProviderStatus::Unknown => {
                            ui.label(RichText::new("⚪").color(Color32::GRAY));
                        }
                        ProviderStatus::Checking => {
                            ui.spinner();
                        }
                        ProviderStatus::Connected => {
                            ui.label(
                                RichText::new(format!("🟢 {} models", provider.model_count))
                                    .color(Color32::LIGHT_GREEN)
                                    .size(10.0),
                            );
                        }
                        ProviderStatus::Error(msg) => {
                            ui.label(
                                RichText::new(format!("🔴 {}", msg))
                                    .color(Color32::from_rgb(255, 100, 100))
                                    .size(10.0),
                            );
                        }
                    }
                });

                // URL configuration
                ui.horizontal(|ui| {
                    ui.label(RichText::new("URL:").size(10.0));
                    let url_response = ui.add(
                        egui::TextEdit::singleline(&mut provider.url)
                            .desired_width(200.0)
                            .hint_text("http://localhost:11434"),
                    );
                    if url_response.lost_focus() {
                        response.url_changed = Some((provider.name.clone(), provider.url.clone()));
                    }
                });

                // Actions
                ui.horizontal(|ui| {
                    if ui.small_button("Test Connection").clicked() {
                        response.test_connection = Some(provider.name.clone());
                    }
                    if ui.small_button("Refresh Models").clicked() {
                        response.refresh_models = Some(provider.name.clone());
                    }
                });
            });
    }

    response
}

// === Metrics Display Widgets ===

use crate::metrics::{MessageMetrics, RagQualityMetrics, SessionMetrics};

/// Render compact session metrics
pub fn session_metrics_compact(ui: &mut Ui, metrics: &SessionMetrics) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("📊 {} msgs", metrics.message_count))
                .size(10.0)
                .color(Color32::LIGHT_GRAY),
        );
        ui.label(
            RichText::new(format!("⏱ {:.0}ms avg", metrics.avg_response_time_ms))
                .size(10.0)
                .color(Color32::LIGHT_GRAY),
        );
        ui.label(
            RichText::new(format!(
                "🔤 {} in / {} out",
                metrics.total_input_tokens, metrics.total_output_tokens
            ))
            .size(10.0)
            .color(Color32::LIGHT_GRAY),
        );
    });
}

/// Render detailed session metrics panel
pub fn session_metrics_panel(ui: &mut Ui, metrics: &SessionMetrics) {
    egui::CollapsingHeader::new(
        RichText::new(format!("Session Metrics ({})", metrics.session_id)).size(12.0),
    )
    .default_open(false)
    .show(ui, |ui| {
        egui::Grid::new("session_metrics_grid")
            .num_columns(2)
            .spacing([20.0, 4.0])
            .show(ui, |ui| {
                ui.label("Messages:");
                ui.label(RichText::new(metrics.message_count.to_string()).strong());
                ui.end_row();

                ui.label("Input tokens:");
                ui.label(metrics.total_input_tokens.to_string());
                ui.end_row();

                ui.label("Output tokens:");
                ui.label(metrics.total_output_tokens.to_string());
                ui.end_row();

                ui.label("Avg response time:");
                ui.label(format!("{:.0} ms", metrics.avg_response_time_ms));
                ui.end_row();

                ui.label("Avg time to first token:");
                ui.label(format!("{:.0} ms", metrics.avg_time_to_first_token_ms));
                ui.end_row();

                ui.label("Knowledge chunks retrieved:");
                ui.label(metrics.total_knowledge_chunks.to_string());
                ui.end_row();

                ui.label("Context limit warnings:");
                ui.label(
                    RichText::new(metrics.context_limit_warnings.to_string()).color(
                        if metrics.context_limit_warnings > 0 {
                            Color32::YELLOW
                        } else {
                            Color32::WHITE
                        },
                    ),
                );
                ui.end_row();

                ui.label("Session duration:");
                ui.label(format!("{} seconds", metrics.session_duration_secs));
                ui.end_row();
            });
    });
}

/// Render RAG quality metrics
pub fn rag_quality_metrics_panel(ui: &mut Ui, metrics: &RagQualityMetrics) {
    egui::CollapsingHeader::new(RichText::new("RAG Quality Metrics").size(12.0))
        .default_open(false)
        .show(ui, |ui| {
            egui::Grid::new("rag_quality_grid")
                .num_columns(2)
                .spacing([20.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Total queries:");
                    ui.label(metrics.total_queries.to_string());
                    ui.end_row();

                    ui.label("Queries with results:");
                    ui.label(format!(
                        "{} ({:.0}%)",
                        metrics.queries_with_results,
                        if metrics.total_queries > 0 {
                            metrics.queries_with_results as f64 / metrics.total_queries as f64
                                * 100.0
                        } else {
                            0.0
                        }
                    ));
                    ui.end_row();

                    ui.label("Avg chunks per query:");
                    ui.label(format!("{:.1}", metrics.avg_chunks_per_query));
                    ui.end_row();

                    ui.label("Avg tokens per query:");
                    ui.label(format!("{:.0}", metrics.avg_tokens_per_query));
                    ui.end_row();

                    ui.label("Cache hit rate:");
                    ui.label(format!("{:.0}%", metrics.cache_hit_rate * 100.0));
                    ui.end_row();
                });

            if !metrics.top_sources.is_empty() {
                ui.add_space(8.0);
                ui.label(RichText::new("Top accessed sources:").size(11.0).strong());
                for (source, count) in &metrics.top_sources {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("  {} ({})", source, count)).size(10.0));
                    });
                }
            }
        });
}

/// Render a single message metrics display
pub fn message_metrics_inline(ui: &mut Ui, metrics: &MessageMetrics) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{}ms", metrics.total_response_time_ms))
                .size(9.0)
                .color(Color32::DARK_GRAY),
        );
        if let Some(ttft) = metrics.time_to_first_token_ms {
            ui.label(
                RichText::new(format!("(TTFT: {}ms)", ttft))
                    .size(9.0)
                    .color(Color32::DARK_GRAY),
            );
        }
        ui.label(
            RichText::new(format!("{} tok", metrics.output_tokens))
                .size(9.0)
                .color(Color32::DARK_GRAY),
        );
        if metrics.knowledge_chunks_retrieved > 0 {
            ui.label(
                RichText::new(format!("📚{}", metrics.knowledge_chunks_retrieved))
                    .size(9.0)
                    .color(Color32::LIGHT_BLUE),
            );
        }
    });
}

// === Test Results Widget ===

use crate::metrics::{TestCaseResult, TestSuiteResults};

/// Render test suite results panel
pub fn test_results_panel(ui: &mut Ui, results: &TestSuiteResults) {
    let pass_color = if results.pass_rate >= 0.8 {
        Color32::LIGHT_GREEN
    } else if results.pass_rate >= 0.5 {
        Color32::YELLOW
    } else {
        Color32::from_rgb(255, 100, 100)
    };

    egui::CollapsingHeader::new(
        RichText::new(format!(
            "Test Results: {} - {}/{} passed ({:.0}%)",
            results.suite_name,
            results.passed,
            results.total_tests,
            results.pass_rate * 100.0
        ))
        .size(12.0)
        .color(pass_color),
    )
    .default_open(results.failed > 0)
    .show(ui, |ui| {
        ui.label(
            RichText::new(format!(
                "Avg response time: {:.0}ms",
                results.avg_response_time_ms
            ))
            .size(10.0)
            .color(Color32::GRAY),
        );

        ui.add_space(4.0);

        for result in &results.results {
            test_case_result_row(ui, result);
        }
    });
}

fn test_case_result_row(ui: &mut Ui, result: &TestCaseResult) {
    let (icon, color) = if result.passed {
        ("✓", Color32::LIGHT_GREEN)
    } else {
        ("✗", Color32::from_rgb(255, 100, 100))
    };

    egui::CollapsingHeader::new(
        RichText::new(format!("{} {}", icon, result.name))
            .size(11.0)
            .color(color),
    )
    .default_open(!result.passed)
    .show(ui, |ui| {
        // Metrics
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "{}ms | {} tokens",
                    result.metrics.total_response_time_ms, result.metrics.output_tokens
                ))
                .size(9.0)
                .color(Color32::GRAY),
            );
        });

        // Failure reasons
        if !result.failure_reasons.is_empty() {
            ui.add_space(4.0);
            for reason in &result.failure_reasons {
                ui.label(
                    RichText::new(format!("  ⚠ {}", reason))
                        .size(10.0)
                        .color(Color32::YELLOW),
                );
            }
        }

        // Response preview
        ui.add_space(4.0);
        let preview: String = result.response.chars().take(150).collect();
        ui.label(
            RichText::new(if result.response.len() > 150 {
                format!("{}...", preview)
            } else {
                preview
            })
            .size(9.0)
            .color(Color32::DARK_GRAY)
            .italics(),
        );
    });
}

// === Sentiment Analysis Widget ===

use crate::analysis::{Sentiment, SentimentAnalysis, SessionSummary};

/// Display sentiment indicator badge
pub fn sentiment_badge(ui: &mut Ui, sentiment: &Sentiment) {
    let (icon, color, label) = match sentiment {
        Sentiment::VeryPositive => ("😄", Color32::from_rgb(100, 220, 100), "Very Positive"),
        Sentiment::Positive => ("🙂", Color32::from_rgb(150, 200, 100), "Positive"),
        Sentiment::Neutral => ("😐", Color32::GRAY, "Neutral"),
        Sentiment::Negative => ("🙁", Color32::from_rgb(220, 150, 100), "Negative"),
        Sentiment::VeryNegative => ("😞", Color32::from_rgb(220, 100, 100), "Very Negative"),
    };

    ui.horizontal(|ui| {
        ui.label(RichText::new(icon).size(14.0));
        ui.label(RichText::new(label).size(10.0).color(color));
    });
}

/// Display full sentiment analysis with score and indicators
pub fn sentiment_analysis_panel(ui: &mut Ui, analysis: &SentimentAnalysis) {
    egui::CollapsingHeader::new(
        RichText::new("📊 Sentiment Analysis")
            .size(12.0)
            .color(Color32::WHITE),
    )
    .default_open(false)
    .show(ui, |ui| {
        ui.horizontal(|ui| {
            sentiment_badge(ui, &analysis.sentiment);
            ui.add_space(10.0);
            ui.label(
                RichText::new(format!("Score: {:.2}", analysis.score))
                    .size(10.0)
                    .color(Color32::GRAY),
            );
            ui.label(
                RichText::new(format!("Confidence: {:.0}%", analysis.confidence * 100.0))
                    .size(10.0)
                    .color(Color32::GRAY),
            );
        });

        // Show positive indicators
        if !analysis.positive_indicators.is_empty() {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("Positive:")
                        .size(9.0)
                        .color(Color32::LIGHT_GREEN),
                );
                for indicator in &analysis.positive_indicators {
                    egui::Frame::none()
                        .fill(Color32::from_rgb(40, 60, 40))
                        .rounding(4.0)
                        .inner_margin(egui::Margin::symmetric(4.0, 2.0))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(indicator)
                                    .size(9.0)
                                    .color(Color32::LIGHT_GREEN),
                            );
                        });
                }
            });
        }

        // Show negative indicators
        if !analysis.negative_indicators.is_empty() {
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("Negative:")
                        .size(9.0)
                        .color(Color32::from_rgb(255, 150, 150)),
                );
                for indicator in &analysis.negative_indicators {
                    egui::Frame::none()
                        .fill(Color32::from_rgb(60, 40, 40))
                        .rounding(4.0)
                        .inner_margin(egui::Margin::symmetric(4.0, 2.0))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(indicator)
                                    .size(9.0)
                                    .color(Color32::from_rgb(255, 150, 150)),
                            );
                        });
                }
            });
        }
    });
}

// === Topic Detection Widget ===

use crate::analysis::Topic;

/// Display detected topics as tags
pub fn topics_panel(ui: &mut Ui, topics: &[Topic]) {
    if topics.is_empty() {
        return;
    }

    egui::CollapsingHeader::new(RichText::new("🏷️ Topics").size(12.0).color(Color32::WHITE))
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for topic in topics {
                    let color = topic_color(&topic.name);
                    egui::Frame::none()
                        .fill(color)
                        .rounding(4.0)
                        .inner_margin(egui::Margin::symmetric(6.0, 3.0))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(format!(
                                    "{} ({:.0}%)",
                                    topic.name,
                                    topic.relevance * 100.0
                                ))
                                .size(10.0)
                                .color(Color32::WHITE),
                            );
                        });
                }
            });
        });
}

fn topic_color(topic: &str) -> Color32 {
    // Generate consistent color from topic name
    let hash: u32 = topic
        .bytes()
        .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
    Color32::from_rgb(
        ((hash >> 16) & 0xFF) as u8 / 2 + 40,
        ((hash >> 8) & 0xFF) as u8 / 2 + 40,
        (hash & 0xFF) as u8 / 2 + 40,
    )
}

// === Session Summary Widget ===

/// Display session summary
pub fn session_summary_panel(ui: &mut Ui, summary: &SessionSummary) {
    egui::CollapsingHeader::new(
        RichText::new("📋 Session Summary")
            .size(12.0)
            .color(Color32::WHITE),
    )
    .default_open(true)
    .show(ui, |ui| {
        // Stats
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("Messages: {}", summary.message_count))
                    .size(10.0)
                    .color(Color32::GRAY),
            );
            ui.separator();
            ui.label(
                RichText::new(format!("Sentiment: {}", summary.sentiment))
                    .size(10.0)
                    .color(Color32::GRAY),
            );
        });

        // Main topics
        if !summary.topics.is_empty() {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Topics:").size(9.0).color(Color32::GRAY));
                for topic in &summary.topics {
                    ui.label(RichText::new(format!("• {}", topic)).size(9.0));
                }
            });
        }

        // User questions
        if !summary.user_questions.is_empty() {
            ui.add_space(4.0);
            ui.label(
                RichText::new("Questions asked:")
                    .size(10.0)
                    .color(Color32::LIGHT_BLUE),
            );
            for question in summary.user_questions.iter().take(3) {
                let preview: String = question.chars().take(60).collect();
                ui.label(
                    RichText::new(format!("  • {}...", preview))
                        .size(9.0)
                        .color(Color32::LIGHT_GRAY),
                );
            }
        }

        // Key points
        if !summary.key_points.is_empty() {
            ui.add_space(4.0);
            ui.label(
                RichText::new("Key Points:")
                    .size(10.0)
                    .color(Color32::WHITE),
            );
            for point in &summary.key_points {
                ui.label(
                    RichText::new(format!("  • {}", point))
                        .size(9.0)
                        .color(Color32::LIGHT_GRAY),
                );
            }
        }

        // Generated summary
        ui.add_space(4.0);
        ui.label(
            RichText::new(&summary.summary)
                .size(10.0)
                .color(Color32::WHITE)
                .italics(),
        );
    });
}

// === Conversation Branching Widget ===

use crate::conversation_control::{BranchPoint, CancellationToken};

/// Branch selector widget - allows switching between conversation branches
pub fn branch_selector(
    ui: &mut Ui,
    branches: &[BranchPoint],
    active_branch: Option<&str>,
) -> Option<String> {
    let mut selected: Option<String> = None;

    if branches.is_empty() {
        return None;
    }

    egui::CollapsingHeader::new(
        RichText::new(format!("🌿 Branches ({})", branches.len()))
            .size(12.0)
            .color(Color32::WHITE),
    )
    .default_open(false)
    .show(ui, |ui| {
        for branch in branches {
            let is_active = active_branch == Some(&branch.id);
            let color = if is_active {
                Color32::LIGHT_GREEN
            } else {
                Color32::GRAY
            };

            ui.horizontal(|ui| {
                if ui
                    .selectable_label(is_active, RichText::new(&branch.name).color(color))
                    .clicked()
                {
                    selected = Some(branch.id.clone());
                }
                ui.label(
                    RichText::new(format!("@ msg {}", branch.branch_index))
                        .size(9.0)
                        .color(Color32::DARK_GRAY),
                );
            });
        }
    });

    selected
}

/// Display cancel button for streaming responses
pub fn cancel_button(ui: &mut Ui, token: &CancellationToken) -> bool {
    let clicked = ui
        .add(
            egui::Button::new(RichText::new("⏹ Cancel").size(11.0))
                .fill(Color32::from_rgb(100, 50, 50)),
        )
        .clicked();

    if clicked {
        token.cancel();
    }

    clicked
}

// === Rate Limiting Widget ===

use crate::security::{AuditEvent, AuditEventType, RateLimiter};

/// Display rate limit status
pub fn rate_limit_status(ui: &mut Ui, limiter: &RateLimiter) {
    let status = limiter.get_status();

    let color = if status.requests_remaining < 5 {
        Color32::from_rgb(255, 150, 100)
    } else if status.requests_remaining < 10 {
        Color32::YELLOW
    } else {
        Color32::LIGHT_GREEN
    };

    ui.horizontal(|ui| {
        ui.label(RichText::new("🚦").size(12.0));
        ui.label(
            RichText::new(format!(
                "{}/{} requests",
                status.requests_remaining, status.requests_per_minute
            ))
            .size(10.0)
            .color(color),
        );

        if status.tokens_remaining < status.tokens_per_minute {
            ui.separator();
            ui.label(
                RichText::new(format!(
                    "{}/{} tokens",
                    status.tokens_remaining, status.tokens_per_minute
                ))
                .size(10.0)
                .color(Color32::GRAY),
            );
        }

        if let Some(cooldown) = status.cooldown_remaining {
            ui.separator();
            ui.label(
                RichText::new(format!("⏳ {}s", cooldown))
                    .size(10.0)
                    .color(Color32::from_rgb(255, 100, 100)),
            );
        }
    });
}

// === Audit Log Widget ===

/// Display recent audit events
pub fn audit_log_panel(ui: &mut Ui, events: &[AuditEvent], max_events: usize) {
    egui::CollapsingHeader::new(
        RichText::new("📝 Audit Log")
            .size(12.0)
            .color(Color32::WHITE),
    )
    .default_open(false)
    .show(ui, |ui| {
        let display_events: Vec<_> = events.iter().rev().take(max_events).collect();

        for event in display_events {
            let (icon, color) = audit_event_style(&event.event_type);

            ui.horizontal(|ui| {
                ui.label(RichText::new(icon).size(10.0));
                ui.label(
                    RichText::new(event.timestamp.format("%H:%M:%S").to_string())
                        .size(9.0)
                        .color(Color32::DARK_GRAY),
                );
                ui.label(
                    RichText::new(format!("{:?}", event.event_type))
                        .size(9.0)
                        .color(color),
                );
                if !event.details.is_empty() {
                    let details_str: String = event
                        .details
                        .iter()
                        .take(2)
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.label(RichText::new(details_str).size(9.0).color(Color32::GRAY));
                }
            });
        }
    });
}

fn audit_event_style(event_type: &AuditEventType) -> (&'static str, Color32) {
    match event_type {
        AuditEventType::MessageSent => ("📤", Color32::LIGHT_BLUE),
        AuditEventType::ResponseReceived => ("📥", Color32::LIGHT_GREEN),
        AuditEventType::ResponseCancelled => ("⏹", Color32::YELLOW),
        AuditEventType::ResponseRegenerated => ("🔄", Color32::LIGHT_BLUE),
        AuditEventType::MessageEdited => ("✏", Color32::YELLOW),
        AuditEventType::SessionCreated => ("🆕", Color32::WHITE),
        AuditEventType::SessionLoaded => ("📂", Color32::WHITE),
        AuditEventType::SessionDeleted => ("🗑", Color32::YELLOW),
        AuditEventType::DocumentIndexed => ("📚", Color32::LIGHT_BLUE),
        AuditEventType::DocumentDeleted => ("📚", Color32::YELLOW),
        AuditEventType::RateLimitHit => ("🚫", Color32::from_rgb(255, 150, 100)),
        AuditEventType::InputSanitized => ("🧹", Color32::GRAY),
        AuditEventType::ConfigChanged => ("⚙", Color32::GRAY),
        AuditEventType::Error => ("❌", Color32::from_rgb(255, 100, 100)),
    }
}

// === Backup Status Widget ===

use crate::persistence::BackupInfo;

/// Display backup status and controls
pub fn backup_status_panel(
    ui: &mut Ui,
    last_backup: Option<&BackupInfo>,
    auto_backup_enabled: bool,
) -> BackupAction {
    let mut action = BackupAction::None;

    egui::CollapsingHeader::new(RichText::new("💾 Backup").size(12.0).color(Color32::WHITE))
        .default_open(false)
        .show(ui, |ui| {
            if let Some(backup) = last_backup {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Last backup:")
                            .size(10.0)
                            .color(Color32::GRAY),
                    );
                    ui.label(
                        RichText::new(backup.created_at.format("%Y-%m-%d %H:%M").to_string())
                            .size(10.0)
                            .color(Color32::WHITE),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Size:").size(9.0).color(Color32::GRAY));
                    ui.label(
                        RichText::new(format_bytes(backup.size_bytes))
                            .size(9.0)
                            .color(Color32::GRAY),
                    );
                });
            } else {
                ui.label(
                    RichText::new("No backups yet")
                        .size(10.0)
                        .color(Color32::GRAY)
                        .italics(),
                );
            }

            ui.add_space(4.0);

            ui.horizontal(|ui| {
                if ui
                    .button(RichText::new("📦 Backup Now").size(10.0))
                    .clicked()
                {
                    action = BackupAction::CreateBackup;
                }
                if last_backup.is_some()
                    && ui.button(RichText::new("🔄 Restore").size(10.0)).clicked()
                {
                    action = BackupAction::RestoreBackup;
                }
            });

            ui.horizontal(|ui| {
                let mut enabled = auto_backup_enabled;
                if ui
                    .checkbox(&mut enabled, RichText::new("Auto-backup").size(10.0))
                    .changed()
                {
                    action = if enabled {
                        BackupAction::EnableAutoBackup
                    } else {
                        BackupAction::DisableAutoBackup
                    };
                }
            });
        });

    action
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackupAction {
    None,
    CreateBackup,
    RestoreBackup,
    EnableAutoBackup,
    DisableAutoBackup,
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

// === Tool Calling Widget ===

use crate::tools::{ToolCall, ToolDefinition, ToolResult};

/// Display available tools
pub fn tools_panel(ui: &mut Ui, tools: &[ToolDefinition]) {
    egui::CollapsingHeader::new(
        RichText::new(format!("🔧 Tools ({})", tools.len()))
            .size(12.0)
            .color(Color32::WHITE),
    )
    .default_open(false)
    .show(ui, |ui| {
        for tool in tools {
            egui::CollapsingHeader::new(
                RichText::new(&tool.name)
                    .size(11.0)
                    .color(Color32::LIGHT_BLUE),
            )
            .default_open(false)
            .show(ui, |ui| {
                ui.label(
                    RichText::new(&tool.description)
                        .size(9.0)
                        .color(Color32::GRAY),
                );

                if !tool.parameters.is_empty() {
                    ui.add_space(4.0);
                    ui.label(RichText::new("Parameters:").size(9.0).color(Color32::WHITE));
                    for param in &tool.parameters {
                        ui.horizontal(|ui| {
                            let required_mark = if param.required { "*" } else { "" };
                            ui.label(
                                RichText::new(format!("  {}{}", param.name, required_mark))
                                    .size(9.0)
                                    .color(Color32::LIGHT_GREEN),
                            );
                            ui.label(
                                RichText::new(format!("({:?})", param.param_type))
                                    .size(9.0)
                                    .color(Color32::GRAY),
                            );
                        });
                        if !param.description.is_empty() {
                            ui.label(
                                RichText::new(format!("    {}", param.description))
                                    .size(8.0)
                                    .color(Color32::DARK_GRAY),
                            );
                        }
                    }
                }
            });
        }
    });
}

/// Display a tool call in the chat
pub fn tool_call_bubble(ui: &mut Ui, call: &ToolCall, result: Option<&ToolResult>) {
    egui::Frame::none()
        .fill(Color32::from_rgb(40, 50, 60))
        .rounding(6.0)
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("🔧").size(12.0));
                ui.label(
                    RichText::new(&call.name)
                        .size(11.0)
                        .color(Color32::LIGHT_BLUE),
                );
            });

            // Show arguments
            if !call.arguments.is_empty() {
                ui.add_space(2.0);
                for (key, value) in &call.arguments {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("  {}: ", key))
                                .size(9.0)
                                .color(Color32::GRAY),
                        );
                        ui.label(
                            RichText::new(value.to_string())
                                .size(9.0)
                                .color(Color32::WHITE),
                        );
                    });
                }
            }

            // Show result if available
            if let Some(res) = result {
                ui.add_space(4.0);
                let (icon, color) = if res.success {
                    ("✓", Color32::LIGHT_GREEN)
                } else {
                    ("✗", Color32::from_rgb(255, 100, 100))
                };

                ui.horizontal(|ui| {
                    ui.label(RichText::new(icon).size(10.0).color(color));
                    let content_preview: String = res.content.chars().take(100).collect();
                    ui.label(
                        RichText::new(if res.content.len() > 100 {
                            format!("{}...", content_preview)
                        } else {
                            content_preview
                        })
                        .size(9.0)
                        .color(Color32::GRAY),
                    );
                });
            }
        });
}

// === Hybrid Search Widget ===

use crate::embeddings::HybridSearchResult;

/// Display hybrid search results
pub fn hybrid_search_results(ui: &mut Ui, results: &[HybridSearchResult]) {
    if results.is_empty() {
        ui.label(
            RichText::new("No results found")
                .size(10.0)
                .color(Color32::GRAY)
                .italics(),
        );
        return;
    }

    for (i, result) in results.iter().enumerate() {
        egui::Frame::none()
            .fill(Color32::from_rgb(40, 45, 55))
            .rounding(4.0)
            .inner_margin(6.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("#{}", i + 1))
                            .size(10.0)
                            .color(Color32::GRAY),
                    );
                    ui.label(
                        RichText::new(format!("Score: {:.3}", result.combined_score))
                            .size(9.0)
                            .color(Color32::LIGHT_GREEN),
                    );

                    // Show individual scores
                    if result.keyword_score > 0.0 {
                        ui.label(
                            RichText::new(format!("KW: {:.2}", result.keyword_score))
                                .size(8.0)
                                .color(Color32::DARK_GRAY),
                        );
                    }
                    if result.semantic_score > 0.0 {
                        ui.label(
                            RichText::new(format!("Sem: {:.2}", result.semantic_score))
                                .size(8.0)
                                .color(Color32::DARK_GRAY),
                        );
                    }
                });

                ui.add_space(2.0);

                // Content preview
                let preview: String = result.content.chars().take(200).collect();
                ui.label(
                    RichText::new(if result.content.len() > 200 {
                        format!("{}...", preview)
                    } else {
                        preview
                    })
                    .size(10.0)
                    .color(Color32::WHITE),
                );
            });

        ui.add_space(4.0);
    }
}

// === Response Variants Widget ===

use crate::conversation_control::ResponseVariant;

/// Display response variants selector
pub fn variant_selector(
    ui: &mut Ui,
    variants: &[ResponseVariant],
    active_index: usize,
) -> Option<usize> {
    let mut selected: Option<usize> = None;

    if variants.len() <= 1 {
        return None;
    }

    ui.horizontal(|ui| {
        ui.label(RichText::new("Variants:").size(9.0).color(Color32::GRAY));

        for (i, _variant) in variants.iter().enumerate() {
            let is_active = i == active_index;
            let color = if is_active {
                Color32::LIGHT_BLUE
            } else {
                Color32::GRAY
            };

            if ui
                .selectable_label(
                    is_active,
                    RichText::new((i + 1).to_string()).size(10.0).color(color),
                )
                .clicked()
                && !is_active
            {
                selected = Some(i);
            }
        }

        ui.separator();
        ui.label(
            RichText::new(format!("T={:.1}", variants[active_index].temperature))
                .size(8.0)
                .color(Color32::DARK_GRAY),
        );
    });

    selected
}

// === Advanced Metrics Panel ===

/// Extended metrics panel with charts
pub fn advanced_metrics_panel(ui: &mut Ui, metrics_history: &[MessageMetrics]) {
    egui::CollapsingHeader::new(
        RichText::new("📈 Performance Metrics")
            .size(12.0)
            .color(Color32::WHITE),
    )
    .default_open(false)
    .show(ui, |ui| {
        if metrics_history.is_empty() {
            ui.label(
                RichText::new("No metrics data yet")
                    .size(10.0)
                    .color(Color32::GRAY)
                    .italics(),
            );
            return;
        }

        // Calculate averages
        let avg_response_time: f64 = metrics_history
            .iter()
            .map(|m| m.total_response_time_ms as f64)
            .sum::<f64>()
            / metrics_history.len() as f64;

        let avg_tokens: f64 = metrics_history
            .iter()
            .map(|m| m.output_tokens as f64)
            .sum::<f64>()
            / metrics_history.len() as f64;

        let avg_ttft: Option<f64> = {
            let ttfts: Vec<u64> = metrics_history
                .iter()
                .filter_map(|m| m.time_to_first_token_ms)
                .collect();
            if !ttfts.is_empty() {
                Some(ttfts.iter().sum::<u64>() as f64 / ttfts.len() as f64)
            } else {
                None
            }
        };

        // Display stats
        ui.horizontal(|ui| {
            ui.label(RichText::new("Responses:").size(10.0).color(Color32::GRAY));
            ui.label(
                RichText::new(metrics_history.len().to_string())
                    .size(10.0)
                    .color(Color32::WHITE),
            );
        });

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Avg Response Time:")
                    .size(10.0)
                    .color(Color32::GRAY),
            );
            ui.label(
                RichText::new(format!("{:.0}ms", avg_response_time))
                    .size(10.0)
                    .color(Color32::LIGHT_GREEN),
            );
        });

        if let Some(ttft) = avg_ttft {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Avg TTFT:").size(10.0).color(Color32::GRAY));
                ui.label(
                    RichText::new(format!("{:.0}ms", ttft))
                        .size(10.0)
                        .color(Color32::LIGHT_BLUE),
                );
            });
        }

        ui.horizontal(|ui| {
            ui.label(RichText::new("Avg Tokens:").size(10.0).color(Color32::GRAY));
            ui.label(
                RichText::new(format!("{:.0}", avg_tokens))
                    .size(10.0)
                    .color(Color32::WHITE),
            );
        });

        // Simple bar chart for recent response times
        ui.add_space(8.0);
        ui.label(
            RichText::new("Recent Response Times:")
                .size(10.0)
                .color(Color32::GRAY),
        );

        let recent: Vec<_> = metrics_history.iter().rev().take(10).collect();
        let max_time = recent
            .iter()
            .map(|m| m.total_response_time_ms)
            .max()
            .unwrap_or(1000);

        ui.horizontal(|ui| {
            for metrics in recent.iter().rev() {
                let height = (metrics.total_response_time_ms as f32 / max_time as f32) * 40.0;
                let height = height.max(2.0);

                let color = if metrics.total_response_time_ms < 1000 {
                    Color32::LIGHT_GREEN
                } else if metrics.total_response_time_ms < 3000 {
                    Color32::YELLOW
                } else {
                    Color32::from_rgb(255, 150, 100)
                };

                let (rect, _) = ui.allocate_exact_size(Vec2::new(8.0, 44.0), egui::Sense::hover());
                ui.painter().rect_filled(
                    egui::Rect::from_min_size(
                        rect.min + Vec2::new(0.0, 44.0 - height),
                        Vec2::new(6.0, height),
                    ),
                    2.0,
                    color,
                );
            }
        });
    });
}

// === Knowledge Source Selection Widgets ===

/// Mode for knowledge source selection
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum KnowledgeSelectionMode {
    /// No knowledge selected
    #[default]
    None,
    /// Single source selected
    Single(String),
    /// Multiple sources selected
    Multiple(Vec<String>),
    /// All available sources
    All,
}

impl KnowledgeSelectionMode {
    /// Check if no knowledge is selected
    pub fn is_none(&self) -> bool {
        matches!(self, KnowledgeSelectionMode::None)
    }

    /// Check if all knowledge is selected
    pub fn is_all(&self) -> bool {
        matches!(self, KnowledgeSelectionMode::All)
    }

    /// Get selected sources as a vec
    pub fn to_sources(&self, all_sources: &[String]) -> Vec<String> {
        match self {
            KnowledgeSelectionMode::None => Vec::new(),
            KnowledgeSelectionMode::Single(s) => vec![s.clone()],
            KnowledgeSelectionMode::Multiple(v) => v.clone(),
            KnowledgeSelectionMode::All => all_sources.to_vec(),
        }
    }

    /// Create from selected sources
    pub fn from_sources(selected: Vec<String>, all_sources: &[String]) -> Self {
        if selected.is_empty() {
            KnowledgeSelectionMode::None
        } else if selected.len() == all_sources.len()
            && all_sources.iter().all(|s| selected.contains(s))
        {
            KnowledgeSelectionMode::All
        } else if selected.len() == 1 {
            KnowledgeSelectionMode::Single(selected[0].clone())
        } else {
            KnowledgeSelectionMode::Multiple(selected)
        }
    }

    /// Check if a specific source is selected
    pub fn contains(&self, source: &str, all_sources: &[String]) -> bool {
        match self {
            KnowledgeSelectionMode::None => false,
            KnowledgeSelectionMode::Single(s) => s == source,
            KnowledgeSelectionMode::Multiple(v) => v.iter().any(|s| s == source),
            KnowledgeSelectionMode::All => all_sources.iter().any(|s| s == source),
        }
    }

    /// Display text for the current selection
    pub fn display_text(&self) -> String {
        match self {
            KnowledgeSelectionMode::None => "No knowledge".to_string(),
            KnowledgeSelectionMode::Single(s) => s.clone(),
            KnowledgeSelectionMode::Multiple(v) => format!("{} sources", v.len()),
            KnowledgeSelectionMode::All => "All knowledge".to_string(),
        }
    }
}

/// State for knowledge source selector widget
#[derive(Debug, Clone, Default)]
pub struct KnowledgeSelectionState {
    /// Current selection mode
    pub selection: KnowledgeSelectionMode,
    /// Whether the multi-select popup is open
    pub popup_open: bool,
    /// Temporary selection during popup editing
    pub temp_selection: Vec<String>,
}

impl KnowledgeSelectionState {
    /// Create a new state with no selection
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with initial selection
    pub fn with_selection(selection: KnowledgeSelectionMode) -> Self {
        Self {
            selection,
            popup_open: false,
            temp_selection: Vec::new(),
        }
    }

    /// Set selection to single source
    pub fn select_single(&mut self, source: impl Into<String>) {
        self.selection = KnowledgeSelectionMode::Single(source.into());
    }

    /// Set selection to multiple sources
    pub fn select_multiple(&mut self, sources: Vec<String>) {
        self.selection = KnowledgeSelectionMode::Multiple(sources);
    }

    /// Set selection to all
    pub fn select_all(&mut self) {
        self.selection = KnowledgeSelectionMode::All;
    }

    /// Clear selection
    pub fn clear(&mut self) {
        self.selection = KnowledgeSelectionMode::None;
    }

    /// Get currently selected sources
    pub fn get_selected(&self, all_sources: &[String]) -> Vec<String> {
        self.selection.to_sources(all_sources)
    }
}

/// Response from knowledge selector widget
#[derive(Debug, Clone, Default)]
pub struct KnowledgeSelectorResponse {
    /// Whether the selection changed
    pub changed: bool,
    /// The new selection (if changed)
    pub selection: Option<KnowledgeSelectionMode>,
}

/// Configuration for knowledge selector widget
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct KnowledgeSelectorConfig {
    /// Label shown before the selector
    pub label: String,
    /// Whether to allow "None" selection
    pub allow_none: bool,
    /// Whether to allow "All" selection
    pub allow_all: bool,
    /// Whether to allow multi-select
    pub allow_multi: bool,
    /// Width of the combo box
    pub width: f32,
    /// Show document stats (chunk count) next to each source
    pub show_stats: bool,
}

impl Default for KnowledgeSelectorConfig {
    fn default() -> Self {
        Self {
            label: "Knowledge:".to_string(),
            allow_none: true,
            allow_all: true,
            allow_multi: true,
            width: 200.0,
            show_stats: true,
        }
    }
}

impl KnowledgeSelectorConfig {
    /// Create config for single-select only
    pub fn single_select() -> Self {
        Self {
            allow_multi: false,
            ..Default::default()
        }
    }

    /// Create config for multi-select only (no All option)
    pub fn multi_select_only() -> Self {
        Self {
            allow_all: false,
            ..Default::default()
        }
    }

    /// Create a compact version
    pub fn compact() -> Self {
        Self {
            label: String::new(),
            width: 150.0,
            show_stats: false,
            ..Default::default()
        }
    }
}

/// Stats for a knowledge source (for display)
#[derive(Debug, Clone)]
pub struct KnowledgeSourceStats {
    /// Source name
    pub name: String,
    /// Number of chunks
    pub chunk_count: usize,
    /// Total tokens
    pub token_count: usize,
}

/// Render a knowledge source selector dropdown
///
/// Simple combo-box style selector that supports None/Single/All modes.
/// For multi-select, use `knowledge_source_multi_selector`.
pub fn knowledge_source_selector(
    ui: &mut Ui,
    state: &mut KnowledgeSelectionState,
    available_sources: &[String],
    config: &KnowledgeSelectorConfig,
) -> KnowledgeSelectorResponse {
    let mut response = KnowledgeSelectorResponse::default();

    if !config.label.is_empty() {
        ui.label(&config.label);
    }

    let display_text = state.selection.display_text();

    egui::ComboBox::from_id_source("knowledge_source_combo")
        .selected_text(&display_text)
        .width(config.width)
        .show_ui(ui, |ui| {
            // None option
            if config.allow_none {
                let is_selected = state.selection.is_none();
                if ui.selectable_label(is_selected, "None").clicked() && !is_selected {
                    state.selection = KnowledgeSelectionMode::None;
                    response.changed = true;
                    response.selection = Some(KnowledgeSelectionMode::None);
                }
            }

            // All option
            if config.allow_all && !available_sources.is_empty() {
                let is_selected = state.selection.is_all();
                let label = format!("All ({} sources)", available_sources.len());
                if ui.selectable_label(is_selected, &label).clicked() && !is_selected {
                    state.selection = KnowledgeSelectionMode::All;
                    response.changed = true;
                    response.selection = Some(KnowledgeSelectionMode::All);
                }
            }

            // Individual sources
            if !available_sources.is_empty() {
                ui.separator();
            }

            for source in available_sources {
                let is_selected =
                    matches!(&state.selection, KnowledgeSelectionMode::Single(s) if s == source);
                if ui.selectable_label(is_selected, source).clicked() && !is_selected {
                    state.selection = KnowledgeSelectionMode::Single(source.clone());
                    response.changed = true;
                    response.selection = Some(KnowledgeSelectionMode::Single(source.clone()));
                }
            }
        });

    response
}

/// Render a multi-select knowledge source selector
///
/// Uses checkboxes to allow selecting multiple sources independently.
pub fn knowledge_source_multi_selector(
    ui: &mut Ui,
    state: &mut KnowledgeSelectionState,
    available_sources: &[String],
    stats: Option<&[KnowledgeSourceStats]>,
    config: &KnowledgeSelectorConfig,
) -> KnowledgeSelectorResponse {
    let mut response = KnowledgeSelectorResponse::default();

    if !config.label.is_empty() {
        ui.label(&config.label);
    }

    let display_text = state.selection.display_text();

    // Main combo that opens popup
    let combo_response = egui::ComboBox::from_id_source("knowledge_multi_combo")
        .selected_text(&display_text)
        .width(config.width)
        .show_ui(ui, |ui| {
            // Quick options at top
            ui.horizontal(|ui| {
                if config.allow_none {
                    if ui.small_button("None").clicked() {
                        state.selection = KnowledgeSelectionMode::None;
                        response.changed = true;
                        response.selection = Some(KnowledgeSelectionMode::None);
                    }
                }
                if config.allow_all && !available_sources.is_empty() {
                    if ui.small_button("All").clicked() {
                        state.selection = KnowledgeSelectionMode::All;
                        response.changed = true;
                        response.selection = Some(KnowledgeSelectionMode::All);
                    }
                }
            });

            if !available_sources.is_empty() {
                ui.separator();
            }

            // Checkboxes for each source
            for source in available_sources {
                let was_selected = state.selection.contains(source, available_sources);
                let mut selected = was_selected;

                // Find stats if available
                let stat_text = if config.show_stats {
                    stats
                        .and_then(|s| s.iter().find(|st| st.name == *source))
                        .map(|s| format!(" ({} chunks)", s.chunk_count))
                        .unwrap_or_default()
                } else {
                    String::new()
                };

                let label = format!("{}{}", source, stat_text);

                if ui.checkbox(&mut selected, &label).changed() {
                    // Update selection
                    let mut current = state.selection.to_sources(available_sources);

                    if selected && !current.contains(source) {
                        current.push(source.clone());
                    } else if !selected {
                        current.retain(|s| s != source);
                    }

                    state.selection =
                        KnowledgeSelectionMode::from_sources(current, available_sources);
                    response.changed = true;
                    response.selection = Some(state.selection.clone());
                }
            }
        });

    // Show tooltip with current selection
    if !state.selection.is_none() {
        combo_response.response.on_hover_text(format!(
            "Selected: {}",
            state.selection.to_sources(available_sources).join(", ")
        ));
    }

    response
}

/// Render a horizontal chip-style multi-selector for knowledge sources
///
/// Shows each source as a toggleable chip/badge.
pub fn knowledge_source_chips(
    ui: &mut Ui,
    state: &mut KnowledgeSelectionState,
    available_sources: &[String],
    config: &KnowledgeSelectorConfig,
) -> KnowledgeSelectorResponse {
    let mut response = KnowledgeSelectorResponse::default();

    if !config.label.is_empty() {
        ui.label(&config.label);
    }

    ui.horizontal_wrapped(|ui| {
        // Quick toggle buttons
        if config.allow_none {
            let is_none = state.selection.is_none();
            if ui
                .selectable_label(is_none, RichText::new("None").size(10.0))
                .clicked()
                && !is_none
            {
                state.selection = KnowledgeSelectionMode::None;
                response.changed = true;
                response.selection = Some(KnowledgeSelectionMode::None);
            }
        }

        if config.allow_all && !available_sources.is_empty() {
            let is_all = state.selection.is_all();
            if ui
                .selectable_label(is_all, RichText::new("All").size(10.0))
                .clicked()
                && !is_all
            {
                state.selection = KnowledgeSelectionMode::All;
                response.changed = true;
                response.selection = Some(KnowledgeSelectionMode::All);
            }
        }

        if (!config.allow_none && !config.allow_all) || !available_sources.is_empty() {
            ui.separator();
        }

        // Individual source chips
        for source in available_sources {
            let is_selected = state.selection.contains(source, available_sources);

            let chip_color = if is_selected {
                Color32::from_rgb(60, 100, 140)
            } else {
                Color32::from_rgb(50, 55, 65)
            };

            let text_color = if is_selected {
                Color32::WHITE
            } else {
                Color32::GRAY
            };

            let chip_response = egui::Frame::none()
                .fill(chip_color)
                .rounding(12.0)
                .inner_margin(egui::Margin::symmetric(8.0, 4.0))
                .show(ui, |ui| {
                    ui.label(RichText::new(source).size(10.0).color(text_color));
                });

            if chip_response
                .response
                .interact(egui::Sense::click())
                .clicked()
            {
                let mut current = state.selection.to_sources(available_sources);

                if is_selected {
                    current.retain(|s| s != source);
                } else {
                    current.push(source.clone());
                }

                state.selection = KnowledgeSelectionMode::from_sources(current, available_sources);
                response.changed = true;
                response.selection = Some(state.selection.clone());
            }
        }
    });

    response
}

/// Render knowledge selection indicator (read-only display)
///
/// Shows what knowledge is currently selected without allowing changes.
/// Useful when the application controls selection externally.
pub fn knowledge_selection_display(
    ui: &mut Ui,
    selection: &KnowledgeSelectionMode,
    available_sources: &[String],
) {
    let (icon, text, color) = match selection {
        KnowledgeSelectionMode::None => ("📭", "No knowledge".to_string(), Color32::GRAY),
        KnowledgeSelectionMode::Single(s) => ("📄", s.clone(), Color32::LIGHT_BLUE),
        KnowledgeSelectionMode::Multiple(v) => {
            let text = if v.len() <= 2 {
                v.join(", ")
            } else {
                format!("{}, {} +{} more", v[0], v[1], v.len() - 2)
            };
            ("📚", text, Color32::LIGHT_BLUE)
        }
        KnowledgeSelectionMode::All => (
            "📚",
            format!("All ({} sources)", available_sources.len()),
            Color32::LIGHT_GREEN,
        ),
    };

    ui.horizontal(|ui| {
        ui.label(RichText::new(icon).size(12.0));
        ui.label(RichText::new(&text).size(10.0).color(color));
    });
}

/// External knowledge control configuration
///
/// Use this when the application wants to control knowledge selection
/// instead of letting the widget manage it.
#[derive(Debug, Clone)]
pub struct ExternalKnowledgeControl {
    /// The sources to use (controlled by the application)
    pub sources: Vec<String>,
    /// Whether to show a read-only indicator
    pub show_indicator: bool,
    /// Label for the indicator
    pub label: String,
}

impl ExternalKnowledgeControl {
    /// Create with specific sources
    pub fn with_sources(sources: Vec<String>) -> Self {
        Self {
            sources,
            show_indicator: true,
            label: "Knowledge:".to_string(),
        }
    }

    /// Create for all sources
    pub fn all() -> Self {
        Self {
            sources: Vec::new(), // Empty means "use all available"
            show_indicator: true,
            label: "Knowledge:".to_string(),
        }
    }

    /// Create with no knowledge
    pub fn none() -> Self {
        Self {
            sources: vec![], // Will be treated as none
            show_indicator: false,
            label: String::new(),
        }
    }

    /// Set whether to show indicator
    pub fn with_indicator(mut self, show: bool) -> Self {
        self.show_indicator = show;
        self
    }

    /// Convert to selection mode
    pub fn to_selection_mode(&self, all_available: &[String]) -> KnowledgeSelectionMode {
        if self.sources.is_empty() {
            // Check if this is meant to be "all" or "none"
            // For "all", we'd set sources to the available ones first
            KnowledgeSelectionMode::None
        } else if self.sources.len() == all_available.len()
            && all_available.iter().all(|s| self.sources.contains(s))
        {
            KnowledgeSelectionMode::All
        } else if self.sources.len() == 1 {
            KnowledgeSelectionMode::Single(self.sources[0].clone())
        } else {
            KnowledgeSelectionMode::Multiple(self.sources.clone())
        }
    }
}

/// Render knowledge selection with external control
///
/// This variant does NOT allow user interaction - the application
/// controls what knowledge is used.
pub fn knowledge_selection_external(
    ui: &mut Ui,
    control: &ExternalKnowledgeControl,
    all_available: &[String],
) {
    if !control.show_indicator {
        return;
    }

    if !control.label.is_empty() {
        ui.label(&control.label);
    }

    let selection = control.to_selection_mode(all_available);
    knowledge_selection_display(ui, &selection, all_available);
}

// ============================================================================
// RAG Tier Selection Widgets
// ============================================================================

use crate::rag_tiers::{RagTierConfig, RagFeatures, RagTier};

/// Colors for RAG tier UI elements
#[derive(Clone)]
pub struct RagTierColors {
    /// Background color for disabled tier
    pub disabled: Color32,
    /// Background color for fast/basic tiers
    pub fast: Color32,
    /// Background color for semantic tiers
    pub semantic: Color32,
    /// Background color for enhanced tiers
    pub enhanced: Color32,
    /// Background color for thorough tiers
    pub thorough: Color32,
    /// Background color for agentic/advanced tiers
    pub agentic: Color32,
    /// Text color for selected tier
    pub selected_text: Color32,
    /// Text color for unselected tier
    pub unselected_text: Color32,
}

impl Default for RagTierColors {
    fn default() -> Self {
        Self {
            disabled: Color32::from_rgb(60, 60, 60),
            fast: Color32::from_rgb(50, 80, 50),
            semantic: Color32::from_rgb(50, 70, 90),
            enhanced: Color32::from_rgb(70, 60, 90),
            thorough: Color32::from_rgb(90, 60, 70),
            agentic: Color32::from_rgb(90, 70, 50),
            selected_text: Color32::WHITE,
            unselected_text: Color32::GRAY,
        }
    }
}

impl RagTierColors {
    /// Get color for a specific tier
    pub fn for_tier(&self, tier: RagTier) -> Color32 {
        match tier {
            RagTier::Disabled => self.disabled,
            RagTier::Fast => self.fast,
            RagTier::Semantic => self.semantic,
            RagTier::Enhanced => self.enhanced,
            RagTier::Thorough => self.thorough,
            RagTier::Agentic | RagTier::Graph | RagTier::Full => self.agentic,
            RagTier::Custom => self.enhanced,
        }
    }
}

/// Response from RAG tier selection widgets
#[derive(Debug, Clone, Default)]
pub struct RagTierResponse {
    /// The newly selected tier, if changed
    pub selected: Option<RagTier>,
    /// Whether the user clicked "Configure" for custom settings
    pub configure_clicked: bool,
    /// Whether the user hovered over a tier (for tooltips)
    pub hovered_tier: Option<RagTier>,
}

impl RagTierResponse {
    /// Check if any change occurred
    pub fn changed(&self) -> bool {
        self.selected.is_some()
    }
}

/// Render a compact RAG tier selector as a dropdown
///
/// Shows the current tier with an emoji indicator and allows selection from all tiers.
pub fn rag_tier_dropdown(ui: &mut Ui, current_tier: &mut RagTier, label: &str) -> RagTierResponse {
    let mut response = RagTierResponse::default();

    ui.horizontal(|ui| {
        if !label.is_empty() {
            ui.label(label);
        }

        let display_text = format!("{} {}", current_tier.emoji(), current_tier.display_name());

        egui::ComboBox::from_id_source("rag_tier_selector")
            .selected_text(&display_text)
            .show_ui(ui, |ui| {
                for tier in RagTier::standard_tiers() {
                    let tier_text = format!("{} {}", tier.emoji(), tier.display_name());
                    let is_selected = *current_tier == *tier;

                    let resp = ui.selectable_label(is_selected, &tier_text);

                    if resp.hovered() {
                        response.hovered_tier = Some(*tier);
                    }

                    // Show tooltip on hover
                    resp.clone().on_hover_text(tier.description());

                    if resp.clicked() && !is_selected {
                        *current_tier = *tier;
                        response.selected = Some(*tier);
                    }
                }
            });
    });

    response
}

/// Render a horizontal button bar for RAG tier selection
///
/// Shows all standard tiers as clickable buttons with visual indicators.
pub fn rag_tier_button_bar(
    ui: &mut Ui,
    current_tier: &mut RagTier,
    colors: &RagTierColors,
    compact: bool,
) -> RagTierResponse {
    let mut response = RagTierResponse::default();

    ui.horizontal(|ui| {
        for tier in RagTier::standard_tiers() {
            let is_selected = *current_tier == *tier;
            let bg_color = if is_selected {
                colors.for_tier(*tier)
            } else {
                Color32::from_rgb(40, 40, 40)
            };
            let text_color = if is_selected {
                colors.selected_text
            } else {
                colors.unselected_text
            };

            let label_text = if compact {
                tier.emoji().to_string()
            } else {
                format!("{} {}", tier.emoji(), tier.short_label())
            };

            let button = egui::Button::new(
                RichText::new(&label_text)
                    .color(text_color)
                    .size(if compact { 14.0 } else { 11.0 }),
            )
            .fill(bg_color)
            .rounding(4.0);

            let resp = ui.add(button);
            let clicked = resp.clicked();
            let hovered = resp.hovered();

            if hovered {
                response.hovered_tier = Some(*tier);
            }

            // Show tooltip (consumes resp)
            resp.on_hover_ui(|ui| {
                ui.label(RichText::new(tier.display_name()).strong());
                ui.label(tier.description());
                let (min, max) = tier.estimated_extra_calls();
                let max_str = max.map(|m| m.to_string()).unwrap_or("∞".to_string());
                ui.label(format!("LLM calls: {}-{}", min, max_str));
                if tier.requires_embeddings() {
                    ui.label(
                        RichText::new("Requires embeddings")
                            .color(Color32::YELLOW)
                            .size(10.0),
                    );
                }
            });

            if clicked && !is_selected {
                *current_tier = *tier;
                response.selected = Some(*tier);
            }
        }
    });

    response
}

/// Render a vertical tier selector with descriptions
///
/// Shows each tier as a radio-button-like selection with full descriptions.
pub fn rag_tier_radio_list(
    ui: &mut Ui,
    current_tier: &mut RagTier,
    colors: &RagTierColors,
) -> RagTierResponse {
    let mut response = RagTierResponse::default();

    for tier in RagTier::standard_tiers() {
        let is_selected = *current_tier == *tier;

        ui.horizontal(|ui| {
            // Radio indicator
            let indicator = if is_selected { "◉" } else { "○" };
            let indicator_color = if is_selected {
                colors.for_tier(*tier)
            } else {
                Color32::GRAY
            };

            ui.label(RichText::new(indicator).color(indicator_color).size(14.0));

            // Tier button/label
            let frame = egui::Frame::none()
                .fill(if is_selected {
                    colors.for_tier(*tier).linear_multiply(0.3)
                } else {
                    Color32::TRANSPARENT
                })
                .rounding(4.0)
                .inner_margin(4.0);

            let resp = frame
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(tier.emoji()).size(14.0));
                        ui.vertical(|ui| {
                            ui.label(RichText::new(tier.display_name()).strong().size(12.0));
                            ui.label(
                                RichText::new(tier.description())
                                    .size(10.0)
                                    .color(Color32::LIGHT_GRAY),
                            );

                            // Show call estimate
                            let (min, max) = tier.estimated_extra_calls();
                            let calls_text = if max == Some(0) {
                                "No extra LLM calls".to_string()
                            } else if let Some(m) = max {
                                format!("{}-{} LLM calls", min, m)
                            } else {
                                format!("{}+ LLM calls (unbounded)", min)
                            };
                            ui.label(RichText::new(calls_text).size(9.0).color(Color32::GRAY));
                        });
                    });
                })
                .response;

            if resp.hovered() {
                response.hovered_tier = Some(*tier);
            }

            if resp.interact(egui::Sense::click()).clicked() && !is_selected {
                *current_tier = *tier;
                response.selected = Some(*tier);
            }
        });

        ui.add_space(2.0);
    }

    response
}

/// Render a slider-based tier selector
///
/// Shows tiers on a horizontal slider, useful for quick adjustment.
pub fn rag_tier_slider(ui: &mut Ui, current_tier: &mut RagTier, label: &str) -> RagTierResponse {
    let mut response = RagTierResponse::default();

    // Skip Custom tier for slider (use standard tiers only)
    let mut level = current_tier.complexity_level().min(7);

    ui.horizontal(|ui| {
        if !label.is_empty() {
            ui.label(label);
        }

        // Emoji indicator
        ui.label(RichText::new(current_tier.emoji()).size(16.0));

        // Slider
        let slider_resp = ui.add(
            egui::Slider::new(&mut level, 0..=7)
                .show_value(false)
                .text(current_tier.display_name()),
        );

        if slider_resp.changed() {
            let new_tier = RagTier::from_level(level);
            if new_tier != *current_tier {
                *current_tier = new_tier;
                response.selected = Some(new_tier);
            }
        }

        if slider_resp.hovered() {
            response.hovered_tier = Some(*current_tier);
            slider_resp.on_hover_text(current_tier.description());
        }
    });

    response
}

/// Render a tier info panel showing current configuration details
///
/// Displays the features enabled for the current tier and requirements.
pub fn rag_tier_info_panel(ui: &mut Ui, tier: RagTier, config: Option<&RagTierConfig>) {
    let features = config
        .map(|c| c.effective_features())
        .unwrap_or_else(|| tier.to_features());

    egui::Frame::none()
        .fill(Color32::from_rgb(30, 30, 35))
        .rounding(6.0)
        .inner_margin(8.0)
        .show(ui, |ui| {
            // Header
            ui.horizontal(|ui| {
                ui.label(RichText::new(tier.emoji()).size(18.0));
                ui.label(RichText::new(tier.display_name()).strong().size(14.0));
            });

            ui.label(
                RichText::new(tier.description())
                    .size(11.0)
                    .color(Color32::LIGHT_GRAY),
            );
            ui.add_space(4.0);

            // Call estimate
            let (min, max) = tier.estimated_extra_calls();
            let calls_text = match max {
                Some(0) => "No additional LLM calls".to_string(),
                Some(m) if m == min => format!("{} LLM calls", min),
                Some(m) => format!("{}-{} LLM calls", min, m),
                None => format!("{}+ LLM calls (iterative)", min),
            };
            ui.label(RichText::new(calls_text).size(10.0));

            ui.add_space(4.0);
            ui.separator();
            ui.add_space(2.0);

            // Enabled features (in a grid)
            let enabled = features.enabled_features();
            if !enabled.is_empty() {
                ui.label(
                    RichText::new("Enabled Features:")
                        .size(10.0)
                        .color(Color32::GRAY),
                );
                ui.horizontal_wrapped(|ui| {
                    for feature in enabled {
                        let nice_name = feature.replace('_', " ");
                        ui.label(
                            RichText::new(format!("• {}", nice_name))
                                .size(9.0)
                                .color(Color32::LIGHT_GREEN),
                        );
                    }
                });
            }

            // Requirements
            if let Some(cfg) = config {
                let reqs = cfg.check_requirements();
                if !reqs.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("Requirements:")
                            .size(10.0)
                            .color(Color32::GRAY),
                    );
                    for req in reqs {
                        ui.label(
                            RichText::new(format!("⚠ {}", req.display_name()))
                                .size(9.0)
                                .color(Color32::YELLOW),
                        );
                    }
                }
            }
        });
}

/// Render a compact tier badge for display in headers/status bars
pub fn rag_tier_badge(ui: &mut Ui, tier: RagTier, colors: &RagTierColors) {
    let bg_color = colors.for_tier(tier);

    egui::Frame::none()
        .fill(bg_color)
        .rounding(3.0)
        .inner_margin(Vec2::new(4.0, 2.0))
        .show(ui, |ui| {
            ui.label(
                RichText::new(format!("{} {}", tier.emoji(), tier.short_label()))
                    .size(10.0)
                    .color(colors.selected_text),
            );
        });
}

/// Combined RAG tier selector widget with info panel
///
/// Provides a full tier selection interface with dropdown, description,
/// and optional feature details.
pub fn rag_tier_selector_full(
    ui: &mut Ui,
    current_tier: &mut RagTier,
    mut config: Option<&mut RagTierConfig>,
    show_details: bool,
) -> RagTierResponse {
    let mut response = RagTierResponse::default();

    ui.vertical(|ui| {
        // Selector row
        ui.horizontal(|ui| {
            ui.label("RAG Tier:");
            let dropdown_resp = rag_tier_dropdown(ui, current_tier, "");
            if dropdown_resp.changed() {
                response = dropdown_resp;
                // Update config if provided
                if let Some(ref mut cfg) = config {
                    cfg.tier = *current_tier;
                    cfg.features = current_tier.to_features();
                    cfg.use_custom_features = false;
                }
            }
        });

        // Details panel
        if show_details {
            ui.add_space(4.0);
            rag_tier_info_panel(ui, *current_tier, config.as_deref());
        }
    });

    response
}

/// Render feature toggles for custom RAG configuration
///
/// Allows enabling/disabling individual RAG features when using Custom tier.
pub fn rag_features_editor(ui: &mut Ui, features: &mut RagFeatures) -> bool {
    let mut changed = false;

    egui::Grid::new("rag_features_grid")
        .num_columns(2)
        .spacing([20.0, 4.0])
        .show(ui, |ui| {
            // Retrieval Methods
            ui.label(RichText::new("Retrieval").strong().size(11.0));
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.fts_search, "Full-text search")
                .changed();
            changed |= ui
                .checkbox(&mut features.semantic_search, "Semantic search")
                .changed();
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.hybrid_search, "Hybrid search")
                .changed();
            changed |= ui
                .checkbox(&mut features.fusion_rrf, "RRF fusion")
                .changed();
            ui.end_row();

            ui.add_space(4.0);
            ui.end_row();

            // Query Enhancement
            ui.label(RichText::new("Query Enhancement").strong().size(11.0));
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.synonym_expansion, "Synonym expansion")
                .changed();
            changed |= ui
                .checkbox(&mut features.query_expansion, "Query expansion (LLM)")
                .changed();
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.multi_query, "Multi-query")
                .changed();
            changed |= ui.checkbox(&mut features.hyde, "HyDE").changed();
            ui.end_row();

            ui.add_space(4.0);
            ui.end_row();

            // Result Processing
            ui.label(RichText::new("Result Processing").strong().size(11.0));
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.reranking, "Reranking (LLM)")
                .changed();
            changed |= ui
                .checkbox(&mut features.cross_encoder_rerank, "Cross-encoder")
                .changed();
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.contextual_compression, "Compression")
                .changed();
            changed |= ui
                .checkbox(&mut features.sentence_window, "Sentence window")
                .changed();
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.parent_document, "Parent document")
                .changed();
            ui.end_row();

            ui.add_space(4.0);
            ui.end_row();

            // Self-Improvement
            ui.label(RichText::new("Self-Improvement").strong().size(11.0));
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.self_reflection, "Self-reflection")
                .changed();
            changed |= ui
                .checkbox(&mut features.corrective_rag, "Corrective RAG")
                .changed();
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.adaptive_strategy, "Adaptive strategy")
                .changed();
            ui.end_row();

            ui.add_space(4.0);
            ui.end_row();

            // Advanced
            ui.label(RichText::new("Advanced").strong().size(11.0));
            ui.end_row();

            changed |= ui
                .checkbox(&mut features.agentic_mode, "Agentic mode")
                .changed();
            changed |= ui.checkbox(&mut features.graph_rag, "Graph RAG").changed();
            ui.end_row();

            changed |= ui.checkbox(&mut features.raptor, "RAPTOR").changed();
            changed |= ui
                .checkbox(&mut features.multimodal, "Multimodal")
                .changed();
            ui.end_row();
        });

    changed
}
