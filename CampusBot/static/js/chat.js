/**
 * CampusBot Client-side Chat Controller
 * Manages conversation history, message streaming, Markdown rendering,
 * source citations, Ollama status checks, and suggested question triggers.
 */

// Global Application State
let activeConversationId = null;
let isGenerating = false;
let currentConversations = [];
let shouldAutoScroll = true;
const defaultSuggestedQuestions = Array.isArray(window.CAMPUSBOT_SUGGESTIONS) ? window.CAMPUSBOT_SUGGESTIONS : [
    "What engineering branches are available?",
    "What is the admission process?",
    "What are the eligibility criteria?",
    "What is the fee structure?",
    "What documents are required for admission?",
    "What scholarships and financial aid are available?"
];

document.addEventListener("DOMContentLoaded", () => {
    initChatApp();
});

async function initChatApp() {
    // 1. Initialize DOM references
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const newChatBtn = document.getElementById("new-chat-btn");
    const clearChatBtn = document.getElementById("clear-active-chat-btn");
    const charCounter = document.getElementById("char-counter");
    const mobileToggle = document.getElementById("chat-mobile-toggle");
    const sidebarCloseBtn = document.getElementById("sidebar-close-btn");
    const sidebarBackdrop = document.getElementById("sidebar-backdrop");

    // 2. Setup textarea auto-resize and keyboard shortcuts
    if (userInput) {
        userInput.addEventListener("input", () => {
            adjustTextareaHeight(userInput);
            const len = userInput.value.length;
            if (charCounter) charCounter.textContent = `${len} / 3000`;
            if (sendBtn) sendBtn.disabled = len === 0 || isGenerating;
        });

        userInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!sendBtn.disabled && !isGenerating) {
                    submitMessage();
                }
            }
        });
    }

    // 3. Form submit handler
    if (chatForm) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            if (!isGenerating && userInput.value.trim()) {
                submitMessage();
            }
        });
    }

    // 4. Action buttons
    if (newChatBtn) {
        newChatBtn.addEventListener("click", () => startNewConversation());
    }

    if (clearChatBtn) {
        clearChatBtn.addEventListener("click", () => clearCurrentConversation());
    }

    // 5. Mobile Sidebar Drawer
    if (mobileToggle) {
        mobileToggle.addEventListener("click", () => toggleSidebar(true));
    }
    if (sidebarCloseBtn) {
        sidebarCloseBtn.addEventListener("click", () => toggleSidebar(false));
    }
    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener("click", () => toggleSidebar(false));
    }

    // 6. Setup Suggestion Cards
    renderSuggestionCards(defaultSuggestedQuestions);

    // 7. Setup Modals
    initModals();

    // 8. Offline retry button
    const retryBtn = document.getElementById("offline-retry-btn");
    if (retryBtn) {
        retryBtn.addEventListener("click", () => checkSystemStatus());
    }

    // 9. Fetch Status & Conversations
    await checkSystemStatus();
    await loadConversations();
}

// ============================================================================
// SYSTEM & OLLAMA STATUS CHECKING
// ============================================================================
async function checkSystemStatus() {
    const statusDot = document.getElementById("ai-status-dot");
    const statusText = document.getElementById("ai-status-text");
    const modelLabel = document.getElementById("ai-model-name");
    const chunksLabel = document.getElementById("ai-chunks-count");
    const topDot = document.getElementById("topbar-status-dot");
    const topText = document.getElementById("topbar-status-text");
    const offlineBanner = document.getElementById("offline-banner");
    const offlineMsg = document.getElementById("offline-banner-msg");

    try {
        const response = await fetch("/api/status");
        if (!response.ok) throw new Error("Status query failed");
        const data = await response.json();

        const isOnline = data.ollama_online === true;
        const isChatReady = data.chat_model_available === true;
        const totalChunks = (data.vector_store && data.vector_store.chunks_count) || 0;

        if (modelLabel) modelLabel.textContent = data.chat_model || "llama3.2:3b";
        if (chunksLabel) chunksLabel.textContent = `${totalChunks} chunks`;

        if (isOnline && isChatReady) {
            // All Green
            if (statusDot) statusDot.className = "status-dot status-online";
            if (statusText) statusText.textContent = "AI Online";
            if (topDot) topDot.className = "pulse-dot-sm status-online";
            if (topText) topText.textContent = "Ollama • LLaMA 3.2";
            if (offlineBanner) offlineBanner.style.display = "none";
        } else if (isOnline && !isChatReady) {
            // Ollama is up, but chat model needs to be pulled
            if (statusDot) statusDot.className = "status-dot status-warning";
            if (statusText) statusText.textContent = "Model Missing";
            if (topDot) topDot.className = "pulse-dot-sm status-warning";
            if (topText) topText.textContent = "Pull Model Required";
            if (offlineBanner) {
                offlineBanner.style.display = "flex";
                if (offlineMsg) {
                    offlineMsg.innerHTML = `Model <code>${data.chat_model}</code> is missing. Run: <code>${data.setup_command || 'ollama pull ' + data.chat_model}</code>`;
                }
            }
        } else {
            // Ollama is completely offline
            if (statusDot) statusDot.className = "status-dot status-offline";
            if (statusText) statusText.textContent = "Offline";
            if (topDot) topDot.className = "pulse-dot-sm status-offline";
            if (topText) topText.textContent = "AI Offline";
            if (offlineBanner) {
                offlineBanner.style.display = "flex";
                if (offlineMsg) {
                    offlineMsg.textContent = "CampusBot's local AI service is unavailable. Please start Ollama and try again.";
                }
            }
        }
    } catch (err) {
        if (statusDot) statusDot.className = "status-dot status-offline";
        if (statusText) statusText.textContent = "Error";
        if (topDot) topDot.className = "pulse-dot-sm status-offline";
        if (topText) topText.textContent = "Offline";
        if (offlineBanner) offlineBanner.style.display = "flex";
    }
}

// ============================================================================
// CONVERSATIONS MANAGEMENT (SQLITE)
// ============================================================================
async function loadConversations() {
    const listEl = document.getElementById("conversations-list");
    if (!listEl) return;

    try {
        const res = await fetch("/api/conversations");
        if (!res.ok) throw new Error("Failed to load conversations");
        const data = await res.json();
        currentConversations = data.conversations || [];

        renderConversationsList();

        if (currentConversations.length > 0 && !activeConversationId) {
            // Select the most recent conversation
            selectConversation(currentConversations[0].id);
        } else if (currentConversations.length === 0 && !activeConversationId) {
            // Show fresh welcome view
            showWelcomeView();
        }
    } catch (e) {
        listEl.innerHTML = `<div class="history-empty">Could not load chats</div>`;
    }
}

function renderConversationsList() {
    const listEl = document.getElementById("conversations-list");
    if (!listEl) return;

    if (currentConversations.length === 0) {
        listEl.innerHTML = `
            <div class="history-empty">
                <i class="fa-regular fa-comment-dots"></i>
                <p>No conversations yet. Start asking a question!</p>
            </div>
        `;
        return;
    }

    listEl.innerHTML = "";
    currentConversations.forEach((conv) => {
        const item = document.createElement("div");
        item.className = `conversation-item ${conv.id === activeConversationId ? "active" : ""}`;
        item.dataset.id = conv.id;

        const titleSpan = document.createElement("span");
        titleSpan.className = "conv-title";
        titleSpan.textContent = conv.title || "New Conversation";
        titleSpan.title = conv.title || "New Conversation";

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "conv-delete-btn";
        deleteBtn.title = "Delete conversation";
        deleteBtn.innerHTML = `<i class="fa-solid fa-trash-can"></i>`;
        deleteBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            deleteConversation(conv.id);
        });

        item.appendChild(titleSpan);
        item.appendChild(deleteBtn);

        item.addEventListener("click", () => {
            if (activeConversationId !== conv.id) {
                selectConversation(conv.id);
            }
            toggleSidebar(false);
        });

        listEl.appendChild(item);
    });
}

async function startNewConversation() {
    try {
        const res = await fetch("/api/conversations", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: "New Conversation" })
        });
        if (!res.ok) throw new Error();
        const newConv = await res.json();
        activeConversationId = newConv.id;
        await loadConversations();
        selectConversation(newConv.id);
    } catch (e) {
        // Fallback to fresh client state
        activeConversationId = null;
        showWelcomeView();
        renderConversationsList();
    }
}

async function selectConversation(convId) {
    activeConversationId = convId;
    renderConversationsList();

    const titleEl = document.getElementById("active-conv-title");
    const activeConv = currentConversations.find((c) => c.id === convId);
    if (titleEl && activeConv) {
        titleEl.textContent = activeConv.title;
    }

    try {
        const res = await fetch(`/api/conversations/${convId}`);
        if (!res.ok) throw new Error();
        const convData = await res.json();

        const messages = convData.messages || [];
        if (messages.length === 0) {
            showWelcomeView();
        } else {
            showMessagesFeed();
            const feed = document.getElementById("messages-feed");
            feed.innerHTML = "";
            messages.forEach((m) => {
                appendMessageToUI(m.role, m.content, m.sources || [], m.mode || "general_ai", false);
            });
            scrollToBottom();
        }
    } catch (e) {
        showWelcomeView();
    }
}

async function deleteConversation(convId) {
    if (!confirm("Are you sure you want to delete this conversation?")) return;

    try {
        await fetch(`/api/conversations/${convId}`, { method: "DELETE" });
        if (activeConversationId === convId) {
            activeConversationId = null;
        }
        await loadConversations();
        if (!activeConversationId) {
            if (currentConversations.length > 0) {
                selectConversation(currentConversations[0].id);
            } else {
                showWelcomeView();
            }
        }
    } catch (e) {
        alert("Failed to delete conversation.");
    }
}

async function clearCurrentConversation() {
    if (!activeConversationId) return;
    if (!confirm("Clear all messages in the current conversation?")) return;

    try {
        await fetch(`/api/conversations/${activeConversationId}/clear`, { method: "POST" });
        showWelcomeView();
        await loadConversations();
    } catch (e) {
        alert("Failed to clear messages.");
    }
}

// ============================================================================
// CHAT MESSAGE SUBMISSION & HANDLING
// ============================================================================
async function submitMessage(queryText = null) {
    const inputEl = document.getElementById("user-input");
    const message = (queryText !== null ? queryText : inputEl.value).trim();
    if (!message || isGenerating) return;

    // Snapshot the typed text so it can be restored if the request fails.
    const typedText = queryText === null ? inputEl.value : null;

    // Switch view to messages feed
    showMessagesFeed();

    // 1. Render User Message (optimistic)
    appendMessageToUI("user", message, [], null, true);
    scrollToBottom();

    // 2. Show Typing Indicator & disable inputs (also blocks duplicate submits)
    setGeneratingState(true);

    let requestSucceeded = false;

    try {
        const payload = {
            message: message,
            conversation_id: activeConversationId
        };

        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        requestSucceeded = true;

        // If conversation_id was assigned or refreshed, update state
        if (data.conversation_id && data.conversation_id !== activeConversationId) {
            activeConversationId = data.conversation_id;
            loadConversations();
        }

        // Update active mode pill in header
        updateActiveModePill(data.mode);

        // 3. Render AI Response
        if (data.success) {
            appendMessageToUI("assistant", data.answer, data.sources || [], data.mode, true);
        } else {
            // Render friendly error / instruction message
            const errorMsg = data.answer || data.error || "CampusBot encountered an issue. Please try again.";
            appendErrorMessageToUI(errorMsg, message);
        }

        // Refresh conversation titles in sidebar
        loadConversations();
        renderSuggestionCards(defaultSuggestedQuestions);
        showResponseSuggestions();
    } catch (err) {
        appendErrorMessageToUI("Network connection error. Please ensure the CampusBot server is running.", message);
        renderSuggestionCards(defaultSuggestedQuestions);
        showResponseSuggestions();
    } finally {
        // Only clear the input once the message has been accepted for processing.
        // If the request failed, restore the typed text so the student can retry.
        if (typedText !== null) {
            if (requestSucceeded) {
                inputEl.value = "";
            } else {
                inputEl.value = typedText;
            }
            adjustTextareaHeight(inputEl);
            const charCounter = document.getElementById("char-counter");
            if (charCounter) charCounter.textContent = `${inputEl.value.length} / 3000`;
            inputEl.focus();
        }
        setGeneratingState(false);
        scrollToBottom();
    }
}

function appendMessageToUI(role, content, sources = [], mode = "general_ai", animate = false) {
    const feed = document.getElementById("messages-feed");
    if (!feed) return;

    const row = document.createElement("div");
    row.className = `message-row message-${role}`;

    // Avatar
    const avatar = document.createElement("div");
    avatar.className = `msg-avatar ${role === "user" ? "user-avatar" : "bot-avatar"}`;
    avatar.innerHTML = role === "user" ? `<i class="fa-solid fa-user"></i>` : `<i class="fa-solid fa-robot"></i>`;

    // Content Bubble
    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";

    if (role === "user") {
        const textP = document.createElement("p");
        textP.className = "user-text-content";
        textP.textContent = content;
        bubble.appendChild(textP);
    } else {
        // AI Markdown Rendering with DOMPurify sanitization
        const contentBox = document.createElement("div");
        contentBox.className = "markdown-body";
        const rawHtml = marked.parse(content || "");
        contentBox.innerHTML = DOMPurify.sanitize(rawHtml);
        bubble.appendChild(contentBox);

        // Grounding Badge & Sources List
        if (sources && sources.length > 0) {
            const citationsWrap = document.createElement("div");
            citationsWrap.className = "sources-citations-wrap";

            const tag = document.createElement("div");
            tag.className = "sources-label";
            const isDemo = sources.some(s => /demo_/i.test(s.document_name || ""));
            const labelText = isDemo ? `Demo Sources (${sources.length}):` : `Verified Sources (${sources.length}):`;
            const iconClass = isDemo ? "fa-solid fa-flask" : "fa-solid fa-file-shield";
            tag.innerHTML = `<i class="${iconClass} text-primary"></i> <strong>${labelText}</strong>`;
            citationsWrap.appendChild(tag);

            const pillsList = document.createElement("div");
            pillsList.className = "source-pills-list";

            sources.forEach((s) => {
                const pill = document.createElement("button");
                pill.className = "source-pill-btn";
                pill.innerHTML = `
                    <i class="fa-solid fa-file-lines"></i>
                    <span>${escapeHtml(s.document_name)}</span>
                    <small>p.${s.page}</small>
                `;
                pill.title = `Click to view excerpt (Similarity: ${s.similarity})`;
                pill.addEventListener("click", () => openSourceModal(s));
                pillsList.appendChild(pill);
            });

            citationsWrap.appendChild(pillsList);
            bubble.appendChild(citationsWrap);
        } else if (mode === "general_ai") {
            const generalNotice = document.createElement("div");
            generalNotice.className = "general-ai-note";
            generalNotice.innerHTML = `<i class="fa-solid fa-circle-info"></i> <em>General AI Guidance (No matching college brochure text found)</em>`;
            bubble.appendChild(generalNotice);
        }

        // Message Footer Actions (Copy Button)
        const actionsBar = document.createElement("div");
        actionsBar.className = "message-actions-bar";

        const copyBtn = document.createElement("button");
        copyBtn.className = "msg-action-btn";
        copyBtn.innerHTML = `<i class="fa-regular fa-copy"></i> Copy`;
        copyBtn.title = "Copy response";
        copyBtn.addEventListener("click", () => copyTextToClipboard(content, copyBtn));

        actionsBar.appendChild(copyBtn);
        bubble.appendChild(actionsBar);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    feed.appendChild(row);
}

function appendErrorMessageToUI(errorText, retryQuery = null) {
    const feed = document.getElementById("messages-feed");
    if (!feed) return;

    const row = document.createElement("div");
    row.className = "message-row message-assistant";

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar bot-avatar";
    avatar.innerHTML = `<i class="fa-solid fa-robot"></i>`;

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble error-bubble";

    const errorContent = document.createElement("div");
    errorContent.className = "markdown-body";
    const rawHtml = marked.parse(errorText);
    errorContent.innerHTML = DOMPurify.sanitize(rawHtml);
    bubble.appendChild(errorContent);

    if (retryQuery) {
        const retryBtn = document.createElement("button");
        retryBtn.className = "btn btn-sm btn-outline btn-retry";
        retryBtn.innerHTML = `<i class="fa-solid fa-rotate-right"></i> Retry Question`;
        retryBtn.addEventListener("click", () => {
            row.remove();
            submitMessage(retryQuery);
        });
        bubble.appendChild(retryBtn);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);
    feed.appendChild(row);
}

// ============================================================================
// UI STATE HELPERS
// ============================================================================
function showWelcomeView() {
    const welcome = document.getElementById("chat-welcome-view");
    const feed = document.getElementById("messages-feed");
    const suggestions = document.getElementById("post-response-suggestions");
    if (welcome) welcome.style.display = "flex";
    if (feed) feed.style.display = "none";
    if (suggestions) suggestions.style.display = "none";
    renderSuggestionCards(defaultSuggestedQuestions);
}

function showMessagesFeed() {
    const welcome = document.getElementById("chat-welcome-view");
    const feed = document.getElementById("messages-feed");
    if (welcome) welcome.style.display = "none";
    if (feed) feed.style.display = "flex";
    const suggestions = document.getElementById("post-response-suggestions");
    if (suggestions) suggestions.style.display = "none";
}

function showResponseSuggestions() {
    const suggestions = document.getElementById("post-response-suggestions");
    if (suggestions) {
        suggestions.style.display = "block";
    }
    renderSuggestionCards(defaultSuggestedQuestions);
}

function setGeneratingState(generating) {
    isGenerating = generating;
    const sendBtn = document.getElementById("send-btn");
    const userInput = document.getElementById("user-input");
    const indicator = document.getElementById("typing-indicator-row");

    if (indicator) indicator.style.display = generating ? "flex" : "none";
    if (sendBtn) sendBtn.disabled = generating || (userInput && userInput.value.trim().length === 0);
    if (userInput) userInput.disabled = generating;

    if (!generating && userInput) {
        userInput.focus();
    }
}

function updateActiveModePill(mode) {
    const pill = document.getElementById("active-mode-pill");
    const text = document.getElementById("active-mode-text");
    if (!pill || !text) return;

    if (mode === "rag") {
        pill.className = "mode-pill pill-rag";
        pill.innerHTML = `<i class="fa-solid fa-file-shield"></i> <span>Document Grounded</span>`;
    } else {
        pill.className = "mode-pill pill-general";
        pill.innerHTML = `<i class="fa-solid fa-brain"></i> <span>General AI Guidance</span>`;
    }
}

function adjustTextareaHeight(textarea) {
    textarea.style.height = "auto";
    const newHeight = Math.min(textarea.scrollHeight, 180);
    textarea.style.height = `${newHeight}px`;
}

function scrollToBottom(force = false) {
    const container = document.getElementById("chat-messages-container");
    if (!container) return;

    const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 140;
    if (force || nearBottom || shouldAutoScroll) {
        container.scrollTop = container.scrollHeight;
    }
}

const chatMessagesContainer = document.getElementById("chat-messages-container");
if (chatMessagesContainer) {
    chatMessagesContainer.addEventListener("scroll", () => {
        const nearBottom = chatMessagesContainer.scrollHeight - chatMessagesContainer.scrollTop - chatMessagesContainer.clientHeight < 140;
        shouldAutoScroll = nearBottom;
    });
}

function toggleSidebar(open) {
    const sidebar = document.getElementById("chat-sidebar");
    const backdrop = document.getElementById("sidebar-backdrop");
    if (!sidebar) return;

    if (open) {
        sidebar.classList.add("open");
        if (backdrop) backdrop.classList.add("show");
    } else {
        sidebar.classList.remove("open");
        if (backdrop) backdrop.classList.remove("show");
    }
}

function renderSuggestionCards(questions) {
    const cardsToRender = Array.isArray(questions) && questions.length ? questions : defaultSuggestedQuestions;
    const cardSets = document.querySelectorAll(".suggestions-grid");

    cardSets.forEach((grid) => {
        grid.innerHTML = "";
        cardsToRender.forEach((question, index) => {
            const card = document.createElement("button");
            card.type = "button";
            card.className = "suggestion-card";
            card.setAttribute("data-question", question);
            card.setAttribute("aria-label", `Ask: ${question}`);

            const iconMap = [
                "fa-solid fa-laptop-code",
                "fa-solid fa-timeline",
                "fa-solid fa-circle-check",
                "fa-solid fa-receipt",
                "fa-solid fa-file-lines",
                "fa-solid fa-hand-holding-dollar"
            ];
            const labelMap = [
                "Engineering Branches",
                "Admission Process",
                "Eligibility Criteria",
                "Fee Structure",
                "Required Documents",
                "Scholarships & Aid"
            ];

            card.innerHTML = `
                <div class="card-icon-wrap"><i class="${iconMap[index % iconMap.length]}"></i></div>
                <div class="card-text-wrap">
                    <strong>${labelMap[index % labelMap.length]}</strong>
                    <span>${question}</span>
                </div>
                <i class="fa-solid fa-arrow-right card-arrow"></i>
            `;

            card.addEventListener("click", () => {
                if (isGenerating) return;
                submitMessage(question);
            });

            grid.appendChild(card);
        });
    });
}

function copyTextToClipboard(text, btnElement) {
    navigator.clipboard.writeText(text).then(() => {
        const originalHtml = btnElement.innerHTML;
        btnElement.innerHTML = `<i class="fa-solid fa-check text-success"></i> Copied!`;
        setTimeout(() => {
            btnElement.innerHTML = originalHtml;
        }, 2000);
    }).catch(() => {
        alert("Failed to copy text.");
    });
}

function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/[&<>"']/g, function (m) {
        return {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        }[m];
    });
}

// ============================================================================
// CITATION & SETUP MODALS
// ============================================================================
function initModals() {
    // Source Modal
    const sourceModal = document.getElementById("source-modal");
    const sourceCloseBtn = document.getElementById("modal-close-btn");
    const sourceDoneBtn = document.getElementById("modal-done-btn");

    const closeSourceModal = () => {
        if (sourceModal) sourceModal.style.display = "none";
    };

    if (sourceCloseBtn) sourceCloseBtn.addEventListener("click", closeSourceModal);
    if (sourceDoneBtn) sourceDoneBtn.addEventListener("click", closeSourceModal);
    if (sourceModal) {
        sourceModal.addEventListener("click", (e) => {
            if (e.target === sourceModal) closeSourceModal();
        });
    }

    // Setup Help Modal
    const helpModal = document.getElementById("setup-help-modal");
    const helpCloseBtn = document.getElementById("setup-modal-close-btn");
    const helpOkBtn = document.getElementById("setup-help-ok-btn");
    const helpTrigger = document.getElementById("offline-setup-info-btn");

    const closeHelpModal = () => {
        if (helpModal) helpModal.style.display = "none";
    };

    if (helpTrigger) {
        helpTrigger.addEventListener("click", () => {
            if (helpModal) helpModal.style.display = "flex";
        });
    }
    if (helpCloseBtn) helpCloseBtn.addEventListener("click", closeHelpModal);
    if (helpOkBtn) helpOkBtn.addEventListener("click", closeHelpModal);
    if (helpModal) {
        helpModal.addEventListener("click", (e) => {
            if (e.target === helpModal) closeHelpModal();
        });
    }
}

function openSourceModal(source) {
    const modal = document.getElementById("source-modal");
    const titleEl = document.getElementById("modal-doc-title");
    const metaEl = document.getElementById("modal-doc-meta");
    const fileEl = document.getElementById("modal-doc-file");
    const scoreEl = document.getElementById("modal-doc-score");
    const textEl = document.getElementById("modal-passage-text");

    if (!modal) return;

    if (titleEl) titleEl.textContent = source.document_name || "Document Citation";
    if (metaEl) metaEl.textContent = `Page ${source.page || 1} • Similarity: ${source.similarity || "N/A"}`;
    if (fileEl) fileEl.textContent = source.document_name || "Unknown";
    if (scoreEl) scoreEl.textContent = `${source.similarity || "N/A"} (Cosine Similarity)`;
    if (textEl) textEl.textContent = source.snippet || source.text || "No snippet available.";

    modal.style.display = "flex";
}
