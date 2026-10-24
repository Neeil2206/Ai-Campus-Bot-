// Wait for DOM to load
document.addEventListener("DOMContentLoaded", () => {
    // Check and initialize status indicators in the header
    initHeaderStatus();

    // Mobile Sidebar Navigation
    initMobileNav();

    // Initialize Page Specific Scripts
    if (document.getElementById("chat-form")) {
        initChatPage();
    }
    
    if (document.getElementById("upload-form")) {
        initAdminPage();
    }
});

// ==========================================================================
// HEADER STATUS INDICATOR
// ==========================================================================
async function initHeaderStatus() {
    const statusDot = document.querySelector(".status-dot");
    const statusText = document.getElementById("header-status-text");
    const setupBanner = document.getElementById("setup-notice-banner");

    try {
        const response = await fetch("/api/status");
        if (!response.ok) throw new Error();
        const data = await response.json();

        statusDot.className = "status-dot"; // reset
        if (data.ai_available === true) {
            statusDot.classList.add("status-online");
            statusText.textContent = "CampusBot AI • Ollama";
            if (setupBanner) setupBanner.style.display = "none";
        } else {
            statusDot.classList.add("status-setup");
            statusText.textContent = "Local AI Offline";
            if (setupBanner) {
                setupBanner.style.display = "flex";
                const bannerSpan = setupBanner.querySelector("span");
                if (bannerSpan) {
                    bannerSpan.innerHTML = '<strong>Local AI is offline:</strong> Please start Ollama and ensure the local service is running.';
                }
            }
            
            // If on chat page, show notice in the chat area too
            const welcomeSubtitle = document.querySelector(".welcome-subtitle");
            if (welcomeSubtitle) {
                welcomeSubtitle.innerHTML = '<span class="response-error-notice">Local AI is offline. Please start Ollama.</span>';
            }
        }
    } catch (err) {
        statusDot.className = "status-dot status-setup";
        statusText.textContent = "Status Error";
    }
}

// ==========================================================================
// MOBILE MENU NAVIGATION
// ==========================================================================
function initMobileNav() {
    const toggleBtn = document.getElementById("mobile-menu-toggle");
    const sidebar = document.getElementById("app-sidebar");

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            sidebar.classList.toggle("show");
        });

        // Close sidebar when clicking main screen
        document.addEventListener("click", (e) => {
            if (sidebar.classList.contains("show") && !sidebar.contains(e.target) && e.target !== toggleBtn) {
                sidebar.classList.remove("show");
            }
        });
    }
}

// ==========================================================================
// CHAT PAGE FUNCTIONALITY
// ==========================================================================
function initChatPage() {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const chatMessagesBox = document.getElementById("chat-messages-box");
    const welcomeContainer = document.getElementById("chat-welcome-container");
    const thinkingIndicator = document.getElementById("thinking-indicator");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    const suggestionBtns = document.querySelectorAll(".suggest-btn");
    const welcomeCardBtns = document.querySelectorAll(".welcome-card-btn");

    // Modal elements for citations
    const modalOverlay = document.getElementById("source-modal-overlay");
    const modalDocName = document.getElementById("modal-doc-name");
    const modalDocPage = document.getElementById("modal-doc-page");
    const modalChunkText = document.getElementById("modal-chunk-text");
    const modalClose = document.getElementById("source-modal-close");
    const modalOk = document.getElementById("source-modal-ok");

    // Configure marked.js options for safe rendering
    marked.setOptions({
        breaks: true,
        sanitize: true
    });

    // Auto-growing textarea logic
    userInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
        if (this.scrollHeight > 120) {
            this.style.overflowY = "scroll";
        } else {
            this.style.overflowY = "hidden";
        }
    });

    // Enter to submit keydown handler
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Form submission listener
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        sendStudentQuery(text);
    });

    // Bind sidebar clicks
    suggestionBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const question = btn.getAttribute("data-question");
            if (question) sendStudentQuery(question);
        });
    });

    // Bind welcome cards clicks
    welcomeCardBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const question = btn.getAttribute("data-question");
            if (question) sendStudentQuery(question);
        });
    });

    // Clear chat listener
    clearChatBtn.addEventListener("click", async () => {
        if (confirm("Are you sure you want to clear your current conversation history?")) {
            try {
                const response = await fetch("/api/clear", { method: "POST" });
                const result = await response.json();
                if (result.status === "success") {
                    // Empty chat messages
                    chatMessagesBox.innerHTML = "";
                    chatMessagesBox.style.display = "none";
                    welcomeContainer.style.display = "flex";
                }
            } catch (err) {
                console.error("Error clearing conversation:", err);
                alert("Failed to clear conversation. Please try again.");
            }
        }
    });

    // Core RAG chat submit routine
    async function sendStudentQuery(queryText) {
        // 1. Hide welcome container and show message box
        welcomeContainer.style.display = "none";
        chatMessagesBox.style.display = "flex";

        // 2. Append User bubble
        appendMessage("user", queryText);

        // 3. Clear, resize, and focus inputs
        userInput.value = "";
        userInput.style.height = "auto";
        userInput.disabled = true;
        document.getElementById("send-btn").disabled = true;

        // 4. Show thinking indicator
        thinkingIndicator.style.display = "flex";
        scrollToBottom();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: queryText })
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const data = await response.json();
            
            // Hide indicator
            thinkingIndicator.style.display = "none";
            userInput.disabled = false;
            document.getElementById("send-btn").disabled = false;
            userInput.focus();

            // 5. Append Bot response with source payload
            appendMessage("bot", data.response, data.sources);
        } catch (err) {
            console.error("Chat error:", err);
            thinkingIndicator.style.display = "none";
            userInput.disabled = false;
            document.getElementById("send-btn").disabled = false;
            
            appendMessage("bot", "I'm unable to generate a response right now. Please try again in a moment.");
        }
        
        scrollToBottom();
    }

    // Creates formatted messages bubbles
    function appendMessage(role, rawContent, sources = []) {
        const wrapper = document.createElement("div");
        wrapper.classList.add("message-wrapper", role);

        const bubble = document.createElement("div");
        bubble.classList.add("message-bubble");

        // Parse markdown text using marked.js
        try {
            bubble.innerHTML = marked.parse(rawContent);
        } catch (e) {
            bubble.textContent = rawContent;
        }

        // Apply RAG ground verification indicators
        if (role === "bot" && sources && sources.length > 0) {
            // Check if it is the fallback response (if so, do not display Grounded badge)
            const isFallback = rawContent.includes("I couldn't find this information in the official college documents");
            
            if (!isFallback) {
                // Add Grounded check badge
                const groundedPill = document.createElement("div");
                groundedPill.className = "grounded-badge";
                groundedPill.innerHTML = '<i class="fa-solid fa-circle-check"></i> Grounded in official documents';
                bubble.appendChild(groundedPill);

                // Add expandable citations badges list
                const citationListDiv = document.createElement("div");
                citationListDiv.className = "source-citations-list";
                citationListDiv.innerHTML = '<span class="citation-header-label">📚 Source:</span>';

                sources.forEach((src) => {
                    const btn = document.createElement("button");
                    btn.className = "citation-btn";
                    btn.textContent = `${src.document_name} — Page ${src.page}`;
                    btn.title = `Click to inspect grounded segment (Score: ${src.similarity})`;
                    
                    // Bind click event to trigger Grounded segment modal drawer
                    btn.addEventListener("click", () => {
                        showSourceModal(src.document_name, src.page, src.text);
                    });
                    
                    citationListDiv.appendChild(btn);
                });
                
                bubble.appendChild(citationListDiv);
            }
        }

        const meta = document.createElement("div");
        meta.classList.add("message-meta");
        
        const sender = role === "user" ? "You" : "CampusBot";
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        meta.textContent = `${sender} • ${time}`;

        wrapper.appendChild(bubble);
        wrapper.appendChild(meta);
        chatMessagesBox.appendChild(wrapper);
        
        scrollToBottom();
    }

    // Modal controls for citations inspection
    function showSourceModal(docName, page, text) {
        modalDocName.textContent = docName;
        modalDocPage.textContent = `Page ${page}`;
        modalChunkText.textContent = text;
        modalOverlay.classList.add("show");
    }

    function hideSourceModal() {
        modalOverlay.classList.remove("show");
    }

    modalClose.addEventListener("click", hideSourceModal);
    modalOk.addEventListener("click", hideSourceModal);
    modalOverlay.addEventListener("click", (e) => {
        if (e.target === modalOverlay) hideSourceModal();
    });

    function scrollToBottom() {
        chatMessagesBox.scrollTop = chatMessagesBox.scrollHeight;
    }
}

// ==========================================================================
// ADMINISTRATOR PORTAL FUNCTIONALITY
// ==========================================================================
function initAdminPage() {
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const selectedFileInfo = document.getElementById("selected-file-info");
    const selectedFileName = document.getElementById("selected-file-name");
    const selectedFileSize = document.getElementById("selected-file-size");
    const removeFileBtn = document.getElementById("remove-file-btn");
    const uploadBtn = document.getElementById("upload-btn");
    const uploadStatus = document.getElementById("upload-status");
    const docTableBody = document.getElementById("documents-table-body");
    const checklistProgress = document.getElementById("ingestion-progress-checklist");

    let currentFile = null;

    // Fetch and populate document list and card stats
    refreshAdminDashboard();

    // Trigger browse file picker
    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag-over hover states
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    // Remove file listener
    removeFileBtn.addEventListener("click", () => {
        resetUploadForm();
    });

    function handleSelectedFile(file) {
        const allowedExts = ["pdf", "docx", "txt"];
        const ext = file.name.split(".").pop().toLowerCase();
        
        hideStatus();

        if (!allowedExts.includes(ext)) {
            showStatus("error", `Unsupported file format: .${ext}. Only PDF, DOCX, and TXT are allowed.`);
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showStatus("error", "File size exceeds 10MB limit. Please compress your file.");
            return;
        }

        currentFile = file;
        selectedFileName.textContent = file.name;
        
        const sizeKb = file.size / 1024;
        selectedFileSize.textContent = sizeKb > 1024 
            ? `${(sizeKb / 1024).toFixed(2)} MB` 
            : `${sizeKb.toFixed(1)} KB`;

        dropZone.style.display = "none";
        selectedFileInfo.style.display = "flex";
        uploadBtn.disabled = false;
    }

    // Submit file upload
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!currentFile) return;

        const formData = new FormData();
        formData.append("file", currentFile);

        // UI processing states
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Indexing...';
        hideStatus();
        
        // Show simulated ingestion checklist
        checklistProgress.style.display = "block";
        resetChecklistProgress();

        // Stage 1: Ingestion Start
        updateStep("step-upload", "active", '<i class="fa-solid fa-circle-notch fa-spin"></i>');

        // Step simulation triggers (RAG processing visual checkpoints)
        const timeouts = [];
        
        timeouts.push(setTimeout(() => {
            updateStep("step-upload", "done", '<i class="fa-solid fa-circle-check text-success"></i>');
            updateStep("step-extract", "active", '<i class="fa-solid fa-circle-notch fa-spin"></i>');
        }, 1000));

        timeouts.push(setTimeout(() => {
            updateStep("step-extract", "done", '<i class="fa-solid fa-circle-check text-success"></i>');
            updateStep("step-chunks", "active", '<i class="fa-solid fa-circle-notch fa-spin"></i>');
        }, 2200));

        timeouts.push(setTimeout(() => {
            updateStep("step-chunks", "done", '<i class="fa-solid fa-circle-check text-success"></i>');
            updateStep("step-embed", "active", '<i class="fa-solid fa-circle-notch fa-spin"></i>');
        }, 3600));

        timeouts.push(setTimeout(() => {
            updateStep("step-embed", "done", '<i class="fa-solid fa-circle-check text-success"></i>');
            updateStep("step-save", "active", '<i class="fa-solid fa-circle-notch fa-spin"></i>');
        }, 4800));

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            // Clear visual timeouts to sync up when server finishes faster or slower
            timeouts.forEach(t => clearTimeout(t));

            if (!response.ok) {
                throw new Error(result.error || `Upload failed with status ${response.status}`);
            }

            // Instantly complete all steps on success
            completeAllChecklistSteps();
            
            showStatus("success", `✓ Document processed successfully<br>✓ Text extracted<br>✓ Embeddings generated<br>✓ Added to knowledge base (${result.chunks} chunks).`);
            
            setTimeout(() => {
                resetUploadForm();
                refreshAdminDashboard();
            }, 1500);

        } catch (err) {
            console.error("Upload error:", err);
            timeouts.forEach(t => clearTimeout(t));
            failCurrentChecklistStep();
            showStatus("error", `Ingestion Failed: ${err.message}`);
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="fa-solid fa-arrow-up-from-bracket"></i> Ingest & Index Document';
        }
    });

    // Ingestion Checklist helpers
    function resetChecklistProgress() {
        const steps = ["step-upload", "step-extract", "step-chunks", "step-embed", "step-save"];
        steps.forEach(id => {
            const li = document.getElementById(id);
            li.className = "";
            li.querySelector(".check-bullet").innerHTML = '<i class="fa-regular fa-circle"></i>';
        });
    }

    function updateStep(id, status, iconHtml) {
        const li = document.getElementById(id);
        if (li) {
            li.className = status === "done" ? "step-done" : (status === "active" ? "step-active" : "");
            li.querySelector(".check-bullet").innerHTML = iconHtml;
        }
    }

    function completeAllChecklistSteps() {
        const steps = ["step-upload", "step-extract", "step-chunks", "step-embed", "step-save"];
        steps.forEach(id => {
            updateStep(id, "done", '<i class="fa-solid fa-circle-check text-success"></i>');
        });
    }

    function failCurrentChecklistStep() {
        const activeLi = document.querySelector(".checklist-items li.step-active");
        if (activeLi) {
            activeLi.className = "";
            activeLi.querySelector(".check-bullet").innerHTML = '<i class="fa-solid fa-circle-xmark text-danger"></i>';
        }
    }

    // Refresh Documents Table & Stats Dashboard
    async function refreshAdminDashboard() {
        fetchDocumentsList();
        
        try {
            const response = await fetch("/api/status");
            if (!response.ok) throw new Error();
            const data = await response.json();

            // Populate dashboard cards text
            const dbStatusText = document.getElementById("db-status-text");
            const dbDocsCount = document.getElementById("db-docs-count");
            const dbChunksCount = document.getElementById("db-chunks-count");
            const dbLastUpdated = document.getElementById("db-last-updated");

            if (data.vector_store.status === "connected") {
                dbStatusText.innerHTML = '<span class="badge status-connected">🟢 Connected</span>';
            } else {
                dbStatusText.innerHTML = '<span class="badge status-failed">🔴 Error</span>';
            }

            dbDocsCount.textContent = data.vector_store.documents_count;
            dbChunksCount.textContent = data.vector_store.chunks_count;
            
            // Display formatted date
            dbLastUpdated.textContent = data.vector_store.last_updated;
        } catch (err) {
            console.error("Error refreshing stats dashboard:", err);
        }
    }

    // Fetch documents list from API
    async function fetchDocumentsList() {
        try {
            const response = await fetch("/api/documents");
            if (!response.ok) throw new Error("Could not load documents");
            
            const data = await response.json();
            renderDocumentsTable(data.documents);
        } catch (err) {
            console.error("Fetch document error:", err);
            docTableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="empty-cell text-danger">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                        Failed to retrieve knowledge base documents list.
                    </td>
                </tr>
            `;
        }
    }

    // Render list table DOM
    function renderDocumentsTable(docs) {
        if (!docs || docs.length === 0) {
            docTableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="empty-cell">
                        <i class="fa-solid fa-folder-open"></i>
                        No college documents found in the database.
                    </td>
                </tr>
            `;
            return;
        }

        docTableBody.innerHTML = "";

        docs.forEach(doc => {
            const tr = document.createElement("tr");

            // Document Name Cell
            const nameTd = document.createElement("td");
            nameTd.classList.add("doc-name-cell");
            nameTd.innerHTML = `<i class="fa-regular fa-file-pdf text-danger" style="margin-right: 8px;"></i> ${doc.name}`;
            
            // Format icon based on extension
            const ext = doc.name.split('.').pop().toLowerCase();
            if (ext === 'docx') {
                nameTd.querySelector('i').className = 'fa-regular fa-file-word text-gold';
            } else if (ext === 'txt') {
                nameTd.querySelector('i').className = 'fa-regular fa-file-lines text-muted';
            }

            // Size Cell
            const sizeTd = document.createElement("td");
            sizeTd.textContent = `${doc.size_kb} KB`;

            // Chunks count Cell
            const chunksTd = document.createElement("td");
            chunksTd.textContent = doc.chunks;

            // Upload Date Cell
            const dateTd = document.createElement("td");
            dateTd.textContent = doc.upload_date;

            // Status Cell
            const statusTd = document.createElement("td");
            const statusClass = doc.status === "Indexed" ? "status-indexed" : "status-failed";
            statusTd.innerHTML = `<span class="badge ${statusClass}">${doc.status}</span>`;

            // Actions Cell
            const actionsTd = document.createElement("td");
            const deleteBtn = document.createElement("button");
            deleteBtn.classList.add("icon-btn-danger");
            deleteBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
            deleteBtn.title = "Delete Document";
            deleteBtn.addEventListener("click", () => triggerDeleteDocument(doc.name));
            actionsTd.appendChild(deleteBtn);

            tr.appendChild(nameTd);
            tr.appendChild(sizeTd);
            tr.appendChild(chunksTd);
            tr.appendChild(dateTd);
            tr.appendChild(statusTd);
            tr.appendChild(actionsTd);

            docTableBody.appendChild(tr);
        });
    }

    // Trigger delete document API
    async function triggerDeleteDocument(filename) {
        if (confirm(`Are you sure you want to permanently delete "${filename}"? This will remove all associated vector embeddings and text chunks.`)) {
            try {
                const response = await fetch(`/api/documents/delete/${encodeURIComponent(filename)}`, {
                    method: "DELETE"
                });

                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.error || "Failed to delete");
                }

                showStatus("success", `Successfully deleted and removed document: "${filename}".`);
                refreshAdminDashboard();
            } catch (err) {
                console.error("Deletion error:", err);
                showStatus("error", `Failed to delete file: ${err.message}`);
            }
        }
    }

    // Reset layout
    function resetUploadForm() {
        currentFile = null;
        fileInput.value = "";
        selectedFileInfo.style.display = "none";
        dropZone.style.display = "block";
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fa-solid fa-arrow-up-from-bracket"></i> Ingest & Index Document';
        checklistProgress.style.display = "none";
    }

    // Helpers
    function showStatus(type, message) {
        uploadStatus.className = `alert-status ${type}`;
        uploadStatus.innerHTML = message;
        uploadStatus.style.display = "block";
    }

    function hideStatus() {
        uploadStatus.style.display = "none";
    }
}
