# CampusBot: An Intelligent College Admission Assistant Using Large Language Models

> **Engineering College Microproject**  
> **Domain:** Artificial Intelligence, Natural Language Processing, Full-Stack Web Development  
> **AI Engine:** 100% Free & Local Inference via **Ollama** (`llama3.2:3b` & `nomic-embed-text`) — **Zero Paid APIs, Zero Cloud Keys, Zero Recurring Costs**

---

## 1. Project Overview

### 1.1 Problem Statement
Prospective students and their parents face significant friction navigating fragmented college websites, voluminous brochures, and outdated admission PDFs. Common inquiries regarding eligibility benchmarks, engineering branches, tuition fees, required documentation, scholarship schemes, and campus amenities frequently overwhelm academic admission offices with repetitive queries. Moreover, relying on commercial cloud LLMs (such as OpenAI or Anthropic) introduces recurring API billing costs, privacy concerns with sensitive institutional data, and API key dependencies.

### 1.2 Proposed Solution
**CampusBot** is an end-to-end, privacy-focused college admission assistant powered by locally hosted Large Language Models and **Retrieval-Augmented Generation (RAG)**. 
- All AI inference is processed locally on the host machine using **Ollama** (`llama3.2:3b`).
- Document retrieval uses a self-contained, offline **TF-IDF** embedding space (no external embedding model required), so the knowledge base works even when Ollama's neural embedding model is not installed.
- College prospectuses (PDF, DOCX, TXT) are parsed, chunked, and vectorized into a persistent local vector database.
- The six starter questions (and every custom question) are answered through the **real** retrieval + local LLM pipeline — there are **no hardcoded answers**.
- Responses to institutional questions are strictly grounded in the indexed college documents, providing exact source filename citations and page numbers.
- When the documents do not contain the answer, the assistant replies with a clear fallback instead of inventing information.

### 1.3 Project Objectives
1. **Instant Automated Guidance:** Deliver 24/7 natural-language answers to college admission inquiries.
2. **Grounded RAG Accuracy:** Eliminate LLM hallucinations by retrieving and citing facts directly from official college documents.
3. **100% Free & Local Execution:** Guarantee zero dependency on paid cloud LLM subscriptions or external API keys.
4. **Data Privacy:** Keep student conversations and academic records strictly within the local host environment.
5. **Demonstrable Engineering Artifact:** Provide a complete full-stack web application suitable for academic demonstration and project viva examination.

---

## 2. System Architecture

```text
                                  +------------------------------------+
                                  |         Prospective Student        |
                                  +-----------------+------------------+
                                                    | Web Browser
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                        FRONTEND WEB INTERFACE                                         |
|  - Modern SaaS Landing Page (index.html): Hero, Features, 5-Step Workflow, Benefits, Viva About     |
|  - Dedicated Chat Interface (chat.html): SQLite Conversations, Markdown, Citations Modal, Live Status |
|  - Admin Dashboard (admin.html): PDF/TXT Ingestion, Chunks Counter, Vector Reindexing                 |
+---------------------------------------------------+---------------------------------------------------+
                                                    | Fetch API (JSON / HTTP)
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                        FLASK BACKEND (app.py)                                         |
|  - Modular Blueprints: /api/chat | /api/status | /api/conversations | /api/documents                  |
|  - Input Validation, Sanitization, Character Limits, Error Boundaries                                 |
+------------------------------------+---------------------------------------------+--------------------+
                                     |                                             |
                                     v                                             v
              +------------------------------------+             +------------------------------------+
              |       SQLite Local Database        |             |      Document Ingestion Pipeline   |
              |         (campusbot.db)             |             |     (scripts/ingest_documents.py)  |
              |  - conversations & messages tables |             |  - PyMuPDF / pypdf text extraction |
              |  - Cascading multi-thread history  |             |  - Semantic chunking with overlap  |
              +------------------------------------+             +-----------------+------------------+
                                                                                   |
                                     +---------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------------------------------------+
|                                        RETRIEVAL-AUGMENTED GENERATION                                 |
|  - Embedding Engine (services/embedding_service.py): Generates dense semantic vectors                |
|  - Vector Database (vector_store/index.pkl): Cosine similarity search over indexed chunks             |
|  - Strict Prompt Assembler: Injects verified excerpts with institutional safety guardrails            |
+---------------------------------------------------+---------------------------------------------------+
                                                    | Local HTTP API (Port 11434)
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                           OLLAMA LOCAL RUNTIME                                        |
|  - Chat Model: llama3.2:3b (Meta LLaMA 3.2 lightweight 3-billion parameter model)                     |
|  - Embedding Model: nomic-embed-text (High-performance dense vector embedder)                         |
|  - 100% Free Edge Inference on Host CPU/GPU (No Internet Required After Download)                      |
+-------------------------------------------------------------------------------------------------------+
```

---

## 3. Technical Concepts & Viva Preparation Guide

### 3.1 What is a Large Language Model (LLM)?
An LLM is a deep neural network based on the **Transformer architecture** (specifically decoder-only autoregressive models). It has been pretrained on massive corpora to predict the next token given a preceding sequence of words. In CampusBot, we utilize Meta's **LLaMA 3.2 (3B)** model running via Ollama. It features 3.2 billion parameters, offering high conversational fluency and reasoning capabilities while maintaining a compact memory footprint (~2.0 GB RAM/VRAM) ideal for local execution.

### 3.2 What is Retrieval-Augmented Generation (RAG)?
Standard LLMs suffer from two main flaws:
1. **Knowledge Cutoffs:** They only know information present up to their pre-training date.
2. **Hallucinations:** When asked about proprietary or institution-specific facts (e.g., this semester's cutoff marks or hostel security deposit), base models may invent convincing but false statements.

**RAG solves this by combining retrieval with generation:**
1. **Retrieve:** When a student asks a question, CampusBot converts the query into a numerical vector and retrieves the top-$k$ most similar chunks from official college documents.
2. **Augment:** CampusBot constructs a prompt containing the student's question and the retrieved excerpts.
3. **Generate:** The local LLM generates an answer synthesized exclusively from the provided excerpts, citing the exact document name.

### 3.3 What are Embeddings & How Does Vector Search Work?
An **embedding** is a vector of floating-point numbers that maps words or sentences into a continuous high-dimensional geometric space. Texts with similar semantic meanings are placed close to each other.
- **Cosine Similarity Formula:**
  $$\text{Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
- When vectors are normalized to unit length ($\|\mathbf{A}\| = 1$), cosine similarity simplifies directly to the dot product $\mathbf{A} \cdot \mathbf{B}$, enabling fast similarity comparisons.

### 3.4 General AI vs. Document-Grounded RAG Responses
| Feature | General AI Mode | Document-Grounded RAG Mode |
| :--- | :--- | :--- |
| **Information Source** | Pretrained neural weights | Verified local college documents |
| **Citations** | None | Exact source filename & page number |
| **Risk of Hallucination** | Moderate | Extremely Low |
| **Institutional Accuracy** | Generic educational advice | Precise fee schedules, dates, & criteria |
| **Active Indicator** | Purple UI Badge (`General AI Guidance`) | Emerald UI Badge (`Document Grounded`) |

---

## 4. Project Directory Structure

```text
CampusBot/
│
├── app.py                      # Main Flask application with blueprint routing
├── config.py                   # Centralized configuration class & environment settings
├── requirements.txt            # Python package dependencies
├── README.md                   # Complete documentation & Windows setup guide
├── .gitignore                  # Git exclusion rules
├── .env                        # Local configuration file (100% free, no API keys)
├── .env.example                # Configuration template
│
├── routes/                     # Modular Flask Blueprints
│   ├── __init__.py             # Exports blueprints
│   ├── chat.py                 # POST /api/chat endpoint
│   ├── status.py               # GET /api/status system health diagnostics
│   ├── conversations.py        # SQLite conversation thread endpoints
│   └── documents.py            # Document upload, deletion, & reindexing
│
├── services/                   # Core business logic & AI orchestration
│   ├── __init__.py             # Exports service singletons
│   ├── ollama_service.py       # Ollama HTTP API client & model status checks
│   ├── rag_service.py          # RAG pipeline & PersistentVectorStore
│   ├── embedding_service.py    # Local Ollama & deterministic embedding generator
│   └── document_processor.py   # PDF (PyMuPDF/pypdf) & TXT text chunker
│
├── database/                   # SQLite persistence layer
│   ├── __init__.py
│   ├── db.py                   # Schema initialization, CRUD for chats/messages
│   └── campusbot.db            # SQLite database file
│
├── scripts/                    # Command-line utility scripts
│   └── ingest_documents.py     # Document indexing & vector store builder
│
├── templates/                  # Frontend HTML5 Jinja2 templates
│   ├── index.html              # Comprehensive modern SaaS landing page
│   ├── chat.html               # Dedicated full-screen chatbot interface
│   └── admin.html              # Knowledge base administrator portal
│
├── static/                     # Frontend static assets
│   ├── css/
│   │   └── style.css           # Modern design system (Navy, Electric Blue, Violet)
│   └── js/
│       ├── chat.js             # Client-side chat manager & markdown renderer
│       └── main.js             # Landing page navigation & status checks
│
├── data/
│   └── documents/              # Storage directory for college PDFs and TXT files
│       ├── demo_admission_faq.txt   # Demonstration knowledge base with disclaimers
│       └── sample_brochure.txt      # Demonstration institutional brochure
│
├── vector_store/               # Persistent local vector indices
│   └── index.pkl               # Serialized document chunks and embeddings
│
└── tests/                      # Automated unit test suite
    ├── test_routes.py          # Route & API integration tests
    └── test_rag.py             # Document chunking & vector retrieval tests
```

---

## 5. Step-by-Step Windows Installation Guide

Follow these steps to set up and run CampusBot on a Windows computer.

### Step 1: Verify Python Installation
Open **Command Prompt** (`cmd`) or **PowerShell** and verify that Python 3.9+ is installed:
```powershell
python --version
```
*If not installed, download Python from [python.org](https://www.python.org/) and ensure **"Add Python to PATH"** is checked.*

### Step 2: Install Ollama on Windows
1. Download the official Windows installer from:  
   👉 **[https://ollama.com/download/windows](https://ollama.com/download/windows)**
2. Run `OllamaSetup.exe` and complete the installation.
3. Ollama will start automatically in the Windows system tray (an alpaca icon near your clock).

### Step 3: Download the Local AI Models
Open Command Prompt or PowerShell and download the chat model:
```powershell
ollama pull llama3.2:3b
```
*(Download size is ~2.0 GB. This requires an active internet connection once during download; subsequent AI inference runs entirely offline.)*

Next, download the local embedding model:
```powershell
ollama pull nomic-embed-text
```
*(Download size is ~274 MB.)*

### Step 4: Verify Ollama Installation
List all locally installed models:
```powershell
ollama list
```
You should see `llama3.2:3b` and `nomic-embed-text` listed.

To test the model interactively:
```powershell
ollama run llama3.2:3b
```
Type `Hello, what can you help me with?` and press Enter. Type `/bye` to exit.

### Step 5: Setup CampusBot Project & Virtual Environment
Navigate to the CampusBot folder on your computer:
```powershell
cd c:\Users\Hussey\OneDrive\Desktop\CampusBot
```
Create a virtual environment:
```powershell
python -m venv .venv
```
Activate the virtual environment:
```powershell
.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

Install dependencies:
```powershell
pip install -r requirements.txt
```

### Step 6: Prepare Sample Admission Documents
Ensure demonstration documents are present in `data/documents/`:
- `data/documents/demo_admission_faq.txt`
- `data/documents/sample_brochure.txt`

You can also place real college admission PDFs or prospectus files directly into `data/documents/`.

### Step 7: Build the RAG Vector Index
Run the ingestion script to process documents and generate vector embeddings:
```powershell
python scripts/ingest_documents.py --rebuild
```
**Expected Output:**
```text
=================================================================
CampusBot Local Document Ingestion Pipeline
=================================================================
Document Directory: data/documents
Vector Store Path:  vector_store
[PROCESS] Extracting and chunking 'demo_admission_faq.txt'...
   [SUCCESS] Successfully indexed 12 chunks from 'demo_admission_faq.txt'.
[PROCESS] Extracting and chunking 'sample_brochure.txt'...
   [SUCCESS] Successfully indexed 16 chunks from 'sample_brochure.txt'.
=================================================================
Ingestion Pipeline Completed Successfully!
 * Total Indexed Documents: 2
 * Total Vector Chunks:     28
=================================================================
```

### Step 8: Start the CampusBot Web Server
Start the Flask application:
```powershell
python app.py
```
Open your browser and navigate to:  
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

- **Landing Page:** `http://127.0.0.1:5000/`
- **Dedicated Chatbot:** `http://127.0.0.1:5000/chat`
- **Admin Dashboard:** `http://127.0.0.1:5000/admin`

---

## 6. Running Automated Tests

CampusBot includes a comprehensive test suite covering route integration, input validation, SQLite conversation management, text chunking, and semantic vector similarity search:

Run the entire test suite:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
All tests execute locally and pass with `OK`.

---

## 7. Sample Questions for Demonstration & Viva

The chat welcome screen shows exactly six clickable starter questions. Selecting any of them (or typing your own) sends the question to the real RAG + local LLM pipeline:

1. **What engineering branches are available?**
2. **What is the admission process?**
3. **What are the eligibility criteria?**
4. **What is the fee structure?**
5. **What documents are required for admission?**
6. **What scholarships and financial aid are available?**

When presenting CampusBot to your professor or viva examiner, test these queries to demonstrate RAG and citation grounding:

1. **Required Documents:**
   > *"What documents are required for admission verification?"*  
   *Result:* CampusBot cites `demo_admission_faq.txt` and lists 10th/12th marksheets, Transfer Certificate, Migration Certificate, and photo IDs.

2. **Engineering Branches:**
   > *"What engineering branches are available?"*  
   *Result:* Lists Computer Science & Engineering (CSE), Information Technology (IT), ECE, and Mechanical Engineering.

3. **Admission Workflow:**
   > *"Explain the admission process step by step."*  
   *Result:* Summarizes registration, merit list verification, entrance score cutoffs, and provisional fee deposits.

4. **Scholarship Guidance:**
   > *"What scholarships can students apply for?"*  
   *Result:* Retrieves merit-based fee concessions and need-based financial aid criteria.

5. **Missing Information Guardrail (Hallucination Prevention):**
   > *"What is the exact seat count for the 2030 Aeronautical Engineering batch?"*  
   *Result:* The bot politely clarifies that this specific information is not available in the indexed knowledge base and advises checking the official university admission office.

---

## 8. System Limitations & Future Enhancements

### Current Limitations
- **Local Compute Bound:** Inference latency depends on host CPU/GPU speed. On standard laptops without dedicated GPUs, 3B model token generation takes 2 to 6 seconds.
- **Single-Host Scoping:** Designed as a localized student microproject. For multi-tenant production deployments, a Redis cache, load balancer, and multi-user authentication layer would be recommended.

### Future Enhancements
- **Voice Input / Speech-to-Text:** Integration with Whisper.cpp for offline voice-based student queries.
- **Multilingual Support:** Support for regional Indian languages (Hindi, Marathi, Tamil, etc.) using multilingual local models.
- **Automated Web Scraping:** Scheduled crawlers to automatically update college admission circulars from university websites.

---

## 9. Viva Questions & Model Answers

**Q1: Why did you choose Ollama instead of OpenAI's API?**  
*Answer:* Ollama enables 100% free, private, offline execution directly on the host computer. Commercial APIs require paid credit cards, charge per input/output token, and transmit sensitive student documents over the internet.

**Q2: What happens if a student asks a question not present in the uploaded PDFs?**  
*Answer:* The RAG pipeline checks the cosine similarity score of retrieved passages against a similarity threshold. If no passages match, CampusBot provides general guidance and explicitly informs the student that the information was not found in the verified documents, preventing hallucinated answers.

**Q3: How does the chunking overlap help?**  
*Answer:* If an important concept or sentence is split exactly at the chunk boundary, context could be lost. An overlap of 120 characters ensures semantic continuity between adjacent chunks.

**Q4: How are conversations stored?**  
*Answer:* In a local SQLite database (`campusbot.db`) with relational foreign keys linking conversations and messages, supporting full conversation creation, retrieval, and deletion.

---

*Project developed for Academic Evaluation • 100% Free & Open Source AI*
