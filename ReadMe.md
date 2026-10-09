# Vaan: Local-First, Multi-Modal AI Voice Assistant

### The Problem (Why I Built This)
I built Vaan because I was personally frustrated by the friction of context-switching. As a developer, I constantly jumped between reading local PDFs/notes, checking weather/APIs, searching the web, and managing media. 

Existing voice assistants failed me for two reasons:
1. **Lack of Local Context:** They couldn't answer questions about my personal documents without uploading them to a third-party cloud.
2. **Rigidity & Cost:** Setting up custom, multi-step workflows required complex, expensive cloud infrastructure or brittle, hardcoded scripts.

Nobody asked me to build this. I built it because the inefficiency bothered me, and I wanted to prove that a modular, locally-aware architecture could replace hours of manual digital work with a single, natural voice command.

### The Solution (Measurable Value)
Vaan is a "one-person company" project: a fully local-first, multi-modal AI voice assistant that I own from end to end. It provides measurable value by:
- **Eliminating context-switching:** Seamlessly transitions between conversational LLM queries, local document Q&A, and real-time API tool calls (weather, web search, Wikipedia).
- **Guaranteeing privacy and zero infrastructure cost:** All document processing and vectorization happen locally on the machine. No personal data is sent to external embedding APIs.
- **Reducing latency:** By optimizing the RAG pipeline for local execution, query-to-speech response times are kept to a minimum, creating a natural conversational flow.

### Architecture & Trade-offs (The "Why" Behind the Stack)
I chose technologies based on the constraints of a lightweight, personal MVP, not just "industry hype."
- **Speech-to-Text:** `openai-whisper` (local `base` model) for reliable, offline transcription without API costs.
- **Text-to-Speech:** `edge-tts` for high-quality, natural-sounding neural voices (with automatic Hindi/English language detection) without heavy local TTS model downloads.
- **LLM & Tool Routing:** `OpenRouter` API with strict, allow-listed JSON tool definitions. This prevents hallucinated tool calls and ensures the LLM only delegates to safe, predefined functions (Weather, Web Search, Wikipedia, Local Docs).
- **RAG Pipeline (The Hard Trade-off):** I used `scikit-learn` (TF-IDF + Cosine Similarity) for local document retrieval. 
  - *The AI's Suggestion:* My AI coding assistant confidently suggested setting up a heavy, managed vector database (like Pinecone) with LangChain and external embedding APIs, calling it the "industry standard."
  - *My Decision:* I rejected this. The AI lacked the situational awareness of my actual constraints: I needed a zero-infrastructure-cost, low-latency, fully offline MVP. Relying on external embedding APIs would introduce latency, cost, and a point of failure. I overruled the AI and directed it to implement the lightweight, in-memory TF-IDF approach. This eliminated external API costs for embeddings, reduced retrieval latency to milliseconds, and kept the entire stack runnable offline.

### How I Drove the AI
In line with modern engineering practices, I used AI aggressively, but as a tool, not a decision-maker. 
- I delegated boilerplate generation (e.g., API parsing, regex extraction) to the AI.
- I acted as the architect, defining the strict `TOOL_DEFINITIONS` schema to prevent the LLM from hallucinating actions.
- I caught the AI when it suggested over-engineered solutions (like the external vector database) and forced it to adapt to my local-first constraints.

> **⚠️ IMPORTANT FOR REVIEWERS:** 
> Per best practices in AI-driven development, **my actual AI working sessions are more important than the final code.** 
> Please see the [`ai-working-sessions/`](./ai-working-sessions) directory in this repository. It contains exported transcripts of my chats with AI coding assistants, highlighting where I pushed back on its output, caught its mistakes, and directed it to the final, optimized solution.

---

### Features
- 🎙️ **Natural Voice I/O:** Whisper STT + Edge TTS (supports English, Hindi, and Hinglish).
- 🧠 **Conversational Memory:** Maintains a rolling 12-message context window for natural follow-up questions.
- 📄 **Local RAG:** Indexes and answers questions from `.txt`, `.md`, and `.pdf` files in the `./docs` directory.
- 🛠️ **Safe Tool Routing:** Dynamic, allow-listed function calling for Weather (Open-Meteo), Web Search, Wikipedia, and Media Control.
- 🛡️ **Graceful Degradation:** If a tool fails or the AI hallucinates a command, the system safely falls back to standard conversational LLM response without crashing.

### Setup & Installation
1. Clone the repository and navigate to the folder.
2. Install dependencies:
   ```bash
   pip install SpeechRecognition PyAudio openai-whisper pywin32 requests openai pywhatkit wikipedia pyjokes imdbpy scikit-learn pypdf asyncio pygame edge-tts
