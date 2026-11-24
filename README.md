# Study Chat

**A local document search and Q&A app.** Drop your notes and PDFs into a folder, ask a question in plain English, and Study Chat finds the passages that answer it and shows you exactly where they came from: the file name and, for PDFs, the page number. Everything runs on your own computer. Your documents never leave it.

I built it because hunting through a semester's worth of PDFs for one formula or definition was slow. Study Chat turns that into a few seconds.

![Study Chat answering a question from the sample notes, with its source passages expanded](docs/screenshot.png)

## What it does

- **Searches your own documents.** Reads PDF, TXT and Markdown files, splits them into passages, and finds the ones that match your question by meaning, not only by keywords.
- **Cites every source.** Each answer lists the passages it used, and you can expand them to read the original text. If your notes don't cover the question, it says so instead of guessing.
- **Reads text inside images.** Many lecture PDFs have formulas embedded as pictures. Study Chat runs OCR on those images and tags the results with their PDF page, so they're searchable too.
- **Organizes by course.** One folder per class. Search a single course or switch between them from a dropdown.
- **Two modes:**
  - **Explain from my notes** (default): answers questions using only what's in your documents.
  - **Solve a problem**: walks through a worked solution step by step, using the methods and formulas from your notes, then shows how to check the answer on a TI-84. The math comes from built-in calculation tools rather than the language model, so the numbers are reliable.
- **Fast updates.** Study Chat fingerprints each file and only processes notes that are new or changed, so updating the index after adding one PDF takes seconds instead of minutes.
- **Formatted math.** LaTeX formulas in answers are drawn with [KaTeX](https://katex.org) 0.19.1 (MIT license), which is included in `vendor/katex`, so nothing loads from the internet.
- **Model picker.** Choose which local model writes the answers. Qwen 27B (`qwen/qwen3.8-27b`) is the default.
- **100% offline.** No API keys, no accounts, no cloud services.

## How it works

Study Chat is a **retrieval-augmented generation (RAG)** app:

1. **Ingest** (`ingest.py`): reads every file in `notes/<course>/`, extracts the text (plus OCR text from PDF images via Tesseract), and splits it into passages tagged with course, file and page.
2. **Embed and store:** each passage is turned into a vector with the Nomic Embed v1.5 model, running on an NVIDIA GPU through PyTorch CUDA, and saved in a local ChromaDB database.
3. **Update incrementally:** each PDF is fingerprinted (SHA-256), so unchanged files reuse their cached text and OCR. Only new or changed passages are embedded, and passages from deleted files are removed. The database is labeled with the embedding model that built it, so switching models triggers a full rebuild instead of mixing incompatible vectors. An update with no changes takes seconds.
4. **Retrieve (hybrid search):** passages from the selected course are ranked by meaning (your question is embedded the same way and compared in ChromaDB) and by exact words (BM25 keyword search), then merged with weighted reciprocal rank fusion. The part of the question that asks for something counts most, so the story in a word problem doesn't crowd out the method. When a typed note and an OCR passage match about equally, the typed note comes first. The keyword ranking is built from the stored passages at question time, so it always matches the current index.
5. **Answer:** the passages go to a language model running in LM Studio, which writes an answer that cites them as [1], [2] and so on.
6. **Interface** (`study_chat.py`, `study_chat.html`, `study_app.pyw`): a small local web server and HTML interface, opened in its own desktop window with pywebview.

## Tech stack

Python 3.12 · ChromaDB · sentence-transformers (Nomic Embed v1.5) · PyTorch with CUDA · LM Studio local server · Tesseract OCR · pypdf · pywebview

**Tested on:** Windows 11 with a Radeon RX 7900 XT for the chat model and an RTX 3060, which runs the embedding model and also holds part of the chat model when LM Studio splits it across both GPUs.

## Benchmarks

User-measured on my PC (RTX 3060 + RX 7900 XT, LM Studio Vulkan), using a Solve-mode question asking for a 95% confidence interval for a proportion (n=120, x=54):

| Model | Answer time | Notes |
| --- | --- | --- |
| Qwen3.8 27B | ~35-41 s | Correct; cited the typed formula note; numbers from tools |
| GPT-OSS 20B | ~15-20 s | Correct interval; before the tool showed the standard error, it made one up |
| Flint | ~490 s | Correct, but about 7,000 words of reasoning |

With Qwen split across both GPUs, generation ran at about 42 tokens/s (about 35 s total answer time). On the 7900 XT alone, it ran at about 25 tokens/s (about 63 s). Answer time also depends on how long the model reasons.

## Setup

**You'll need:** Windows, Python 3.12, an NVIDIA GPU (recommended for speed), [LM Studio](https://lmstudio.ai/) with at least one chat model downloaded (the default is Qwen 27B, `qwen/qwen3.8-27b`), and optionally [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for reading text in PDF images.

1. Clone the repo and open a terminal in its folder:
   ```
   git clone https://github.com/JJSpectre/study-chat-local-rag.git
   cd study-chat-local-rag
   ```
2. Create a virtual environment and install the packages. Install PyTorch first, with CUDA support:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
3. Add your documents, one folder per course. To try it first, copy the `AUTO-101` folder from `sample-notes` into a new `notes` folder. It has two short sample notes on brakes and check engine codes, plus starter questions.
   ```
   notes\
       COURSE-1\
           chapter-1.pdf
           my-notes.txt
       COURSE-2\
           ...
   ```
4. In LM Studio, load a chat model and turn on the **Local Server** (port 1234).
5. Double-click **Setup Study Chat.cmd** once. It creates a **Study Chat** shortcut on your desktop.
6. Open Study Chat from the shortcut and click **Update index** to build the search database.

## Starter questions

Each course can have its own starter cards on the welcome screen. Put a file named `starters.toml` in the course folder, with one `[[card]]` section per card:

```toml
[[card]]
title = "Review a concept"
detail = "How disc brakes work"
question = "How does a disc brake slow a wheel?"
```

Reopen Study Chat or click **Update index** to see your changes. Courses without this file show no cards. Everything under `notes` stays out of Git, so your own cards stay private.

## Notes on accuracy

- OCR can misread math symbols. For formulas you rely on, keep a typed version in a `.txt` file in the course folder. Study Chat prefers typed text over OCR when both exist.
- Answers are only as good as the documents behind them. Use the source passages to check anything important against the original.

## How this was built

I designed Study Chat, tested it on my own hardware, and directed every change, working with AI coding assistants to write and review much of the code. I'm studying Computer Science (AI-Assisted Software Engineering) at UNC Charlotte, and this project is how I learned RAG, embeddings, vector databases and local LLM tooling hands-on.
