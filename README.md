# Study Chat

**A local document search and Q&A app.** Drop your notes and PDFs into a folder, ask a question in plain English, and Study Chat finds the passages that answer it and shows you exactly where they came from: the file name and, for PDFs, the page number. Everything runs on your own computer. Your documents never leave it.

I built it because hunting through a semester's worth of PDFs for one formula or definition was slow. Study Chat turns that into a few seconds.

<!-- Screenshot: add one at docs/screenshot.png and remove the comment markers below -->
<!-- ![Study Chat answering a question with cited sources](docs/screenshot.png) -->

## What it does

- **Searches your own documents.** Reads PDF, TXT and Markdown files, splits them into passages, and finds the ones that match your question by meaning, not only by keywords.
- **Cites every source.** Each answer lists the passages it used, and you can expand them to read the original text. If your notes don't cover the question, it says so instead of guessing.
- **Reads text inside images.** Many lecture PDFs have formulas embedded as pictures. Study Chat runs OCR on those images and tags the results with their PDF page, so they're searchable too.
- **Organizes by course.** One folder per class. Search a single course or switch between them from a dropdown.
- **Two modes:**
  - **Explain from my notes** (default): answers questions using only what's in your documents.
  - **Solve a problem**: walks through a worked solution step by step, using the methods and formulas from your notes, then shows how to check the answer on a TI-84 Plus CE. The math comes from built-in calculation tools rather than the language model, so the numbers are reliable.
- **Model picker and comparison.** Choose which local model writes the answers, or run the same question through two models side by side to compare them.
- **100% offline.** No API keys, no accounts, no cloud services.

## How it works

Study Chat is a **retrieval-augmented generation (RAG)** app:

1. **Ingest** (`ingest.py`): reads every file in `notes/<course>/`, extracts the text (plus OCR text from PDF images via Tesseract), and splits it into passages tagged with course, file and page.
2. **Embed and store:** each passage is turned into a vector with the Nomic Embed v1.5 model, running on an NVIDIA GPU through PyTorch CUDA, and saved in a local ChromaDB database.
3. **Retrieve:** your question is embedded the same way, and ChromaDB returns the closest passages from the selected course.
4. **Answer:** the passages go to a language model running in LM Studio, which writes an answer that cites them as [1], [2] and so on.
5. **Interface** (`study_chat.py`, `study_chat.html`, `study_app.pyw`): a small local web server and HTML interface, opened in its own desktop window with pywebview.

## Tech stack

Python 3.12 · ChromaDB · sentence-transformers (Nomic Embed v1.5) · PyTorch with CUDA · LM Studio local server · Tesseract OCR · pypdf · pywebview

**Tested on:** Windows 11, RTX 3060 for embeddings, Radeon RX 7900 XT for the language model.

## Setup

**You'll need:** Windows, Python 3.12, an NVIDIA GPU (recommended for speed), [LM Studio](https://lmstudio.ai/) with at least one chat model downloaded, and optionally [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for reading text in PDF images.

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
3. Add your documents, one folder per course. To try it first, copy the `AUTO-101` folder from `sample-notes` into a new `notes` folder. It has two short sample notes on brakes and check engine codes.
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

## Notes on accuracy

- OCR can misread math symbols. For formulas you rely on, keep a typed version in a `.txt` file in the course folder. Study Chat prefers typed text over OCR when both exist.
- Answers are only as good as the documents behind them. Use the source passages to check anything important against the original.

## How this was built

I designed Study Chat, tested it on my own hardware, and directed every change, working with AI coding assistants to write and review much of the code. I'm studying Computer Science (AI-Assisted Software Engineering) at UNC Charlotte, and this project is how I learned RAG, embeddings, vector databases and local LLM tooling hands-on.

## Roadmap

- Make the default course and starter questions configurable instead of hard-coded
- Render equations instead of showing raw LaTeX
