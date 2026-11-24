"""Read course notes, including OCR text from raster images embedded in PDFs."""

import os
import hashlib
import json
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, __version__ as PILLOW_VERSION
from pypdf import PdfReader, __version__ as PYPDF_VERSION

NOTES_DIR = Path(__file__).resolve().parent / "notes"
CHUNK_SIZE = 500
OVERLAP = 50
OCR_CACHE = NOTES_DIR.parent / ".ocr_cache"
# Change this when extract_image_text() changes, so old OCR results aren't reused.
OCR_SETTINGS = "ocr-v1-eng-psm3"
# Change this when parse_pdf() changes, so old saved PDFs aren't reused.
PDF_CACHE_VERSION = "pdf-v1"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class OCRError(RuntimeError):
    pass


def find_tesseract():
    executable = shutil.which("tesseract")
    if executable:
        return executable
    if os.name == "nt":
        for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
            folder = os.environ.get(variable)
            if folder:
                candidate = Path(folder) / "Tesseract-OCR" / "tesseract.exe"
                if candidate.is_file():
                    return str(candidate)
    raise SystemExit(
        "Tesseract OCR was not found. Install it in the default location "
        "or make tesseract available on PATH before indexing PDF images."
    )


def extract_image_text(image, tesseract):
    """Convert a PDF image to a clean PNG and OCR it locally."""
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)

    image = ImageOps.grayscale(image)
    scale = min(3, 1800 / image.width, 2600 / image.height)
    if scale != 1:
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)

    with BytesIO() as png:
        image.save(png, format="PNG")
        result = subprocess.run(
            [tesseract, "stdin", "stdout", "-l", "eng", "--psm", "3"],
            input=png.getvalue(),
            capture_output=True,
            creationflags=NO_WINDOW,
            timeout=90,
        )

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Tesseract exited with code {result.returncode}: {detail}")
    return result.stdout.decode("utf-8", errors="replace").strip()


def write_cache(target, text):
    """Save a cache file in one step. If it can't be saved, it is just redone next time."""
    tmp = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(text)
        tmp.replace(target)
    except OSError:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def cached_image_text(image_file, tesseract, cache):
    executable = Path(tesseract).stat()
    key = hashlib.sha256()
    key.update(f"{OCR_SETTINGS}:{executable.st_size}:{executable.st_mtime_ns}:".encode())
    key.update(image_file.data)
    target = cache / (key.hexdigest() + ".txt")
    try:
        return target.read_text(encoding="utf-8")
    except OSError:
        pass
    text = extract_image_text(image_file.image, tesseract)
    write_cache(target, text)
    return text


def pdf_cache_file(data, cache):
    """Name the cache file after the PDF's bytes and every tool that affects the result."""
    try:
        executable = Path(find_tesseract()).stat()
        tesseract = f"{executable.st_size}:{executable.st_mtime_ns}"
    except (SystemExit, OSError):
        # Text-only PDFs don't need Tesseract. parse_pdf() reports it if a PDF does.
        tesseract = "none"
    key = hashlib.sha256()
    key.update(f"{PDF_CACHE_VERSION}:{OCR_SETTINGS}:pypdf-{PYPDF_VERSION}:"
               f"pillow-{PILLOW_VERSION}:{tesseract}:".encode())
    key.update(data)
    return cache / f"pdf-{key.hexdigest()}.json"


def load_pdf_cache(target):
    """Return (pages, image_texts) saved earlier, or None if missing or damaged."""
    try:
        saved = json.loads(target.read_text(encoding="utf-8"))
        pages = [(page, text) for page, text in saved["pages"]]
        image_texts = [(page, image, text) for page, image, text in saved["image_texts"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    for *numbers, text in pages + image_texts:
        if not isinstance(text, str) or not all(type(number) is int for number in numbers):
            return None
    return pages, image_texts


def read_pdf(path, *, cache=OCR_CACHE, progress=None):
    """Read a PDF, or reuse the saved result if this exact file was read before."""
    data = Path(path).read_bytes()
    target = pdf_cache_file(data, cache)
    saved = load_pdf_cache(target)
    if saved is not None:
        return saved
    pages, image_texts = parse_pdf(path, data, cache=cache, progress=progress)
    write_cache(target, json.dumps({"pages": pages, "image_texts": image_texts}))
    return pages, image_texts


def parse_pdf(path, data, *, cache=OCR_CACHE, progress=None):
    """Keep original PDF text separate from OCR, with page numbers on OCR."""
    reader = PdfReader(BytesIO(data))
    pages = []
    image_texts = []
    tesseract = None

    for page_number, page in enumerate(reader.pages, 1):
        pages.append((page_number, page.extract_text() or ""))
        images = page.images
        for image_number, key in enumerate(images.keys(), 1):
            try:
                image_file = images[key]
                image = image_file.image
                if image is None or min(image.size) < 24:
                    continue
                if tesseract is None:
                    tesseract = find_tesseract()
                if progress is not None:
                    progress(f"Reading {path.name} · page {page_number} of {len(reader.pages)}")
                text = cached_image_text(image_file, tesseract, cache)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                raise OCRError(
                    f"OCR failed in {path.name}, PDF page {page_number}, "
                    f"image {image_number}: {exc}"
                ) from exc
            if text:
                image_texts.append((page_number, image_number, text))

    return pages, image_texts


def word_spans(count, size=CHUNK_SIZE, overlap=OVERLAP):
    """Yield the start and exclusive end of each overlapping chunk."""
    step = size - overlap
    for start in range(0, count, step):
        end = min(start + size, count)
        yield start, end
        if end == count:
            break


def chunk_words(text, size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    return [" ".join(words[start:end])
            for start, end in word_spans(len(words), size, overlap)]


def chunk_pages(pages, size=CHUNK_SIZE, overlap=OVERLAP):
    """Return text chunks with the PDF pages of their first and last words."""
    words, word_pages = [], []
    for page_number, text in pages:
        page_words = text.split()
        words.extend(page_words)
        word_pages.extend([page_number] * len(page_words))
    return [(" ".join(words[start:end]), word_pages[start], word_pages[end - 1])
            for start, end in word_spans(len(words), size, overlap)]


def load_chunks(notes_dir=NOTES_DIR, *, progress=None):
    all_chunks = []
    for course_dir in sorted(notes_dir.iterdir()):
        if not course_dir.is_dir():
            continue
        course = course_dir.name

        for path in sorted(course_dir.rglob("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext == ".pdf":
                pages, image_texts = read_pdf(path, cache=notes_dir.parent / ".ocr_cache", progress=progress)
                text_chunks = chunk_pages(pages)
            elif ext in (".txt", ".md"):
                text = path.read_text(encoding="utf-8", errors="ignore")
                text_chunks = [(chunk, None, None) for chunk in chunk_words(text)]
                image_texts = []
            else:
                continue

            source = path.relative_to(notes_dir).as_posix()
            for i, (chunk, first, last) in enumerate(text_chunks):
                entry = {
                    "id": f"{source}#{i}",
                    "text": chunk,
                    "course": course,
                    "source": source,
                    "chunk_index": i,
                }
                if first is not None:
                    entry["page_start"], entry["page_end"] = first, last
                    if first == last:
                        entry["source"] = f"{source} (PDF page {first})"
                    else:
                        entry["source"] = f"{source} (PDF pages {first}–{last})"
                all_chunks.append(entry)

            for page_number, image_number, text in image_texts:
                for i, chunk in enumerate(chunk_words(text)):
                    all_chunks.append({
                        "id": f"{source}#ocr-p{page_number}-i{image_number}-{i}",
                        "text": (
                            "[Unverified OCR from a PDF image. Check the original "
                            "page for exact formulas and symbols.] " + chunk
                        ),
                        "course": course,
                        "source": f"{source} (PDF page {page_number}, OCR)",
                        "chunk_index": i,
                        "pdf_page": page_number,
                        "page_start": page_number,
                        "page_end": page_number,
                    })

    return all_chunks


if __name__ == "__main__":
    if not NOTES_DIR.is_dir():
        raise SystemExit(f"Create a notes folder next to ingest.py: {NOTES_DIR}")

    chunks = load_chunks()
    ocr_chunks = [chunk for chunk in chunks if "pdf_page" in chunk]
    print(f"Total chunks: {len(chunks)}")
    print(f"OCR chunks: {len(ocr_chunks)}")
    courses = {}
    for chunk in chunks:
        course = chunk["course"]
        courses[course] = courses.get(course, 0) + 1
    for course, count in courses.items():
        print(f"  {course}: {count} chunks")
    if ocr_chunks or chunks:
        first = (ocr_chunks or chunks)[0]
        print("\nSample source:", first["source"])
        print(first["text"][:300], "...")
