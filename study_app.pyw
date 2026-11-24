"""Open the existing Study Chat backend in a standalone WebView2 window."""
import ctypes
import hashlib
import logging
import os
import sys
from pathlib import Path
from threading import Event, Thread

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "study-chat-desktop.log"
ICON = ROOT / "study-chat.ico"


def open_window():
    import study_chat as backend
    import webview

    if not (ROOT / "study_chat.html").is_file():
        raise RuntimeError("Copy the desktop files into rag-study next to study_chat.py and study_chat.html.")

    # Bind before opening the window so there is no fixed timer or startup race.
    # Windows assigns a free loopback port, without affecting LM Studio's port.
    backend.LOG = LOG
    with backend.ThreadingHTTPServer((backend.HOST, 0), backend.Handler) as server:
        backend.PORT = server.server_port
        backend.ORIGIN = f"http://{backend.HOST}:{backend.PORT}"

        window = webview.create_window(
            "Study Chat", backend.ORIGIN, width=1140, height=810,
            min_size=(690, 510), background_color="#f8f8f2",
        )
        closing = Event()

        def mark_closed():
            closing.set()

        def serve():
            try:
                server.serve_forever(poll_interval=0.15)
            finally:
                # The in-app Quit button stops the server, then closes the window.
                if not closing.is_set():
                    try:
                        window.destroy()
                    except Exception:
                        logging.exception("Could not close the Study Chat window")

        window.events.closed += mark_closed
        thread = Thread(target=serve, name="study-chat-server", daemon=True)
        thread.start()
        logging.info("Study Chat desktop starting on %s", backend.ORIGIN)
        try:
            webview.start(gui="edgechromium", private_mode=True)
        finally:
            # The window's X stops this app's backend, not LM Studio or its models.
            closing.set()
            server.shutdown()
            thread.join(timeout=3)
            logging.info("Study Chat desktop stopped")


def launch_windows():
    import ctypes.wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.wintypes.BOOL, ctypes.wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = ctypes.wintypes.HANDLE
    kernel.CloseHandle.argtypes = (ctypes.wintypes.HANDLE,)
    kernel.CloseHandle.restype = ctypes.wintypes.BOOL

    root_id = hashlib.sha256(str(ROOT).casefold().encode("utf-8")).hexdigest()[:16]
    mutex = kernel.CreateMutexW(None, False, "Local\\StudyChatDesktop-" + root_id)
    error = ctypes.get_last_error()
    if not mutex:
        raise ctypes.WinError(error)

    try:
        if error == 183:
            notify("Study Chat is already open. Look for its window on the taskbar.", 0x40)
            return
        open_window()
    finally:
        kernel.CloseHandle(mutex)


def notify(text, flags=0x10):
    ctypes.windll.user32.MessageBoxW(None, text, "Study Chat", flags)


def main():
    if os.name != "nt":
        raise SystemExit("This desktop launcher is for Windows.")
    with LOG.open("a", encoding="utf-8", errors="replace", buffering=1) as stream:
        sys.stdout = stream
        sys.stderr = stream
        logging.basicConfig(
            stream=stream, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
        )
        try:
            launch_windows()
        except Exception as error:
            logging.exception("Desktop startup failed")
            notify(f"Study Chat couldn't start.\n\n{error}\n\n"
                   "Run Setup Study Chat again if pywebview is missing. "
                   f"For details, see {LOG.name} in rag-study.")


if __name__ == "__main__":
    main()
