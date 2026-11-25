"""Open the local Spectre Studio backend in a standalone WebView2 window."""
import ctypes
import hashlib
import logging
import os
import sys
from pathlib import Path
from threading import Event, Thread

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "study-chat-desktop.log"
ICON = ROOT / "spectre-studio.ico"


def enable_right_click(window):
    """Turn on WebView2's right-click menu, which pywebview only enables in debug mode.

    It opens on the question box and on selected text. Elsewhere it would offer
    Refresh, which clears the chat.
    """
    view = window.native.webview

    def on_menu(_, args):
        try:
            target = args.ContextMenuTarget
            if not (target.IsEditable or target.HasSelection):
                args.Handled = True  # Handled without a choice: no menu opens.
                return
            for item in list(args.MenuItems):
                if item.Name == "reload":
                    args.MenuItems.Remove(item)
        except Exception:
            logging.exception("Could not prepare the right-click menu")
            args.Handled = True

    def turn_on(*_):
        try:
            if view.CoreWebView2 is not None:
                view.CoreWebView2.Settings.AreDefaultContextMenusEnabled = True
                view.CoreWebView2.ContextMenuRequested += on_menu
        except Exception:
            logging.exception("Could not enable the right-click menu")

    # WebView2 starts in the background. pywebview's own handler sets its defaults
    # first, then this one runs, before the page loads.
    if view.CoreWebView2 is None:
        view.CoreWebView2InitializationCompleted += turn_on
    else:
        turn_on()


def darken_title_bar(window):
    """Match the Windows title bar to the sidebar, whatever the Windows theme is."""
    try:
        set_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        handle = ctypes.c_void_p(window.native.Handle.ToInt64())
        # 20: dark window buttons. 35 and 36: title bar and title colors, written
        # as blue-green-red. Windows 10 ignores 35 and 36.
        for attribute, value in ((20, 1), (35, 0x1C150F), (36, 0xFBF2EA)):
            set_attribute(handle, attribute, ctypes.byref(ctypes.c_int(value)), 4)
    except Exception:
        logging.exception("Could not darken the title bar")


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
            "Spectre Studio", backend.ORIGIN, width=1360, height=860,
            min_size=(690, 510), background_color="#0b1118", text_select=True,
        )
        window.events.before_show += lambda: enable_right_click(window)
        window.events.before_show += lambda: darken_title_bar(window)
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
                        logging.exception("Could not close the Spectre Studio window")

        window.events.closed += mark_closed
        thread = Thread(target=serve, name="study-chat-server", daemon=True)
        thread.start()
        logging.info("Spectre Studio desktop starting on %s", backend.ORIGIN)
        try:
            webview.start(gui="edgechromium", private_mode=True, icon=str(ICON))
        finally:
            # The window's X stops this app's backend, not LM Studio or its models.
            closing.set()
            server.shutdown()
            thread.join(timeout=3)
            logging.info("Spectre Studio desktop stopped")


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
            notify("Spectre Studio is already open. Look for its window on the taskbar.", 0x40)
            return
        open_window()
    finally:
        kernel.CloseHandle(mutex)


def notify(text, flags=0x10):
    ctypes.windll.user32.MessageBoxW(None, text, "Spectre Studio", flags)


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
            notify(f"Spectre Studio couldn't start.\n\n{error}\n\n"
                   "Run Setup Spectre Studio again if pywebview is missing. "
                   f"For details, see {LOG.name} in rag-study.")


if __name__ == "__main__":
    main()
