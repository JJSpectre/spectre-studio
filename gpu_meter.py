"""Live GPU memory for every physical adapter, read from Windows itself.

Names and adapter IDs come from DXGI, usage from the Windows performance counters that
cover both AMD and NVIDIA, and installed sizes from the display drivers' registry keys.
Readings are adapter-wide: they can't tell which program uses the memory.
"""
import ctypes
import os
import re
import threading
import time
from ctypes import wintypes

RETRY_SECONDS = 30       # after a failure, wait this long before setting up again
PDH_FMT_LARGE = 0x400
PDH_MORE_DATA = 0x800007D2
SOFTWARE_ADAPTER = 2
DISPLAY_DRIVERS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
COUNTERS = {"dedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
            "shared": r"\GPU Adapter Memory(*)\Shared Usage"}
# One sample at a time. This lock is never held for anything else.
LOCK = threading.Lock()
LATEST = None            # (time.monotonic(), result) of the newest sample
QUERY = None             # (PDH query, {value name: counter})
ADAPTERS = {}            # LUID -> {"name", "total", "software"}
RETRY_AT = 0.0
LOGGED = set()


class LUID(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]


class AdapterDesc(ctypes.Structure):   # DXGI_ADAPTER_DESC1
    _fields_ = [("name", ctypes.c_wchar * 128), ("vendor", wintypes.UINT),
                ("device", wintypes.UINT), ("subsystem", wintypes.UINT),
                ("revision", wintypes.UINT), ("dedicated", ctypes.c_size_t),
                ("dedicated_system", ctypes.c_size_t), ("shared", ctypes.c_size_t),
                ("luid", LUID), ("flags", wintypes.UINT)]


class CounterValue(ctypes.Structure):  # PDH_FMT_COUNTERVALUE holding a LONGLONG
    _fields_ = [("status", wintypes.DWORD), ("value", ctypes.c_longlong)]


class CounterItem(ctypes.Structure):   # PDH_FMT_COUNTERVALUE_ITEM_W
    _fields_ = [("name", wintypes.LPWSTR), ("reading", CounterValue)]


def installed_size(desc):
    """The card's installed memory, as Task Manager shows it, or DXGI's usable size.

    The registry can list one card twice, so it never decides how many cards there are.
    """
    import winreg
    wanted = f"PCI\\VEN_{desc.vendor:04X}&DEV_{desc.device:04X}&SUBSYS_{desc.subsystem:08X}"
    sizes = set()
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, DISPLAY_DRIVERS) as drivers:
            for number in range(winreg.QueryInfoKey(drivers)[0]):
                try:
                    with winreg.OpenKey(drivers, winreg.EnumKey(drivers, number)) as driver:
                        device = winreg.QueryValueEx(driver, "MatchingDeviceId")[0].upper()
                        if device == wanted or device.startswith(wanted + "&"):
                            sizes.add(winreg.QueryValueEx(
                                driver, "HardwareInformation.qwMemorySize")[0])
                except (OSError, AttributeError):
                    continue   # Not a display driver, or not readable.
    except OSError:
        pass
    size = sizes.pop() if len(sizes) == 1 else None
    if isinstance(size, int) and 0 <= size - desc.dedicated <= 0.05 * size:
        return size
    return desc.dedicated


def list_adapters():
    """Every adapter by LUID, the ID that the usage counters also carry."""
    def call(pointer, slot, *args, types=(), result=ctypes.c_long):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *types)(table[slot])(pointer, *args)

    def release(pointer):
        call(pointer, 2, result=wintypes.ULONG)

    # IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
    interface = (ctypes.c_ubyte * 16).from_buffer_copy(
        bytes.fromhex("78ae0a776ff2ba4da829253c83d1b387"))
    factory = ctypes.c_void_p()
    if ctypes.WinDLL("dxgi").CreateDXGIFactory1(ctypes.byref(interface), ctypes.byref(factory)):
        raise OSError("Windows didn't list the graphics adapters.")
    found = {}
    try:
        for number in range(64):
            adapter = ctypes.c_void_p()
            if call(factory, 12, number, ctypes.byref(adapter),
                    types=(wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))):
                break   # No more adapters.
            try:
                desc = AdapterDesc()
                if call(adapter, 10, ctypes.byref(desc), types=(ctypes.POINTER(AdapterDesc),)):
                    continue
            finally:
                release(adapter)
            software = bool(desc.flags & SOFTWARE_ADAPTER) or not desc.dedicated
            found[desc.luid.high & 0xFFFFFFFF, desc.luid.low] = {
                "name": desc.name.strip(), "software": software,
                "total": 0 if software else installed_size(desc)}
    finally:
        release(factory)
    return found


def open_query():
    """Open the usage counters. A counter that isn't available is left out."""
    pdh = ctypes.WinDLL("pdh")
    query = ctypes.c_void_p()
    if pdh.PdhOpenQueryW(None, None, ctypes.byref(query)):
        raise OSError("Windows performance counters aren't available.")
    counters = {}
    for value, path in COUNTERS.items():
        counter = ctypes.c_void_p()
        # The English name works on every Windows language.
        if not pdh.PdhAddEnglishCounterW(query, path, None, ctypes.byref(counter)):
            counters[value] = counter
    if "dedicated" not in counters:
        pdh.PdhCloseQuery(query)
        raise OSError("Windows has no GPU memory counters.")
    return query, counters


def read_counter(counter):
    """One counter's value for each LUID. None marks a value that can't be trusted."""
    pdh = ctypes.WinDLL("pdh")
    size, count = wintypes.DWORD(0), wintypes.DWORD(0)
    arguments = (counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count))
    if pdh.PdhGetFormattedCounterArrayW(*arguments, None) & 0xFFFFFFFF != PDH_MORE_DATA:
        return {}
    buffer = ctypes.create_string_buffer(size.value)
    if pdh.PdhGetFormattedCounterArrayW(*arguments, buffer):
        return {}
    items = ctypes.cast(buffer, ctypes.POINTER(CounterItem))
    values = {}
    for number in range(count.value):
        name, reading = items[number].name, items[number].reading
        match = re.match(r"luid_0x([0-9a-f]+)_0x([0-9a-f]+)_phys_\d+", name, re.I)
        if match:
            luid = int(match[1], 16), int(match[2], 16)
            # Linked cards share one LUID. Don't guess which memory belongs to which.
            good = reading.status in (0, 1) and reading.value >= 0 and luid not in values
            values[luid] = reading.value if good else None
    return values


def sample():
    global QUERY, ADAPTERS
    if os.name != "nt":
        raise OSError("GPU memory is only read on Windows.")
    if QUERY is None:
        QUERY = open_query()
    query, counters = QUERY
    if ctypes.WinDLL("pdh").PdhCollectQueryData(query):
        raise OSError("Windows didn't return GPU memory readings.")
    usage = {value: read_counter(counter) for value, counter in counters.items()}
    if not usage["dedicated"]:
        raise OSError("Windows returned no GPU memory readings.")
    if set(usage["dedicated"]) != set(ADAPTERS):
        # First sample, or the adapters changed: a driver reset gives a card a new LUID.
        ADAPTERS = list_adapters()
    adapters = [{"name": adapter["name"], "total": adapter["total"] or None,
                 "dedicated": usage["dedicated"].get(luid),
                 "shared": usage.get("shared", {}).get(luid)}
                for luid, adapter in ADAPTERS.items() if not adapter["software"]]
    adapters.sort(key=lambda adapter: (-(adapter["total"] or 0), adapter["name"]))
    return {"adapters": adapters}


def close_query():
    global QUERY
    if QUERY is not None:
        ctypes.WinDLL("pdh").PdhCloseQuery(QUERY[0])
        QUERY = None


def snapshot():
    """Take a sample, or return the newest one if a sample is already running."""
    global LATEST, RETRY_AT
    if LOCK.acquire(blocking=False):
        try:
            if time.monotonic() >= RETRY_AT:
                try:
                    result = sample()
                except Exception as error:
                    close_query()
                    RETRY_AT = time.monotonic() + RETRY_SECONDS
                    result = {"error": "GPU readings unavailable"}
                    if str(error) not in LOGGED:
                        LOGGED.add(str(error))
                        print(f"Couldn't read GPU memory: {error}", flush=True)
                LATEST = time.monotonic(), result
        finally:
            LOCK.release()
    if LATEST is None:
        return {"waiting": True}
    taken, result = LATEST
    return {**result, "age": round(time.monotonic() - taken, 1)}
