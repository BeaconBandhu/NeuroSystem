"""
WebSocket reader — connects to ESP32 at ws://192.168.1.33:81
Runs in a daemon thread; pushes parsed JSON dicts into a queue.
"""

import json
import queue
import threading
import time

try:
    import websocket
    WS_OK = True
except ImportError:
    WS_OK = False

from eeg_decision.config import WS_URL


class WSReader:
    def __init__(self, url: str = WS_URL, maxsize: int = 64):
        self.url      = url
        self.q        = queue.Queue(maxsize=maxsize)
        self._ws      = None
        self._thread  = None
        self._running = False
        self.connected    = False
        self.epoch_count  = 0
        self.last_quality = "disconnected"
        self.on_connect_cb    = None
        self.on_disconnect_cb = None
        self.last_error       = ""

    def start(self):
        if not WS_OK:
            raise RuntimeError("websocket-client not installed. Run: pip install websocket-client")
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="WSReader")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def get(self, timeout: float = 0.05):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self):
        """Return all available items without blocking."""
        items = []
        while True:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                break
        return items

    # ── private ───────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.url,
                    on_open    = self._on_open,
                    on_message = self._on_msg,
                    on_error   = self._on_err,
                    on_close   = self._on_close,
                )
                # ping_interval=0 disables client-side pings —
                # ESP32 ArduinoWebSockets doesn't respond to pings from clients,
                # causing websocket-client to drop the connection after ping_timeout.
                self._ws.run_forever(ping_interval=0)
            except Exception:
                pass
            if self._running:
                self.connected = False
                if self.on_disconnect_cb:
                    self.on_disconnect_cb()
                time.sleep(3)

    def _on_open(self, ws):
        self.connected = True
        if self.on_connect_cb:
            self.on_connect_cb()

    def _on_close(self, ws, *args):
        self.connected = False
        if self.on_disconnect_cb:
            self.on_disconnect_cb()

    def _on_err(self, ws, err):
        self.last_error = str(err)

    def _on_msg(self, ws, raw):
        try:
            data = json.loads(raw)
            self.epoch_count  += 1
            self.last_quality  = data.get("quality", "unknown")
            try:
                self.q.put_nowait(data)
            except queue.Full:
                self.q.get_nowait()   # drop oldest, keep latest
                self.q.put_nowait(data)
        except Exception:
            pass
