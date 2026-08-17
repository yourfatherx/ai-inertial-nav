import http.server, base64

OUT = r"D:\AI-inertial-nav\results\demo_last_frame.png"

class H(http.server.BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n).decode("ascii")
        b64 = data.split(",", 1)[1] if data.startswith("data:") else data
        with open(OUT, "wb") as f:
            f.write(base64.b64decode(b64))
        self.send_response(200); self._cors(); self.end_headers()
        self.wfile.write(b"saved")
        print("SAVED", len(b64), flush=True)
    def log_message(self, *a): pass

http.server.HTTPServer(("127.0.0.1", 8766), H).serve_forever()
