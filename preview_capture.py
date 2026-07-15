#!/usr/bin/env python3
"""Live preview + capture button + instant OCR.
View at http://192.168.137.165:8000 - click CAPTURE to shoot and OCR."""
import io
from http import server
from threading import Condition
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from libcamera import controls
import pytesseract
from PIL import Image

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

last_ocr = "No capture yet"

class Handler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        global last_ocr
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = f"""<html><body style="margin:0;background:#111;color:#eee;font-family:sans-serif">
            <img src="/stream" style="width:100%"/>
            <div style="padding:12px">
            <a href="/capture" style="display:inline-block;padding:14px 28px;background:#c00;color:#fff;
               text-decoration:none;font-size:18px;border-radius:8px">CAPTURE + OCR</a>
            <h3>Last OCR result:</h3><pre style="white-space:pre-wrap">{last_ocr}</pre>
            </div></body></html>"""
            self.wfile.write(html.encode())
        elif self.path == '/capture':
            req = picam2.capture_request()
            req.save("main", "/home/raspberrypi/fyp/ocr_capture.jpg")
            req.release()
            img = Image.open("/home/raspberrypi/fyp/ocr_capture.jpg")
            text = pytesseract.image_to_string(img).strip()
            last_ocr = text if text else "(no text detected)"
            self.send_response(303)
            self.send_header('Location', '/')
            self.end_headers()
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)).encode())
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception:
                pass

    def log_message(self, *args):
        pass

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (2304, 1296)},
    lores={"size": (1152, 648)},
    encode="lores")
picam2.configure(config)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))
print("Open http://192.168.137.165:8000 - Ctrl+C to stop")
try:
    server.HTTPServer(('', 8000), Handler).serve_forever()
except KeyboardInterrupt:
    picam2.stop_recording()
