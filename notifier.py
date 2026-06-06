import os
import threading
import queue
import smtplib
from email.message import EmailMessage
import mimetypes
from dotenv import load_dotenv

# 1. Look for the hidden .env file in the directory and load it into environment memory
load_dotenv()

# 2. Extract configuration variables securely with safe fallback defaults
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.example.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "alerts@example.com")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "app-password")  
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "ops@example.com")


# Internal queue + worker for non-blocking sends
_send_queue: "queue.Queue[tuple[str, str | None]]" = queue.Queue()

def _send_email(timestamp_str: str, image_path: str | None) -> None:
    """
    Build and send an email. Attaches the image if image_path is provided.
    This runs in a separate worker thread to avoid blocking the main loop.
    """
    msg = EmailMessage()
    msg['Subject'] = f"GarudAI Intrusion Alert - {timestamp_str}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    # Simple HTML body
    html_body = f"""
    <html><body>
      <h2>Intrusion Detected</h2>
      <p>Timestamp: {timestamp_str}</p>
      <p>A detected human entered the restricted zone. See attached snapshot for reference.</p>
    </body></html>
    """
    msg.set_content("Intrusion detected. See HTML alternative for details.")
    msg.add_alternative(html_body, subtype="html")

    # Attach image if provided
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as img_f:
                # Guess MIME type; default to jpeg if unknown
                mime_type, _ = mimetypes.guess_type(image_path)
                if mime_type is None:
                    mime_type = "image/jpeg"
                maintype, subtype = mime_type.split('/', 1)
                if maintype == "image":
                    msg.add_attachment(img_f.read(),
                                       maintype=maintype,
                                       subtype=subtype,
                                       filename=os.path.basename(image_path))
        except Exception as e:
            print(f"[notifier] Failed attaching image: {e}")

    # Send via SMTP with TLS
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"[notifier] Alert email sent for {timestamp_str}")
    except Exception as e:
        print(f"[notifier] Email send failed: {e}")

def _worker():
    while True:
        item = _send_queue.get()
        if item is None:
            break
        try:
            timestamp_str, image_path = item
            _send_email(timestamp_str, image_path)
        finally:
            _send_queue.task_done()

# Start a daemon worker thread
_thread = threading.Thread(target=_worker, daemon=True)
_thread.start()

def send_intrusion_alert(timestamp_str: str, image_path: str | None = None) -> None:
    """
    Non-blocking wrapper to enqueue an intrusion alert email.
    - timestamp_str: human-readable timestamp (e.g., "2026-06-05 16:20:00")
    - image_path: optional path to breach snapshot to attach
    """
    _send_queue.put((timestamp_str, image_path))