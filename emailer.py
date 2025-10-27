import smtplib
from email.mime.text import MIMEText
from config import Config

def send_result_email(to_email: str, subject: str, body: str):
    if not to_email:
        return
    if not Config.SMTP_USER or not Config.SMTP_PASSWORD:
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = Config.SMTP_FROM or Config.SMTP_USER
    msg["To"] = to_email
    with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT) as server:
        server.starttls()
        server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
        server.send_message(msg)
