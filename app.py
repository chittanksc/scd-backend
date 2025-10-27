import os
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import psycopg2.extras
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from PIL import Image
import numpy as np

from config import Config
from model_loader import get_model, predict_image
from emailer import send_result_email

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS", "*")}}, supports_credentials=True)

connection_kwargs = {
    "host": app.config.get("DB_HOST", "127.0.0.1"),
    "user": app.config.get("DB_USER", "root"),
    "password": app.config.get("DB_PASSWORD", ""),
    "dbname": app.config.get("DB_NAME", "scd"),
}

os.makedirs(os.path.join(os.path.dirname(__file__), "uploads"), exist_ok=True)


def get_db_connection():
    return psycopg2.connect(**connection_kwargs)


def create_jwt(payload):
    token = jwt.encode(
        {**payload, "exp": datetime.utcnow() + timedelta(hours=12)},
        app.config["JWT_SECRET_KEY"],
        algorithm="HS256",
    )
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def decode_jwt(token):
    return jwt.decode(token, app.config["JWT_SECRET_KEY"], algorithms=["HS256"])


@app.post("/api/register")
def register():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not name or not email or not password:
        return jsonify({"error": "Missing fields"}), 400
    password_hash = generate_password_hash(password)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (name, email, password_hash, created_at) VALUES (%s, %s, %s, NOW())",
                    (name, email, password_hash),
                )
                conn.commit()
    except psycopg2.errors.UniqueViolation:
        return jsonify({"error": "Email already registered"}), 409
    return jsonify({"message": "Registered successfully"}), 201


@app.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not email or not password:
        return jsonify({"error": "Missing fields"}), 400
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, email, password_hash FROM users WHERE email=%s", (email,))
            user = cur.fetchone()
            if not user or not check_password_hash(user["password_hash"], password):
                return jsonify({"error": "Invalid credentials"}), 401
    token = create_jwt({"uid": user["id"], "email": user["email"], "name": user["name"]})
    return jsonify({"token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"]}})


def auth_required(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = decode_jwt(token)
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        request.user = payload
        return fn(*args, **kwargs)

    return wrapper


@app.get("/api/me")
@auth_required
def me():
    return jsonify({"user": request.user})


@app.post("/api/upload")
@auth_required
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    filepath = os.path.join(upload_dir, f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{file.filename}")
    file.save(filepath)
    try:
        img = Image.open(filepath).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400
    model = get_model()
    label, confidence = predict_image(model, img)
    try:
        send_result_email(
            to_email=request.user.get("email"),
            subject="Skin Cancer Detection Result",
            body=f"Prediction: {label}\nConfidence: {confidence:.2%}",
        )
    except Exception:
        pass
    return jsonify({"prediction": label, "confidence": confidence})


@app.get("/api/setup-db")
def setup_database():
    """Temporary endpoint to initialize database - REMOVE AFTER FIRST USE"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        email VARCHAR(191) NOT NULL UNIQUE,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP NOT NULL
                    )
                ''')
                conn.commit()
        return jsonify({"message": "Database initialized successfully!", "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
