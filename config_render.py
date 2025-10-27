import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-this-jwt-secret")
    
    # Database Configuration - Render uses PostgreSQL or MySQL
    # For MySQL (if you prefer)
    DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
    DB_NAME = os.environ.get("DB_NAME", "scd")
    
    # CORS Configuration
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
    
    # Model Path - Render uses /opt/render/project/src/
    MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "skin_cancer_cnn.h5"))
    
    # SMTP Email Configuration
    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USER = os.environ.get("SMTP_USER", "")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
    SMTP_FROM = os.environ.get("SMTP_FROM", os.environ.get("SMTP_USER", ""))
