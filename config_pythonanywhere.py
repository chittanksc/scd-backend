import os

class Config:
    # Database Configuration for PythonAnywhere
    DB_HOST = os.environ.get("DB_HOST", "chittanks.mysql.pythonanywhere-services.com")
    DB_USER = os.environ.get("DB_USER", "chittanks")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "your_mysql_password")
    DB_NAME = os.environ.get("DB_NAME", "chittanks$scd")
    
    # JWT Secret Key
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
    
    # SMTP Email Configuration (Optional)
    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USER = os.environ.get("SMTP_USER", "")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
    SMTP_FROM = os.environ.get("SMTP_FROM", "")
    
    # Model Path - PythonAnywhere uses absolute path
    MODEL_PATH = os.environ.get("MODEL_PATH", "/home/chittanks/scd_backend/skin_cancer_cnn.h5")
    
    # Upload folder
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/home/chittanks/scd_backend/uploads")
