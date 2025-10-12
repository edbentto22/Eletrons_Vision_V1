import os
from typing import List, Optional

class Settings:
    """Simple settings loader from environment variables for service configuration."""
    APP_NAME: str = os.getenv("APP_NAME", "Eletrons Vision Service")
    AUTH_TOKEN: Optional[str] = os.getenv("AUTH_TOKEN")
    N8N_WEBHOOK_URL: Optional[str] = os.getenv("N8N_WEBHOOK_URL")
    CORS_ORIGINS: List[str] = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")] if os.getenv("CORS_ORIGINS") else ["*"]
    IP_WHITELIST: List[str] = [ip.strip() for ip in os.getenv("IP_WHITELIST", "").split(",") if ip.strip()]

    # Base directories
    BASE_DIR: str = os.path.abspath(os.path.dirname(__file__))
    MODELS_DIR: str = os.getenv("MODELS_DIR", os.path.join(BASE_DIR, "models"))
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
    RUNS_DIR: str = os.getenv("RUNS_DIR", os.path.join(BASE_DIR, "runs"))
    WEB_DIR: str = os.path.join(BASE_DIR, "web")
    STATIC_DIR: str = os.path.join(WEB_DIR, "static")

    # Public base URL used to build absolute URLs in webhooks/responses (e.g., https://vision.example.com)
    PUBLIC_BASE_URL: Optional[str] = os.getenv("PUBLIC_BASE_URL")

    ACTIVE_MODEL_PATH: str = os.getenv("ACTIVE_MODEL", os.path.join(MODELS_DIR, "production.pt"))
    MODEL_VARIANT: str = os.getenv("MODEL_VARIANT", "yolov8n.pt")

    SAVE_ANNOTATIONS: bool = os.getenv("SAVE_ANNOTATIONS", "true").lower() == "true"
    MAX_WEBHOOK_PAYLOAD_MB: int = int(os.getenv("MAX_WEBHOOK_PAYLOAD_MB", "16"))

    # Authentication (UI login)
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@example.com")
    ADMIN_PASSWORD_HASH: Optional[str] = os.getenv("ADMIN_PASSWORD_HASH")
    ADMIN_PASSWORD: Optional[str] = os.getenv("ADMIN_PASSWORD")  # only for development; prefer ADMIN_PASSWORD_HASH in production
    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "change-me-please")

    # Environment
    APP_ENV: str = os.getenv("APP_ENV", "development")

    def ensure_dirs(self) -> None:
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.MODELS_DIR, "history"), exist_ok=True)
        os.makedirs(os.path.join(self.MODELS_DIR, "onnx"), exist_ok=True)
        os.makedirs(os.path.join(self.MODELS_DIR, "trt"), exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.RUNS_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "infer"), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "zips"), exist_ok=True)
        os.makedirs(os.path.join(self.DATA_DIR, "extracted"), exist_ok=True)
        os.makedirs(os.path.join(self.STATIC_DIR, "detections"), exist_ok=True)
        # Optional: a dedicated static folder for uploads mapping
        os.makedirs(os.path.join(self.STATIC_DIR, "uploads"), exist_ok=True)

settings = Settings()
settings.ensure_dirs()