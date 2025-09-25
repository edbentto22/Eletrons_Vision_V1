from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    label: Optional[str] = None

class ImageResult(BaseModel):
    image_id: str
    source: str
    source_url: Optional[str] = None  # URL absoluta para o arquivo de entrada servido pelo app
    width: int
    height: int
    detections: List[Box]
    annotated_url: Optional[str] = None  # será absoluto se PUBLIC_BASE_URL estiver definido

class InferResponse(BaseModel):
    count: int
    results: List[ImageResult]
    params: Dict[str, Any]

class TrainParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    epochs: int = 50
    imgsz: int = 640
    lr0: float = 0.01
    batch: int = 16
    device: Optional[str] = None
    pretrained: bool = True
    resume: bool = False
    model_variant: Optional[str] = None

class TrainStatus(BaseModel):
    job_id: str
    status: str
    best_pt: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

class ValidateRequest(BaseModel):
    data_yaml_path: Optional[str] = None
    device: Optional[str] = None

class ExportRequest(BaseModel):
    format: str = Field(pattern="^(onnx|trt)$")
    half: bool = False
    dynamic: bool = False
    device: Optional[str] = None

class PromoteRequest(BaseModel):
    job_id: Optional[str] = None
    best_pt_path: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    active_model: Optional[str]
    versions: Dict[str, str]

class MetricsResponse(BaseModel):
    inferences: int
    last_latency_ms: Optional[float]
    avg_latency_ms: Optional[float]
    active_jobs: int


class UIConfigRequest(BaseModel):
    # Webhook de entrada (recebe imagens)
    WEBHOOK_INFER_ENABLED: Optional[bool] = None
    WEBHOOK_INFER_TOKEN: Optional[str] = None
    WEBHOOK_INFER_PATH: Optional[str] = Field(default=None, pattern="^[a-zA-Z0-9_-]+$")
    
    # Webhook de saída (envia para n8n)
    N8N_WEBHOOK_URL: Optional[str] = None
    N8N_WEBHOOK_ENABLED: Optional[bool] = None
    N8N_WEBHOOK_INCLUDE_IMAGE: Optional[bool] = None
    
    # Parâmetros de inferência
    CONF_DEFAULT: Optional[float] = Field(default=None, ge=0.01, le=0.99)
    IOU_DEFAULT: Optional[float] = Field(default=None, ge=0.01, le=0.99)
    IMGSZ_DEFAULT: Optional[int] = Field(default=None, ge=64, le=1536)
    DEVICE_DEFAULT: Optional[str] = None

class UIConfigResponse(BaseModel):
    # Webhook de entrada (recebe imagens)
    WEBHOOK_INFER_ENABLED: bool
    WEBHOOK_INFER_TOKEN: Optional[str]
    WEBHOOK_INFER_PATH: str
    
    # Webhook de saída (envia para n8n)
    N8N_WEBHOOK_URL: Optional[str]
    N8N_WEBHOOK_ENABLED: bool
    N8N_WEBHOOK_INCLUDE_IMAGE: bool
    
    # Parâmetros de inferência
    CONF_DEFAULT: float
    IOU_DEFAULT: float
    IMGSZ_DEFAULT: int
    DEVICE_DEFAULT: Optional[str]

class WebhookLogEntry(BaseModel):
    timestamp: str
    direction: str  # "in" ou "out"
    endpoint: str
    status: int
    payload_summary: Dict[str, Any]
    
class WebhookLogResponse(BaseModel):
    entries: List[WebhookLogEntry]
    count: int