import json
import os
import time
from typing import Any, Dict, List, Optional

from .config import settings

CONFIG_PATH = os.path.join(settings.DATA_DIR, "config.json")
WEBHOOK_LOG_PATH = os.path.join(settings.DATA_DIR, "webhook_log.json")

_default_config: Dict[str, Any] = {
    # Webhook de entrada (recebe imagens)
    "WEBHOOK_INFER_ENABLED": True,
    "WEBHOOK_INFER_TOKEN": None,
    "WEBHOOK_INFER_PATH": "infer",
    
    # Webhook de saída (envia para n8n)
    "N8N_WEBHOOK_URL": settings.N8N_WEBHOOK_URL,
    "N8N_WEBHOOK_ENABLED": True,
    "N8N_WEBHOOK_INCLUDE_IMAGE": False,
    
    # Parâmetros de inferência
    "CONF_DEFAULT": 0.25,
    "IOU_DEFAULT": 0.45,
    "IMGSZ_DEFAULT": 640,
    "DEVICE_DEFAULT": None,
}

_cached: Dict[str, Any] | None = None


def _ensure_file() -> None:
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(_default_config, f)


def load_config() -> Dict[str, Any]:
    global _cached
    _ensure_file()
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
    except Exception:
        data = {}
    for k, v in _default_config.items():
        data.setdefault(k, v)
    _cached = data
    return data


def save_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    data = load_config()
    data.update({k: v for k, v in updates.items() if v is not None})
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f)
    return data


def log_webhook(direction: str, endpoint: str, status: int, payload: Dict[str, Any]) -> None:
    """Registra uma entrada de log para webhooks (entrada ou saída)
    
    Args:
        direction: "in" para webhooks recebidos, "out" para webhooks enviados
        endpoint: caminho do endpoint ou URL de destino
        status: código de status HTTP
        payload: payload do webhook (será resumido)
    """
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    
    # Resumir o payload para não ocupar muito espaço
    payload_summary = {}
    if payload:
        if isinstance(payload, dict):
            # Copiar apenas chaves de primeiro nível e contar itens em listas
            for k, v in payload.items():
                if isinstance(v, list):
                    payload_summary[k] = f"[{len(v)} itens]"
                elif isinstance(v, dict):
                    payload_summary[k] = "{...}"
                else:
                    payload_summary[k] = str(v)[:50]
        else:
            payload_summary = {"data": str(payload)[:50]}
    
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "direction": direction,
        "endpoint": endpoint,
        "status": status,
        "payload_summary": payload_summary
    }
    
    # Carregar logs existentes
    entries = []
    if os.path.exists(WEBHOOK_LOG_PATH):
        try:
            with open(WEBHOOK_LOG_PATH) as f:
                entries = json.load(f)
        except Exception:
            entries = []
    
    # Adicionar nova entrada e limitar a 100 entradas
    entries = [entry] + entries
    if len(entries) > 100:
        entries = entries[:100]
    
    # Salvar logs
    with open(WEBHOOK_LOG_PATH, "w") as f:
        json.dump(entries, f)


def get_webhook_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retorna os logs de webhook mais recentes
    
    Args:
        limit: número máximo de logs a retornar
        
    Returns:
        Lista de logs de webhook, do mais recente para o mais antigo
    """
    if not os.path.exists(WEBHOOK_LOG_PATH):
        return []
    
    try:
        with open(WEBHOOK_LOG_PATH) as f:
            entries = json.load(f)
            return entries[:limit]
    except Exception:
        return []