import os
import uuid
import asyncio
import time
import base64
from typing import List, Optional, Union

from fastapi import FastAPI, UploadFile, File, Body, Query, Request, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import httpx

from .config import settings
from .schemas import InferResponse, TrainParams, TrainStatus, ValidateRequest, ExportRequest, PromoteRequest, HealthResponse, MetricsResponse, UIConfigRequest, UIConfigResponse, WebhookLogResponse
from .yolo_service import infer as yolo_infer, start_training, validate_model, promote_model, export_model, _jobs, register_job_task, cancel_job_task, clear_jobs
from .yolo_service import send_n8n_webhook
from .state import load_config, save_config

app = FastAPI(title=settings.APP_NAME)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.CORS_ORIGINS != ['*'] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and templates
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=os.path.join(settings.WEB_DIR, "templates"))

# Security dependencies
async def require_auth(request: Request):
    if not settings.AUTH_TOKEN:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] != settings.AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # IP whitelist if configured
    client_ip = request.client.host if request.client else None
    if settings.IP_WHITELIST and client_ip not in settings.IP_WHITELIST:
        raise HTTPException(status_code=403, detail="Forbidden IP")

# Metrics
_metrics = {
    'inferences': 0,
    'lat_total_ms': 0.0,
    'last_latency_ms': None,
}

@app.get("/health", response_model=HealthResponse)
async def health():
    cuda = False
    try:
        import torch  # type: ignore
        cuda = bool(getattr(torch, 'cuda', None) and torch.cuda.is_available())
    except Exception:
        cuda = False
    versions = {}
    try:
        import ultralytics  # type: ignore
        versions['ultralytics'] = getattr(ultralytics, '__version__', 'unknown')
    except Exception:
        versions['ultralytics'] = 'not-installed'
    try:
        import torch  # type: ignore
        versions['torch'] = getattr(torch, '__version__', 'unknown')
    except Exception:
        versions['torch'] = 'not-installed'
    return HealthResponse(status="ok", cuda_available=cuda, active_model=settings.ACTIVE_MODEL_PATH if os.path.exists(settings.ACTIVE_MODEL_PATH) else None, versions=versions)

@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    avg = (_metrics['lat_total_ms'] / _metrics['inferences']) if _metrics['inferences'] else None
    return MetricsResponse(
        inferences=_metrics['inferences'],
        last_latency_ms=_metrics['last_latency_ms'],
        avg_latency_ms=avg,
        active_jobs=sum(1 for j in _jobs.values() if j.get('status') == 'running')
    )

@app.post("/infer", response_model=InferResponse)
async def infer_endpoint(
    request: Request,
    files: List[UploadFile] = File(...),
    urls: Optional[List[str]] = Form(default=None),
    conf: float = Query(default=load_config().get("CONF_DEFAULT", 0.25), ge=0.01, le=0.99),
    iou: float = Query(default=load_config().get("IOU_DEFAULT", 0.45), ge=0.01, le=0.99),
    imgsz: int = Query(default=load_config().get("IMGSZ_DEFAULT", 640), ge=64, le=1536),
    device: Optional[str] = Query(default=None),
    registro: Optional[int] = Form(default=None),
    ponto: Optional[int] = Form(default=None),
    sheet_id: Optional[str] = Form(default=None),
    send_webhook: bool = Form(default=False)
):
    # Gather local paths
    paths: List[str] = []
    infer_dir = os.path.join(settings.DATA_DIR, 'infer')
    os.makedirs(infer_dir, exist_ok=True)
    # Uploads
    for f in files:
        fid = uuid.uuid4().hex
        out_path = os.path.join(infer_dir, f"{fid}-{f.filename}")
        with open(out_path, 'wb') as w:
            w.write(await f.read())
        paths.append(out_path)
    # URLs
    if urls:
        async with httpx.AsyncClient(timeout=20) as client:
            for u in urls:
                fid = uuid.uuid4().hex
                out_path = os.path.join(infer_dir, f"{fid}.jpg")
                try:
                    resp = await client.get(u)
                    resp.raise_for_status()
                    with open(out_path, 'wb') as w:
                        w.write(resp.content)
                    paths.append(out_path)
                except Exception:
                    continue
    if not paths:
        raise HTTPException(status_code=400, detail="Nenhuma imagem enviada ou URL válida")

    result = await yolo_infer(paths, conf=conf, iou=iou, imgsz=imgsz, device=device, send_webhook=send_webhook, extra_meta={"registro": registro, "ponto": ponto, "sheet_id": sheet_id})
    _metrics['inferences'] += 1
    _metrics['lat_total_ms'] += result.get('latency_ms', 0.0) or 0.0
    _metrics['last_latency_ms'] = result.get('latency_ms')
    _metrics['last_latency_ms'] = result.get('latency_ms')
    return InferResponse(count=result['count'], results=result['results'], params=result['params'])

@app.get("/config", response_model=UIConfigResponse)
async def get_ui_config():
    cfg = load_config()
    return UIConfigResponse(**cfg)

@app.post("/config", response_model=UIConfigResponse, dependencies=[Depends(require_auth)])
async def set_ui_config(req: UIConfigRequest):
    cfg = save_config(req.model_dump(exclude_none=True))
    return UIConfigResponse(**cfg)

@app.get("/panel", include_in_schema=False)
async def panel(request: Request):
    cfg = load_config()
    # Load recent detections
    log_path = os.path.join(settings.DATA_DIR, 'infer', 'log.json')
    recent = []
    if os.path.exists(log_path):
        import json
        try:
            recent = json.load(open(log_path))[-20:]
        except Exception:
            recent = []
    return templates.TemplateResponse("panel.html", {
        "request": request,
        "app_name": settings.APP_NAME,
        "active_model": settings.ACTIVE_MODEL_PATH if os.path.exists(settings.ACTIVE_MODEL_PATH) else None,
        "detections": recent,
        "jobs": list(_jobs.values())[::-1],
        "cfg": cfg,
        "n8n_webhook": cfg.get("N8N_WEBHOOK_URL"),
        "base_url": str(request.base_url)
    })

@app.post("/panel/upload", include_in_schema=False, dependencies=[Depends(require_auth)])
async def panel_upload_dataset(dataset: UploadFile = File(...), epochs: int = Form(50), imgsz: int = Form(640), batch: int = Form(16), lr0: float = Form(0.01), resume: bool = Form(False), device: Optional[str] = Form(None), pretrained: bool = Form(True)):
    params = TrainParams(epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0, resume=resume, device=device, pretrained=pretrained)
    import uuid as _uuid
    job_id = _uuid.uuid4().hex
    up_dir = os.path.join(settings.DATA_DIR, 'zips')
    os.makedirs(up_dir, exist_ok=True)
    up_path = os.path.join(up_dir, f"{job_id}-{dataset.filename}")
    with open(up_path, 'wb') as w:
        w.write(await dataset.read())
    if dataset.filename.lower().endswith('.zip'):
        from .yolo_service import _extract_zip
        data_yaml_path = os.path.join(settings.DATA_DIR, 'extracted', job_id)
        data_yaml_path = _extract_zip(up_path, data_yaml_path)
    elif dataset.filename.lower().endswith('.yaml'):
        data_yaml_path = os.path.join(settings.DATA_DIR, 'data.yaml')
        with open(data_yaml_path, 'wb') as w:
            w.write(open(up_path, 'rb').read())
    else:
        raise HTTPException(status_code=400, detail="Arquivo deve ser .zip ou data.yaml")
    asyncio.create_task(start_training(job_id, data_yaml_path, params.model_dump()))
    return RedirectResponse(url="/panel", status_code=303)

@app.post("/train", dependencies=[Depends(require_auth)], response_model=TrainStatus)
async def train_endpoint(
    dataset: UploadFile = File(...),
    params: TrainParams = Depends(),
):
    # Prepare data.yaml path
    job_id = uuid.uuid4().hex
    data_yaml_path = None
    # Save upload
    up_dir = os.path.join(settings.DATA_DIR, 'zips')
    os.makedirs(up_dir, exist_ok=True)
    up_path = os.path.join(up_dir, f"{job_id}-{dataset.filename}")
    with open(up_path, 'wb') as w:
        w.write(await dataset.read())

    if dataset.filename.lower().endswith('.zip'):
        data_yaml_path = os.path.join(settings.DATA_DIR, 'extracted', job_id)
        from .yolo_service import _extract_zip
        data_yaml_path = _extract_zip(up_path, data_yaml_path)
    elif dataset.filename.lower().endswith('.yaml'):
        data_yaml_path = os.path.join(settings.DATA_DIR, 'data.yaml')
        with open(data_yaml_path, 'wb') as w:
            w.write(open(up_path, 'rb').read())
    else:
        raise HTTPException(status_code=400, detail="Arquivo deve ser .zip ou data.yaml")

    asyncio.create_task(start_training(job_id, data_yaml_path, params.model_dump()))
    return TrainStatus(job_id=job_id, status="running")

@app.get("/train/{job_id}", response_model=TrainStatus)
async def train_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return TrainStatus(**job)

@app.post("/validate", dependencies=[Depends(require_auth)])
async def validate_endpoint(req: ValidateRequest):
    data_yaml = req.data_yaml_path or os.path.join(settings.DATA_DIR, 'data.yaml')
    metrics = await validate_model(data_yaml, device=req.device)
    return JSONResponse(metrics)

@app.post("/models/promote", dependencies=[Depends(require_auth)])
async def promote_endpoint(req: PromoteRequest):
    active = await promote_model(req.job_id, req.best_pt_path)
    return {"active_model": active}

@app.post("/export", dependencies=[Depends(require_auth)])
async def export_endpoint(req: ExportRequest):
    out_path = await export_model(req.format, req.half, req.dynamic, req.device)
    return {"export_path": out_path}

@app.get("/", include_in_schema=False)
async def ui_index(request: Request):
    # Load recent detections
    log_path = os.path.join(settings.DATA_DIR, 'infer', 'log.json')
    recent = []
    if os.path.exists(log_path):
        import json
        try:
            recent = json.load(open(log_path))[-20:]
        except Exception:
            recent = []
    return templates.TemplateResponse("index.html", {
        "request": request,
        "app_name": settings.APP_NAME,
        "active_model": settings.ACTIVE_MODEL_PATH if os.path.exists(settings.ACTIVE_MODEL_PATH) else None,
        "detections": recent,
        "jobs": list(_jobs.values())[-20:]
    })

@app.post("/webhook/infer", response_model=InferResponse)
async def webhook_infer_endpoint(
    request: Request,
    payload: dict = Body(..., description="JSON com images_base64 (lista) ou urls (lista)"),
    conf: float = Query(default=None, ge=0.01, le=0.99),
    iou: float = Query(default=None, ge=0.01, le=0.99),
    imgsz: int = Query(default=None, ge=64, le=1536),
    device: Optional[str] = Query(default=None)
):
    # Debug
    print(f"Payload recebido: {payload}")
    
    # Verificar se webhook está habilitado
    cfg = load_config()
    if not cfg.get("WEBHOOK_INFER_ENABLED", True):
        raise HTTPException(status_code=403, detail="Webhook de inferência desabilitado")
    
    # Verificar token se configurado
    if cfg.get("WEBHOOK_INFER_TOKEN"):
        auth = request.headers.get("Authorization", "")
        token = None
        if auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1]
        if not token or token != cfg.get("WEBHOOK_INFER_TOKEN"):
            raise HTTPException(status_code=401, detail="Token inválido")
    
    # Usar valores padrão da configuração se não fornecidos
    conf = conf or cfg.get("CONF_DEFAULT", 0.25)
    iou = iou or cfg.get("IOU_DEFAULT", 0.45)
    imgsz = imgsz or cfg.get("IMGSZ_DEFAULT", 640)
    device = device or cfg.get("DEVICE_DEFAULT")
    
    paths: List[str] = []
    infer_dir = os.path.join(settings.DATA_DIR, 'infer')
    os.makedirs(infer_dir, exist_ok=True)

    # Decode base64 images
    b64_list = payload.get('images_base64')
    print(f"Base64 list: {b64_list}")
    if isinstance(b64_list, str):
        b64_list = [b64_list]
    if not b64_list and payload.get('image_base64'):
        b64_list = [payload.get('image_base64')]
    if b64_list:
        for b64img in b64_list:
            try:
                if isinstance(b64img, str) and b64img.strip().startswith('data:') and ',' in b64img:
                    b64img = b64img.split(',', 1)[1]
                raw = base64.b64decode(b64img, validate=False)
                fid = uuid.uuid4().hex
                out_path = os.path.join(infer_dir, f"{fid}.jpg")
                with open(out_path, 'wb') as w:
                    w.write(raw)
                paths.append(out_path)
            except Exception as e:
                print(f"Erro ao processar base64: {e}")
                continue

    # Download URLs
    urls = payload.get('urls')
    print(f"URLs: {urls}")
    if isinstance(urls, str):
        urls = [urls]
    if not urls and payload.get('url'):
        urls = [payload.get('url')]
    if urls:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            for u in urls:
                try:
                    fid = uuid.uuid4().hex
                    out_path = os.path.join(infer_dir, f"{fid}.jpg")
                    print(f"Baixando URL: {u}")
                    resp = await client.get(u)
                    resp.raise_for_status()
                    with open(out_path, 'wb') as w:
                        w.write(resp.content)
                    paths.append(out_path)
                    print(f"Imagem salva em: {out_path}")
                except Exception as e:
                    print(f"Erro ao baixar URL {u}: {e}")
                    continue

    print(f"Paths para inferência: {paths}")
    if not paths:
        from .state import log_webhook
        log_webhook("in", "/webhook/infer", 400, payload)
        raise HTTPException(status_code=400, detail="Nenhuma imagem recebida (images_base64/url/urls)")

    result = await yolo_infer(paths, conf=conf, iou=iou, imgsz=imgsz, device=device)
    _metrics['inferences'] += 1
    _metrics['lat_total_ms'] += result.get('latency_ms', 0.0) or 0.0
    _metrics['last_latency_ms'] = result.get('latency_ms')
    
    # Registrar log do webhook
    from .state import log_webhook
    log_webhook("in", "/webhook/infer", 200, {
        "images": len(paths),
        "detections": result['count'],
        "params": {"conf": conf, "iou": iou, "imgsz": imgsz}
    })
    
    return InferResponse(count=result['count'], results=result['results'], params=result['params'])

@app.get("/webhooks/logs", response_model=WebhookLogResponse, dependencies=[Depends(require_auth)])
async def webhook_logs(limit: int = Query(default=50, ge=1, le=100)):
    from .state import get_webhook_logs
    entries = get_webhook_logs(limit)
    return WebhookLogResponse(entries=entries, count=len(entries))


@app.get("/settings", include_in_schema=False)
async def settings_page(request: Request):
    cfg = load_config()
    # Carregar logs de webhook
    from .state import get_webhook_logs
    webhook_logs = get_webhook_logs(50)
    
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "app_name": settings.APP_NAME,
        "active_model": settings.ACTIVE_MODEL_PATH if os.path.exists(settings.ACTIVE_MODEL_PATH) else None,
        "cfg": cfg,
        "webhook_logs": webhook_logs,
        "base_url": str(request.base_url)
    })

@app.post("/webhook/send")
async def webhook_send_endpoint(payload: dict = Body(...)):
    """Endpoint para enviar manualmente os resultados ao webhook do n8n.
    Espera um payload contendo 'results' (lista) e opcionalmente 'summary' e 'meta'."""
    try:
        await send_n8n_webhook(payload)
        return {"status": "sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao enviar webhook: {str(e)}")

@app.get("/system/stats")
async def get_system_stats():
    import psutil
    import platform
    import socket
    from datetime import datetime, timedelta

    def get_size(bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024:
                return f"{bytes:.2f}{unit}"
            bytes /= 1024

    # CPU
    try:
        cpu_freq = psutil.cpu_freq()
        cpu_speed = f"{cpu_freq.current:.2f}GHz" if cpu_freq else '2.30GHz'  # Valor padrão se não disponível
    except:
        cpu_speed = '2.30GHz'  # Valor padrão em caso de erro
        
    cpu_info = {
        'usage': psutil.cpu_percent(),
        'model': platform.processor() or 'Intel Xeon',  # Valor padrão se não disponível
        'speed': cpu_speed,
        'cores': psutil.cpu_count()
    }

    # RAM
    ram = psutil.virtual_memory()
    ram_info = {
        'usage': ram.percent,
        'total': get_size(ram.total),
        'speed': '3200MHz',  # Valor fixo para exemplo
        'type': 'DDR4'  # Valor fixo para exemplo
    }

    # HDD
    disk = psutil.disk_usage('/')
    disk_info = {
        'usage': disk.percent,
        'model': 'Samsung EVO 870 Pro',  # Valor fixo para exemplo
        'type': 'NVMe SSD',  # Valor fixo para exemplo
        'capacity': get_size(disk.total)
    }

    # Network
    net = psutil.net_io_counters()
    network_info = {
        'upload': get_size(net.bytes_sent),
        'download': get_size(net.bytes_recv)
    }

    # System Info
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.now() - boot_time
    uptime_str = str(timedelta(seconds=int(uptime.total_seconds())))

    system_info = {
        'ip': socket.gethostbyname(socket.gethostname()),
        'os': f"{platform.system()} {platform.release()}",
        'version': 'v1.0',  # Versão do painel
        'status': 'Running',
        'uptime': uptime_str
    }

    return {
        'cpu': cpu_info,
        'ram': ram_info,
        'hdd': disk_info,
        'network': network_info,
        'system': system_info
    }

@app.get("/train/active")
async def get_active_training():
    active_job = None
    for job_id, job in _jobs.items():
        if job['status'] in ['running', 'paused']:
            active_job = {
                'job_id': job_id,
                'status': job['status'],
                'started_at': job['started_at'],
                'metrics': job.get('metrics'),
                'total_epochs': job.get('total_epochs'),
                'dataset_name': job.get('dataset_name')
            }
            break
    return {'active_job': active_job}

@app.post("/train/{job_id}/pause", dependencies=[Depends(require_auth)])
async def pause_training(job_id: str):
    """Pausar treinamento ativo"""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    job = _jobs[job_id]
    if job['status'] != 'running':
        raise HTTPException(status_code=400, detail="Job não está em execução")
    _jobs[job_id]['status'] = 'paused'
    _jobs[job_id]['paused_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    cancel_job_task(job_id)
    from .yolo_service import _persist_jobs
    _persist_jobs()
    return {"message": "Treinamento pausado com sucesso", "job_id": job_id}

@app.post("/train/{job_id}/resume", dependencies=[Depends(require_auth)])
async def resume_training(job_id: str):
    """Retomar treinamento pausado"""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    job = _jobs[job_id]
    if job['status'] != 'paused':
        raise HTTPException(status_code=400, detail="Job não está pausado")
    _jobs[job_id]['status'] = 'running'
    _jobs[job_id]['resumed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    # Relançar treinamento com resume=True usando os parâmetros anteriores
    data_yaml_path = job.get('data_yaml_path')
    params = (job.get('train_params') or {})
    params['resume'] = True
    if data_yaml_path:
        t = asyncio.create_task(start_training(job_id, data_yaml_path, params))
        register_job_task(job_id, t)
    from .yolo_service import _persist_jobs
    _persist_jobs()
    return {"message": "Treinamento retomado com sucesso", "job_id": job_id}

@app.post("/train/{job_id}/stop", dependencies=[Depends(require_auth)])
async def stop_training(job_id: str):
    """Parar treinamento definitivamente"""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    job = _jobs[job_id]
    if job['status'] not in ['running', 'paused']:
        raise HTTPException(status_code=400, detail="Job não pode ser parado")
    _jobs[job_id]['status'] = 'stopped'
    _jobs[job_id]['stopped_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    cancel_job_task(job_id)
    from .yolo_service import _persist_jobs
    _persist_jobs()
    return {"message": "Treinamento parado com sucesso", "job_id": job_id}

@app.post("/train/jobs/clear", dependencies=[Depends(require_auth)])
async def clear_jobs_endpoint():
    """Limpa o histórico de jobs e persiste em runs/jobs.json"""
    clear_jobs()
    return {"message": "Histórico de jobs limpo com sucesso"}