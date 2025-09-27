import os
import uuid
import asyncio
import time
import base64
from typing import List, Optional, Union

from fastapi import FastAPI, UploadFile, File, Body, Query, Request, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates

import httpx

from .config import settings
from .schemas import InferResponse, TrainParams, TrainStatus, ValidateRequest, ExportRequest, PromoteRequest, HealthResponse, MetricsResponse, UIConfigRequest, UIConfigResponse, WebhookLogResponse
from .yolo_service import infer as yolo_infer, start_training, validate_model, promote_model, export_model, _jobs, register_job_task, cancel_job_task, clear_jobs, _extract_zip, _jobs_lock, _persist_jobs
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

# Session middleware for UI login
# Ajuste de flags de cookie conforme ambiente
_cookie_kwargs = {
    "secret_key": settings.SESSION_SECRET,
    "session_cookie": "ev_session",
    "same_site": "lax",
}
if settings.APP_ENV.lower() in {"production", "prod", "staging"}:
    _cookie_kwargs["https_only"] = True
    _cookie_kwargs["same_site"] = "strict"
app.add_middleware(SessionMiddleware, **_cookie_kwargs)

# Static and templates
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
templates = Jinja2Templates(directory=os.path.join(settings.WEB_DIR, "templates"))

# ===== CSRF utilities (double-submit cookie) =====
import secrets
from starlette.datastructures import MutableHeaders

CSRF_COOKIE_NAME = "ev_csrf"
CSRF_HEADER_NAME = "x-csrf-token"

async def issue_csrf(response):
    token = secrets.token_urlsafe(32)
    # Cookie não HttpOnly para leitura pelo front
    response.set_cookie(CSRF_COOKIE_NAME, token, samesite=_cookie_kwargs.get("same_site", "lax"), secure=_cookie_kwargs.get("https_only", False))
    return token

async def verify_csrf(request: Request, token_from_form: Optional[str] = None):
    """Verifica token CSRF via double-submit (cookie + form/header)."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get(CSRF_HEADER_NAME)
    candidate = token_from_form or header_token
    if not cookie_token or not candidate or cookie_token != candidate:
        raise HTTPException(status_code=403, detail="CSRF token inválido")

# Dependência para exigir CSRF quando a autenticação é via sessão (sem Bearer)
async def require_csrf_if_session(request: Request):
    auth = request.headers.get("Authorization")
    if not auth and request.cookies.get(_cookie_kwargs.get("session_cookie", "ev_session")):
        await verify_csrf(request)

# Security dependencies
async def require_auth(request: Request):
    if not settings.AUTH_TOKEN:
        # If no token based auth configured, allow if session authenticated
        user = request.session.get("user")
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] != settings.AUTH_TOKEN:
        # Fall back to session auth
        user = request.session.get("user")
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
    # IP whitelist if configured
    client_ip = request.client.host if request.client else None
    if settings.IP_WHITELIST and client_ip not in settings.IP_WHITELIST:
        raise HTTPException(status_code=403, detail="Forbidden IP")

# Enforce secure admin password in production
if settings.APP_ENV.lower() in {"production", "prod", "staging"}:
    if not settings.ADMIN_PASSWORD_HASH:
        raise RuntimeError("ADMIN_PASSWORD_HASH é obrigatório em produção. Não use ADMIN_PASSWORD em produção.")

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
    # Se request contém cookie de sessão, exigir CSRF
    if request.cookies.get(_cookie_kwargs.get("session_cookie", "ev_session")):
        await verify_csrf(request)
    # Limitar quantidade de arquivos
    if not files or len(files) > 30:
        raise HTTPException(status_code=400, detail="Envie entre 1 e 30 imagens")
    # Gather local paths
    paths: List[str] = []
    infer_dir = os.path.join(settings.DATA_DIR, 'infer')
    os.makedirs(infer_dir, exist_ok=True)
    # Uploads - validar MIME e sanitizar nomes
    allowed_mimes = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    max_img_mb = int(os.getenv("MAX_IMAGE_MB", "10"))  # limite por imagem
    for f in files:
        if not f.content_type or f.content_type.lower() not in allowed_mimes:
            raise HTTPException(status_code=400, detail=f"Tipo de arquivo não suportado: {f.content_type}")
        import re
        fid = uuid.uuid4().hex
        base_name = os.path.basename(f.filename or "image.jpg")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)
        out_path = os.path.join(infer_dir, f"{fid}-{safe_name}")
        # Escrever em chunks e limitar tamanho
        size = 0
        with open(out_path, 'wb') as w:
            while True:
                chunk = await f.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                size += len(chunk)
                if size > max_img_mb * 1024 * 1024:
                    try:
                        w.close()
                    finally:
                        try:
                            os.remove(out_path)
                        except Exception:
                            pass
                    raise HTTPException(status_code=413, detail=f"Imagem excede {max_img_mb}MB")
                w.write(chunk)
        paths.append(out_path)
    # URLs - apenas http(s) e content-type imagem e até 10MB
    if urls:
        async with httpx.AsyncClient(timeout=20) as client:
            for u in urls:
                if not (u.startswith('http://') or u.startswith('https://')):
                    continue
                fid = uuid.uuid4().hex
                out_path = os.path.join(infer_dir, f"{fid}.jpg")
                try:
                    resp = await client.get(u)
                    resp.raise_for_status()
                    ctype = resp.headers.get('Content-Type', '')
                    if 'image' not in ctype.lower():
                        continue
                    cl = resp.headers.get('Content-Length')
                    if cl and int(cl) > 10_000_000:
                        continue
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

@app.post("/config", response_model=UIConfigResponse, dependencies=[Depends(require_auth), Depends(require_csrf_if_session)])
async def set_ui_config(req: UIConfigRequest):
    cfg = save_config(req.model_dump(exclude_none=True))
    return UIConfigResponse(**cfg)

@app.get("/panel", include_in_schema=False)
async def panel(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/login", status_code=303)
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
    response = templates.TemplateResponse("panel.html", {
        "request": request,
        "app_name": settings.APP_NAME,
        "active_model": settings.ACTIVE_MODEL_PATH if os.path.exists(settings.ACTIVE_MODEL_PATH) else None,
        "detections": recent,
        "jobs": list(_jobs.values())[::-1],
        "cfg": cfg,
        "n8n_webhook": cfg.get("N8N_WEBHOOK_URL"),
        "base_url": str(request.base_url)
    })
    # emitir CSRF
    await issue_csrf(response)
    return response

@app.post("/panel/upload", include_in_schema=False, dependencies=[Depends(require_auth), Depends(require_csrf_if_session)])
async def panel_upload(
    request: Request,
    dataset: UploadFile = File(...),
    epochs: int = Form(50),
    imgsz: int = Form(640),
    batch: int = Form(16),
    csrf_token: str = Form(None)
):
    # Verifica CSRF explícito (double-submit)
    await verify_csrf(request, csrf_token)

    # Validar tipo de arquivo e extensão
    content_type = (dataset.content_type or "").lower()
    allowed_types = {"application/zip", "application/x-zip-compressed", "multipart/x-zip", "application/x-zip"}
    filename = dataset.filename or "dataset.zip"
    if not filename.lower().endswith(".zip") or (content_type and content_type not in allowed_types):
        raise HTTPException(status_code=400, detail="Envie um arquivo .zip válido")

    # Sanitizar nome de arquivo
    import re
    base = os.path.basename(filename)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base)

    # Limitar tamanho lendo em chunks
    max_mb = int(os.getenv("MAX_DATASET_ZIP_MB", "1024"))  # default 1GB
    zips_dir = os.path.join(settings.DATA_DIR, "zips")
    os.makedirs(zips_dir, exist_ok=True)
    zip_path = os.path.join(zips_dir, f"{uuid.uuid4().hex}-{safe_name}")

    size = 0
    with open(zip_path, "wb") as out:
        while True:
            chunk = await dataset.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            size += len(chunk)
            if size > max_mb * 1024 * 1024:
                out.close()
                try:
                    os.remove(zip_path)
                except Exception:
                    pass
                raise HTTPException(status_code=413, detail=f"Tamanho do ZIP excede {max_mb}MB")
            out.write(chunk)

    # Checagem básica contra path traversal no ZIP
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for n in zf.namelist():
                norm = n.replace("\\", "/")
                if os.path.isabs(n) or ".." in norm.split("/"):
                    raise HTTPException(status_code=400, detail="ZIP inválido (path traversal detectado)")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="ZIP corrompido ou inválido")

    # Extrair e iniciar treinamento
    job_id = uuid.uuid4().hex
    extract_dir = os.path.join(settings.DATA_DIR, "extracted", job_id)
    os.makedirs(extract_dir, exist_ok=True)
    data_yaml_path = _extract_zip(zip_path, extract_dir)

    params = {"epochs": epochs, "imgsz": imgsz, "batch": batch}
    task = asyncio.create_task(start_training(job_id, data_yaml_path, params))
    register_job_task(job_id, task)

    return RedirectResponse(url="/panel", status_code=303)

@app.get("/settings", include_in_schema=False)
async def settings_page(request: Request):
    if not request.session.get("user"):
        return RedirectResponse(url="/login", status_code=303)
    cfg = load_config()
    from .state import get_webhook_logs
    webhook_logs = get_webhook_logs(50)
    response = templates.TemplateResponse("settings.html", {
        "request": request,
        "app_name": settings.APP_NAME,
        "active_model": settings.ACTIVE_MODEL_PATH if os.path.exists(settings.ACTIVE_MODEL_PATH) else None,
        "cfg": cfg,
        "webhook_logs": webhook_logs,
        "base_url": str(request.base_url)
    })
    await issue_csrf(response)
    return response

@app.post("/webhook/send", dependencies=[Depends(require_auth), Depends(require_csrf_if_session)])
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

    # GPU/CUDA (opcional)
    gpu_info = {
        'available': False,
        'device_count': 0,
        'devices': []
    }
    try:
        import torch  # type: ignore
        has_cuda = torch.cuda.is_available()
        gpu_info['available'] = bool(has_cuda)
        if has_cuda:
            device_count = torch.cuda.device_count()
            gpu_info['device_count'] = int(device_count)
            # Tentar NVML para métricas extras
            nvml = None
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                nvml = pynvml
            except Exception:
                nvml = None
            for i in range(device_count):
                # Memória
                total_mem = None
                free_mem = None
                try:
                    # Preferir mem_get_info quando disponível
                    free_b, total_b = torch.cuda.mem_get_info(i)  # type: ignore[attr-defined]
                    total_mem = total_b
                    free_mem = free_b
                except Exception:
                    try:
                        props = torch.cuda.get_device_properties(i)
                        total_mem = getattr(props, 'total_memory', None)
                        # Aproximação usando reserved/allocated
                        reserved = torch.cuda.memory_reserved(i)
                        allocated = torch.cuda.memory_allocated(i)
                        free_mem = max(0, (total_mem or 0) - max(reserved, allocated))
                    except Exception:
                        pass
                # Nome
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = f"CUDA:{i}"
                # Utilização e temperatura via NVML
                utilization = None
                temperature = None
                mem_util = None
                if nvml is not None:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        utilization = int(util.gpu)
                        mem_util = int(util.memory)
                        temperature = int(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
                    except Exception:
                        pass
                gpu_info['devices'].append({
                    'id': i,
                    'name': name,
                    'total_mem_bytes': int(total_mem) if total_mem is not None else None,
                    'free_mem_bytes': int(free_mem) if free_mem is not None else None,
                    'total_mem': get_size(total_mem) if total_mem is not None else None,
                    'free_mem': get_size(free_mem) if free_mem is not None else None,
                    'utilization': utilization,  # %
                    'mem_utilization': mem_util,  # %
                    'temperature': temperature  # °C
                })
            # Versão CUDA detectada pelo Torch
            try:
                gpu_info['cuda_version'] = getattr(torch.version, 'cuda', None)
            except Exception:
                pass
            # Último dispositivo ativo
            try:
                gpu_info['current_device'] = int(torch.cuda.current_device())
            except Exception:
                pass
            # Fechar NVML
            try:
                if nvml is not None:
                    nvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        # Torch não disponível ou erro inesperado
        pass

    return {
        'cpu': cpu_info,
        'ram': ram_info,
        'hdd': disk_info,
        'network': network_info,
        'system': system_info,
        'gpu': gpu_info
    }

@app.post("/train/{job_id}/pause", dependencies=[Depends(require_auth), Depends(require_csrf_if_session)])
async def pause_training(job_id: str):
    """Pausar treinamento ativo"""
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job não encontrado")
        job = _jobs[job_id]
        if job['status'] != 'running':
            raise HTTPException(status_code=400, detail="Job não está em execução")
        _jobs[job_id]['status'] = 'paused'
        _jobs[job_id]['paused_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        _persist_jobs()
    cancel_job_task(job_id)
    return {"message": "Treinamento pausado com sucesso", "job_id": job_id}

@app.post("/train/{job_id}/resume", dependencies=[Depends(require_auth), Depends(require_csrf_if_session)])
async def resume_training(job_id: str):
    """Retomar treinamento pausado"""
    # Capturar parâmetros sob trava e depois lançar treino fora da trava
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job não encontrado")
        job = _jobs[job_id]
        if job['status'] != 'paused':
            raise HTTPException(status_code=400, detail="Job não está pausado")
        _jobs[job_id]['status'] = 'running'
        _jobs[job_id]['resumed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        data_yaml_path = job.get('data_yaml_path')
        params = (job.get('train_params') or {}).copy()
        params['resume'] = True
        _persist_jobs()
    if data_yaml_path:
        t = asyncio.create_task(start_training(job_id, data_yaml_path, params))
        register_job_task(job_id, t)
    return {"message": "Treinamento retomado com sucesso", "job_id": job_id}

@app.post("/train/{job_id}/stop", dependencies=[Depends(require_auth), Depends(require_csrf_if_session)])
async def stop_training(job_id: str):
    """Parar treinamento definitivamente"""
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job não encontrado")
        job = _jobs[job_id]
        if job['status'] not in ['running', 'paused']:
            raise HTTPException(status_code=400, detail="Job não pode ser parado")
        _jobs[job_id]['status'] = 'stopped'
        _jobs[job_id]['stopped_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        _persist_jobs()
    cancel_job_task(job_id)
    return {"message": "Treinamento parado com sucesso", "job_id": job_id}

@app.post("/train/jobs/clear", dependencies=[Depends(require_auth), Depends(require_csrf_if_session)])
async def clear_jobs_endpoint():
    """Limpa o histórico de jobs e persiste em runs/jobs.json"""
    clear_jobs()
    return {"message": "Histórico de jobs limpo com sucesso"}

@app.get("/i/{image_id}.jpg", include_in_schema=False)
async def short_image(image_id: str):
    path = os.path.join(settings.STATIC_DIR, "detections", f"{image_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Imagem não encontrada")
    return FileResponse(path, media_type="image/jpeg")

@app.get("/train/{job_id}")
async def get_training_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job não encontrado")
        return dict(job)


@app.get("/login", include_in_schema=False)
async def login_page(request: Request):
    response = templates.TemplateResponse("login.html", {"request": request, "app_name": settings.APP_NAME})
    await issue_csrf(response)
    return response

@app.get("/", include_in_schema=False)
async def root(request: Request):
    """Redireciona a raiz para /login (anônimo) ou /panel (logado)."""
    if not request.session.get("user"):
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/panel", status_code=303)

@app.post("/login", include_in_schema=False)
async def login_submit(request: Request, email: str = Form(...), password: str = Form(...), csrf_token: str = Form(None)):
    await verify_csrf(request, csrf_token)
    from passlib.hash import bcrypt
    # Rate limiting por IP+email
    ip = request.client.host if request.client else 'unknown'
    key = f"{ip}:{(email or '').lower()}"
    now = time.time()
    entry = _login_attempts.get(key) or {"fails": [], "locked_until": 0}
    if entry.get("locked_until", 0) > now:
        wait = int(entry["locked_until"] - now)
        response = templates.TemplateResponse("login.html", {"request": request, "app_name": settings.APP_NAME, "error": f"Muitas tentativas. Tente novamente em {wait}s"}, status_code=429)
        await issue_csrf(response)
        return response

    valid_email = settings.ADMIN_EMAIL
    # Prefer hash; fallback to plain password for development
    if settings.ADMIN_PASSWORD_HASH:
        is_ok = (email.lower() == valid_email.lower()) and bcrypt.verify(password, settings.ADMIN_PASSWORD_HASH)
    else:
        is_ok = (email.lower() == valid_email.lower()) and (settings.ADMIN_PASSWORD is not None) and (password == settings.ADMIN_PASSWORD)

    if not is_ok:
        # registrar falha e bloquear se necessário
        fails = [t for t in entry["fails"] if now - t <= _LOGIN_WINDOW_SEC]
        fails.append(now)
        entry["fails"] = fails
        if len(fails) >= _LOGIN_MAX_FAILS:
            entry["locked_until"] = now + _LOGIN_LOCK_SEC
        _login_attempts[key] = entry
        response = templates.TemplateResponse("login.html", {"request": request, "app_name": settings.APP_NAME, "error": "Credenciais inválidas"}, status_code=401)
        await issue_csrf(response)
        return response

    # sucesso: limpar tentativas
    if key in _login_attempts:
        _login_attempts.pop(key, None)
    request.session["user"] = {"email": email}
    return RedirectResponse(url="/panel", status_code=303)

@app.post("/logout", include_in_schema=False)
async def logout(request: Request, csrf_token: str = Form(None)):
    await verify_csrf(request, csrf_token)
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)

# ===== Account management (change admin email/password) =====
@app.post("/settings/account", include_in_schema=False, dependencies=[Depends(require_auth)])
async def update_account(request: Request, new_email: Optional[str] = Form(None), current_password: Optional[str] = Form(None), new_password: Optional[str] = Form(None), csrf_token: str = Form(None)):
    """Atualiza e-mail e/ou senha do administrador.
    - Requer sessão autenticada.
    - Verifica CSRF.
    - Para alterar senha, exige current_password válido.
    - Persiste em variáveis de ambiente efetivas para a sessão atual (em memória) e recomenda configurar variáveis no ambiente para persistência entre reinícios.
    """
    await verify_csrf(request, csrf_token)
    from passlib.hash import bcrypt

    # Validar nova senha se fornecida
    if new_password and (len(new_password) < 8):
        raise HTTPException(status_code=400, detail="Senha deve ter pelo menos 8 caracteres")

    # Verificação da senha atual se for alterar senha
    if new_password:
        # Checar contra hash se existir; senão contra plain dev password
        ok = False
        if settings.ADMIN_PASSWORD_HASH:
            try:
                ok = current_password is not None and bcrypt.verify(current_password, settings.ADMIN_PASSWORD_HASH)
            except Exception:
                ok = False
        else:
            ok = (settings.ADMIN_PASSWORD is not None) and (current_password == settings.ADMIN_PASSWORD)
        if not ok:
            raise HTTPException(status_code=401, detail="Senha atual inválida")
        # Gerar hash e aplicar na instância
        new_hash = bcrypt.hash(new_password)
        settings.ADMIN_PASSWORD_HASH = new_hash
        settings.ADMIN_PASSWORD = None  # desabilitar plain

    # Atualizar e-mail se fornecido
    if new_email:
        if "@" not in new_email:
            raise HTTPException(status_code=400, detail="E-mail inválido")
        settings.ADMIN_EMAIL = new_email
        # Se usuário logado trocou e-mail, atualizar sessão também
        if request.session.get("user"):
            request.session["user"]["email"] = new_email

    # Redirecionar de volta às configurações
    return RedirectResponse(url="/settings", status_code=303)

@app.get("/train/active", dependencies=[Depends(require_auth)])
async def get_active_training():
    """Retorna o job de treinamento ativo (status running ou paused).
    Retorna 404 quando não houver job ativo.
    """
    def ts_of(job: dict) -> str:
        # Usa ordem de preferência para timestamp, mantendo formato YYYY-MM-DD HH:MM:SS
        for k in ("resumed_at", "paused_at", "started_at"):
            if job.get(k):
                return str(job[k])
        return job.get("started_at") or ""

    with _jobs_lock:
        candidates = [dict(j) for j in _jobs.values() if j.get("status") in ("running", "paused")]
        if not candidates:
            raise HTTPException(status_code=404, detail="Nenhum treinamento ativo")
        candidates.sort(key=lambda j: ts_of(j), reverse=True)
        return {"active_job": candidates[0]}

# === Login rate limiting globals ===
# Janela de observação (segundos), número máximo de falhas, e tempo de bloqueio após exceder
_LOGIN_WINDOW_SEC = int(os.getenv("LOGIN_WINDOW_SEC", "300"))  # 5 min
_LOGIN_MAX_FAILS = int(os.getenv("LOGIN_MAX_FAILS", "5"))
_LOGIN_LOCK_SEC = int(os.getenv("LOGIN_LOCK_SEC", "900"))     # 15 min
# Mapa: key = "<ip>:<email>", value = {"fails": [timestamps], "locked_until": epoch_seconds}
_login_attempts: dict = {}