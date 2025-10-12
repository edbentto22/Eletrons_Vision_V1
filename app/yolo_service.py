import os
import uuid
import time
import json
import shutil
import asyncio
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
import httpx
from threading import RLock

from .config import settings
from .state import load_config

# Lazy import ultralytics and torch inside functions to allow server start without them

def _load_yolo_model(device: Optional[str] = None) -> Any:
    from ultralytics import YOLO  # type: ignore
    model_path = settings.ACTIVE_MODEL_PATH if os.path.exists(settings.ACTIVE_MODEL_PATH) else settings.MODEL_VARIANT
    model = YOLO(model_path)
    if device:
        model.to(device)
    return model

def _abs_url(path: Optional[str]) -> Optional[str]:
    """Build absolute URL from a relative path using PUBLIC_BASE_URL if available."""
    if not path:
        return None
    base = settings.PUBLIC_BASE_URL
    if not base:
        return path
    return f"{base.rstrip('/')}/{path.lstrip('/')}"

async def send_n8n_webhook(payload: Dict[str, Any]) -> None:
     cfg = load_config()
     
     # Verificar se webhook está habilitado
     if not cfg.get("N8N_WEBHOOK_ENABLED", True):
         return
     
     # Obter URL do webhook
     url = cfg.get("N8N_WEBHOOK_URL") or settings.N8N_WEBHOOK_URL
     if not url:
         return
     
     # Verificar se deve incluir imagem em base64
     if cfg.get("N8N_WEBHOOK_INCLUDE_IMAGE", False) and "results" in payload:
         for result in payload["results"]:
             if result.get("annotated_url"):
                 try:
                     # Carregar imagem anotada e converter para base64
                     img_path = os.path.join(settings.STATIC_DIR, "detections", os.path.basename(result["annotated_url"]))
                     if os.path.exists(img_path):
                         import base64
                         with open(img_path, "rb") as img_file:
                             img_data = base64.b64encode(img_file.read()).decode('utf-8')
                             result["annotated_base64"] = f"data:image/jpeg;base64,{img_data}"
                 except Exception:
                     pass
     
     # Enviar webhook
     try:
         from .state import log_webhook
         async with httpx.AsyncClient(timeout=10) as client:
             response = await client.post(url, json=payload)
             # Registrar log do webhook
             log_webhook("out", url, response.status_code, payload)
     except Exception as e:
         # Registrar falha
         from .state import log_webhook
         log_webhook("out", url, 0, {"error": str(e)})

def _result_to_boxes(result: Any) -> Tuple[List[Dict[str, Any]], int, int]:
     boxes_out: List[Dict[str, Any]] = []
     # Ultralytics Results: result.boxes.xyxy, result.boxes.conf, result.boxes.cls
     try:
         xyxy = result.boxes.xyxy.cpu().numpy()
         conf = result.boxes.conf.cpu().numpy()
         cls = result.boxes.cls.cpu().numpy().astype(int)
         labels = result.names if hasattr(result, 'names') else None
         for i in range(len(xyxy)):
             x1, y1, x2, y2 = xyxy[i].tolist()
             boxes_out.append({
                 'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
                 'conf': float(conf[i]), 'cls': int(cls[i]), 'label': labels.get(int(cls[i])) if isinstance(labels, dict) else None
             })
     except Exception:
         pass
     # Get image size
     try:
         h, w = result.orig_shape[0], result.orig_shape[1]
     except Exception:
         h, w = 0, 0
     return boxes_out, w, h

def _annotate_and_save(result: Any, image_id: str) -> Optional[str]:
    if not settings.SAVE_ANNOTATIONS:
        return None
    try:
        annotated = result.plot()
        out_dir = os.path.join(settings.STATIC_DIR, "detections")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{image_id}.jpg")
        cv2.imwrite(out_path, annotated)
        return f"/static/detections/{image_id}.jpg"
    except Exception:
        return None

async def infer(paths: List[str], conf: float, iou: float, imgsz: int, device: Optional[str], send_webhook: bool = True, extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    model = _load_yolo_model(device)
    t0 = time.time()
    results = model.predict(paths, conf=conf, iou=iou, imgsz=imgsz, device=device, stream=False, verbose=False)
    elapsed = (time.time() - t0) * 1000.0
    payload_results = []
    for path, res in zip(paths, results):
        image_id = uuid.uuid4().hex
        boxes, w, h = _result_to_boxes(res)
        annotated_url = _annotate_and_save(res, image_id)
        # Copiar a imagem de origem para static/uploads para servir publicamente
        try:
            ext = os.path.splitext(path)[1].lower() or ".jpg"
            up_dir = os.path.join(settings.STATIC_DIR, "uploads")
            os.makedirs(up_dir, exist_ok=True)
            dest_name = f"{image_id}{ext}"
            dest_path = os.path.join(up_dir, dest_name)
            shutil.copyfile(path, dest_path)
            source_url_rel = f"/static/uploads/{dest_name}"
        except Exception:
            source_url_rel = None

        # Montar URL curta absoluta para imagem anotada
        short_annot_rel = f"/i/{image_id}.jpg" if annotated_url else None
        annotated_url_abs = _abs_url(short_annot_rel) if short_annot_rel else None
        source_url_abs = _abs_url(source_url_rel) if source_url_rel else None
        payload_results.append({
            'image_id': image_id,
            'source': path,
            'source_url': source_url_abs or source_url_rel,
            'width': w,
            'height': h,
            'detections': boxes,
            'annotated_url': annotated_url_abs or short_annot_rel
        })

    # Persist lightweight log
    log_path = os.path.join(settings.DATA_DIR, 'infer', 'log.json')
    try:
        existing = []
        if os.path.exists(log_path):
            existing = json.load(open(log_path))
        existing = (existing + payload_results)[-50:]
        json.dump(existing, open(log_path, 'w'))
    except Exception:
        pass
    # Send webhook summarized (optional)
    if send_webhook:
        webhook_payload: Dict[str, Any] = {
            'summary': {
                'count': len(payload_results),
                'latency_ms': elapsed
            },
            'results': payload_results
        }
        if extra_meta:
            webhook_payload['meta'] = extra_meta
        asyncio.create_task(send_n8n_webhook(webhook_payload))
    return {
        'count': len(payload_results),
        'results': payload_results,
        'params': {'conf': conf, 'iou': iou, 'imgsz': imgsz, 'device': device},
        'latency_ms': elapsed,
        **({'meta': extra_meta} if extra_meta else {})
    }

# ---------- Training Management ----------
_jobs: Dict[str, Dict[str, Any]] = {}
_job_tasks: Dict[str, asyncio.Task] = {}
_jobs_lock: RLock = RLock()


def register_job_task(job_id: str, task: asyncio.Task) -> None:
    """Registra a task async responsável pelo treinamento de um job."""
    _job_tasks[job_id] = task


def cancel_job_task(job_id: str) -> bool:
    """Cancela a task do job se possível. Observação: Ultralytics treina em loop de épocas; a parada efetiva acontece no callback."""
    task = _job_tasks.get(job_id)
    if task and not task.done():
        try:
            task.cancel()
        except Exception:
            pass
        return True
    return False


def clear_jobs() -> None:
    """Limpa o histórico de jobs e cancela tasks ativas."""
    # Cancelar tasks
    for jid, t in list(_job_tasks.items()):
        try:
            t.cancel()
        except Exception:
            pass
        _job_tasks.pop(jid, None)
    # Limpar histórico
    with _jobs_lock:
        _jobs.clear()
        _persist_jobs()

def _extract_zip(zip_path: str, dest_dir: str) -> str:
    import zipfile
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    
    # Verificar se já existe data.yaml
    for root, _, files in os.walk(dest_dir):
        if 'data.yaml' in files:
            return os.path.join(root, 'data.yaml')
    
    # Se não existir data.yaml, criar um
    print(f"data.yaml não encontrado no ZIP. Criando automaticamente...")
    
    # Verificar estrutura do dataset
    images_dir = os.path.join(dest_dir, 'images')
    labels_dir = os.path.join(dest_dir, 'labels')
    classes_file = os.path.join(dest_dir, 'classes.txt')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise ValueError(f"Estrutura de diretórios inválida. Esperado: images/ e labels/")
    
    # Ler classes
    classes = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
    
    if not classes:
        # Tentar inferir classes a partir dos arquivos de label
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        if label_files:
            class_ids = set()
            for label_file in label_files[:10]:  # Verificar apenas os primeiros 10 arquivos
                try:
                    with open(os.path.join(labels_dir, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                except Exception:
                    pass
            
            # Criar nomes genéricos para as classes
            classes = [f"class_{i}" for i in range(max(class_ids) + 1)]
    
    # Obter caminhos absolutos
    abs_dir = os.path.abspath(dest_dir)
    abs_images = os.path.join(abs_dir, 'images')
    
    # Criar data.yaml
    data_yaml_path = os.path.join(dest_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {abs_dir}\n")
        f.write(f"train: {abs_images}\n")
        f.write(f"val: {abs_images}\n")
        f.write(f"test: {abs_images}\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")
    
    print(f"data.yaml criado em {data_yaml_path}")
    return data_yaml_path

def _persist_jobs():
    path = os.path.join(settings.RUNS_DIR, 'jobs.json')
    try:
        with _jobs_lock:
            json.dump(_jobs, open(path, 'w'))
    except Exception:
        pass

def _load_jobs():
    global _jobs
    path = os.path.join(settings.RUNS_DIR, 'jobs.json')
    if os.path.exists(path):
        try:
            with _jobs_lock:
                _jobs = json.load(open(path))
        except Exception:
            _jobs = {}

_load_jobs()

async def start_training(job_id: str, data_yaml_path: str, params: Dict[str, Any]) -> None:
    with _jobs_lock:
        _jobs[job_id] = {
            'job_id': job_id,
            'status': 'running',
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'best_pt': None,
            'metrics': None,
            'total_epochs': int(params.get('epochs', 50)),
            'dataset_name': os.path.basename(data_yaml_path).replace('.yaml', ''),
            'data_yaml_path': data_yaml_path,
            'train_params': params,
        }
        _persist_jobs()
    try:
        from ultralytics import YOLO  # type: ignore
        model_variant = params.get('model_variant') or settings.MODEL_VARIANT
        model = YOLO(model_variant)
        run_name = job_id
        train_args = {
            'data': data_yaml_path,
            'epochs': int(params.get('epochs', 50)),
            'imgsz': int(params.get('imgsz', 640)),
            'lr0': float(params.get('lr0', 0.01)),
            'batch': int(params.get('batch', 16)),
            'device': params.get('device'),
            'project': settings.RUNS_DIR,
            'name': run_name,
            'resume': bool(params.get('resume', False)),
            'pretrained': bool(params.get('pretrained', True)),
            'plots': True,  # Gerar gráficos
            'save': True,  # Salvar checkpoints
            'save_period': -1,  # Salvar apenas o melhor modelo
            'val': True,  # Executar validação
        }
        
        # Métricas por época
        metrics_history = {
            'box_loss_per_epoch': [],
            'cls_loss_per_epoch': [],
            'metrics': {
                'mAP50_per_epoch': [],
                'mAP50_95_per_epoch': [],
                'per_class': {}
            }
        }
        
        # Callback para coletar métricas e checar sinal de pausa/parada
        def on_train_epoch_end(trainer):
            try:
                metrics = trainer.metrics
                metrics_history['box_loss_per_epoch'].append(float(metrics.get('train/box_loss', 0)))
                metrics_history['cls_loss_per_epoch'].append(float(metrics.get('train/cls_loss', 0)))
                
                if metrics.get('metrics/mAP50(B)') is not None:
                    metrics_history['metrics']['mAP50_per_epoch'].append(float(metrics['metrics/mAP50(B)']))
                if metrics.get('metrics/mAP50-95(B)') is not None:
                    metrics_history['metrics']['mAP50_95_per_epoch'].append(float(metrics['metrics/mAP50-95(B)']))
                
                # Métricas por classe
                try:
                    for i, (cls_name, cls_metrics) in enumerate(zip(model.names, metrics.get('metrics/per_class', []))):
                        if cls_name not in metrics_history['metrics']['per_class']:
                            metrics_history['metrics']['per_class'][cls_name] = {
                                'precision': float(cls_metrics[0]),
                                'recall': float(cls_metrics[1]),
                                'mAP50': float(cls_metrics[2]),
                                'mAP50_95': float(cls_metrics[3])
                            }
                except Exception:
                    pass
                
                # Atualizar job com métricas parciais
                metrics_history['epochs'] = trainer.epoch + 1
                metrics_history['train_box_loss'] = float(metrics.get('train/box_loss', 0))
                metrics_history['train_cls_loss'] = float(metrics.get('train/cls_loss', 0))
                metrics_history['metrics']['mAP50'] = float(metrics.get('metrics/mAP50(B)', 0))
                metrics_history['metrics']['mAP50_95'] = float(metrics.get('metrics/mAP50-95(B)', 0))
                
                with _jobs_lock:
                    if job_id in _jobs:
                        _jobs[job_id].update({
                            'metrics': metrics_history
                        })
                        _persist_jobs()

                # Checar pedido de pausa/parada para encerrar ao fim da época
                with _jobs_lock:
                    status_now = _jobs.get(job_id, {}).get('status')
                if status_now in ('paused', 'stopped'):
                    try:
                        setattr(trainer, 'stop', True)  # Ultralytics Trainer aceita sinal de parada
                    except Exception:
                        pass
            except Exception as e:
                print(f"Erro ao coletar métricas: {e}")

        # Registrar callback
        model.add_callback('on_train_epoch_end', on_train_epoch_end)
        
        # Treinar modelo em thread para não bloquear event loop
        await asyncio.to_thread(model.train, **train_args)
        
        # Caminhos de saída
        # Compatibilidade com layouts diferentes do Ultralytics (com ou sem subpasta 'detect')
        candidate1 = os.path.join(settings.RUNS_DIR, 'detect', run_name, 'weights', 'best.pt')
        candidate2 = os.path.join(settings.RUNS_DIR, run_name, 'weights', 'best.pt')
        best_pt_src = candidate1 if os.path.exists(candidate1) else (candidate2 if os.path.exists(candidate2) else None)
        
        # Salvar em histórico
        hist_dir = os.path.join(settings.MODELS_DIR, 'history', run_name)
        os.makedirs(hist_dir, exist_ok=True)
        with _jobs_lock:
            if best_pt_src and os.path.exists(best_pt_src):
                shutil.copy2(best_pt_src, os.path.join(hist_dir, 'best.pt'))
                _jobs[job_id]['best_pt'] = os.path.join(hist_dir, 'best.pt')
            else:
                _jobs[job_id]['best_pt'] = None
        
        # Decidir status final
        with _jobs_lock:
            final_status = _jobs.get(job_id, {}).get('status', 'running')
            if final_status == 'running':
                _jobs[job_id].update({
                    'status': 'completed',
                    'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'metrics': metrics_history
                })
            elif final_status == 'stopped':
                _jobs[job_id].update({
                    'status': 'stopped',
                    'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'metrics': metrics_history
                })
            else:
                # paused: manter sem finished_at
                _jobs[job_id].update({
                    'status': 'paused',
                    'metrics': metrics_history
                })
            _persist_jobs()
    except Exception as e:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update({'status': 'failed', 'error': str(e), 'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')})
            _persist_jobs()

async def validate_model(data_yaml_path: str, device: Optional[str] = None) -> Dict[str, Any]:
    from ultralytics import YOLO  # type: ignore
    model = YOLO(settings.ACTIVE_MODEL_PATH if os.path.exists(settings.ACTIVE_MODEL_PATH) else settings.MODEL_VARIANT)
    metrics = model.val(data=data_yaml_path, device=device, project=settings.RUNS_DIR, name=f"val-{uuid.uuid4().hex}")
    # metrics is a dict-like
    return dict(metrics) if hasattr(metrics, 'items') else {}

async def promote_model(job_id: Optional[str], best_pt_path: Optional[str]) -> str:
    # Tentar resolver caminho do best.pt
    resolved_best = best_pt_path
    if not resolved_best and job_id:
        # Primeiro tenta o histórico
        candidate_hist = os.path.join(settings.MODELS_DIR, 'history', job_id, 'best.pt')
        resolved_best = candidate_hist if os.path.exists(candidate_hist) else None
        # Se não existir no histórico, tenta diretamente em runs/<job_id>/weights/best.pt
        if not resolved_best:
            candidate_runs1 = os.path.join(settings.RUNS_DIR, 'detect', job_id, 'weights', 'best.pt')
            candidate_runs2 = os.path.join(settings.RUNS_DIR, job_id, 'weights', 'best.pt')
            if os.path.exists(candidate_runs1):
                resolved_best = candidate_runs1
            elif os.path.exists(candidate_runs2):
                resolved_best = candidate_runs2
    if not resolved_best or not os.path.exists(resolved_best):
        raise FileNotFoundError("best.pt não encontrado para promoção")
    os.makedirs(os.path.dirname(settings.ACTIVE_MODEL_PATH), exist_ok=True)
    shutil.copy2(resolved_best, settings.ACTIVE_MODEL_PATH)
    return settings.ACTIVE_MODEL_PATH

async def export_model(fmt: str, half: bool, dynamic: bool, device: Optional[str]) -> str:
    from ultralytics import YOLO  # type: ignore
    model = YOLO(settings.ACTIVE_MODEL_PATH if os.path.exists(settings.ACTIVE_MODEL_PATH) else settings.MODEL_VARIANT)
    out_dir = os.path.join(settings.MODELS_DIR, 'onnx' if fmt == 'onnx' else 'trt')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"model_{int(time.time())}.{fmt}")
    model.export(format=fmt, half=half, dynamic=dynamic, device=device, imgsz=640, optimize=True)
    # Ultralytics saves into model dir; move/rename if produced
    produced = None
    for f in os.listdir(os.getcwd()):
        if f.endswith(f'.{fmt}'):
            produced = os.path.abspath(f)
            break
    if produced:
        shutil.move(produced, out_path)
    return out_path