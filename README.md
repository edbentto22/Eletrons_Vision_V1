# Eletrons Vision V1

Sistema de Visão Computacional com FastAPI + Ultralytics YOLO para detecção de componentes em postes (caixa de derivação, drop, fibra óptica, etc.). Pipeline pronto para execução local, Docker e cloud.

## Destaques
- API REST em FastAPI com endpoints `/health`, `/infer` e `/panel`.
- Modelo YOLO carregado de `app/models/production.pt` (Ultralytics 8.3.10, Torch 2.8.0). 
- Suporte a inferência em imagens locais e URLs.
- Preparado para versionamento com Git LFS para pesos de modelos.

## Requisitos
- Python 3.11+
- (Opcional) NVIDIA GPU com CUDA 12.x e cuDNN compatíveis
- macOS ou Linux

### Hardware
- Mínimo: CPU 4 cores, 8 GB RAM
- Ideal: GPU NVIDIA RTX (>= 8 GB), 16+ GB RAM

## Instalação
```bash
# Clonar repositório
git clone https://github.com/edbentto22/Eletrons_Vision_V1.git
cd Eletrons_Vision_V1

# Criar ambiente e instalar dependências
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Execução local (Uvicorn)
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
# Painel
# http://127.0.0.1:8001/panel
```

## Docker
```bash
# Build
docker build -t eletrons-vision:v1 .
# Run
docker run --rm -it -p 8001:8001 eletrons-vision:v1
```

## Estrutura
```
app/
  main.py
  yolo_service.py
  config.py
  data/
  models/
  web/
YOLOv8/
requirements.txt
Dockerfile
```

## Endpoints
- GET `/health`: status do servidor, versão Ultralytics/Torch, modelo ativo, disponibilidade de CUDA.
- POST `/infer`: corpo com `{"img_path": "path|url", "conf": 0.25, "iou": 0.45, "imgsz": 640, "device": "cpu|cuda:0", "registro": true}`.
- GET `/panel`: painel web simples para navegação.

### Exemplo de inferência
```bash
curl -X POST "http://127.0.0.1:8001/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "img_path": "app/data/postes_dataset/images/060206c6-ONET.JPG",
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640,
    "device": "cpu",
    "registro": true
  }'
```

Resposta esperada (exemplo):
```json
{
  "count": 1,
  "detections": [
    {"class": "caixa_derivacao", "conf": 0.30, "bbox": [x1,y1,x2,y2]}
  ],
  "image_url": "http://127.0.0.1:8001/static/infer/annotated/....jpg"
}
```

## Exportações
- ONNX: usar Ultralytics `model.export(format="onnx")`.
- TensorRT: `model.export(format="engine")` (requer CUDA/TensorRT).

## Datasets
- Metadados esperados: `data.yaml`, `classes.txt`, `images/`, `labels/`.
- Por padrão, imagens e labels em `app/data/**` são ignoradas no Git (ver `.gitignore`).

## Tuning e Troubleshooting
- Ajustar `conf`, `iou` e `imgsz` no `/infer` para controlar precisão e velocidade.
- Se CUDA indisponível, o `/health` reportará `cuda_available=false`. Rodar em CPU ou configurar drivers CUDA.
- Out of Memory (GPU): reduzir `imgsz`, usar `device='cpu'` ou mixed precision.

## Versionamento de Modelos
- Pesos `.pt/.onnx/.engine` são rastreados via Git LFS. Suba apenas checkpoints necessários.

## Licença
MIT (veja LICENSE).