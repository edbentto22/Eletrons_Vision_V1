# Eletrons Vision Service — PRD (Product Requirements Document)

## 1) Sumário Executivo
Serviço web de Visão Computacional para detecção/segmentação/rastreamento com YOLO (Ultralytics), oferecendo APIs REST, painel web e automações via webhooks. Deve rodar localmente e em cloud (Docker/Coolify), com foco em desempenho, segurança e operação simples.

## 2) Objetivos
- Inferência em imagens, vídeos e streams com modelos YOLO (export ONNX/TRT).
- Painel para login, configuração e acompanhamento de jobs/estatísticas.
- Automação via webhooks (N8N/integrações REST) e logs/telemetria.

## 3) Escopo
- IN: Autenticação (email/senha ou hash bcrypt), endpoints REST, paginações, métricas (/metrics), health (/health), export de modelos, tracking básico.
- OUT: Treinamento distribuído, rotulador UI completo, billing, multi-tenant.

## 4) Principais Casos de Uso
- Operador valida produção com detecção em tempo real.
- Engenheiro configura modelo/limiares e integra com N8N.
- Analista audita eventos, baixa logs e acompanha métricas.

## 5) Arquitetura (alto nível)
- Backend: FastAPI/Uvicorn (<app/main.py>), estados em <app/state.py> e configs em <app/config.py>.
- Modelos/serviço YOLO: <app/yolo_service.py>, pesos em <app/models/>.
- Web (templates/assets): <app/web/>.
- Dockerfile, docker-compose.coolify.yml para deploy; requirements.txt com deps.

## 6) Principais Endpoints (REST/UI)
- Saúde/Observabilidade: GET /health, GET /metrics
- Auth/UI: GET /login, POST /login, GET /panel, POST /logout
- Config: GET/POST /config
- Inferência: POST /infer (imagem/base64/url), GET /infer/{id}
- Treino/Jobs: POST /train, GET /train/{job_id}, POST /train/{job_id}/pause|resume|stop
- Webhooks: POST /webhook/n8n, GET /webhook/logs

## 7) Requisitos Funcionais
- Login com ADMIN_EMAIL + ADMIN_PASSWORD, ou ADMIN_PASSWORD_HASH (bcrypt).
- Sessões com cookie seguro; CSRF desativado apenas em dev.
- CORS configurável; whitelist de IPs opcional.
- Upload de imagem e retorno JSON (bboxes, scores, classes, masks quando aplicável) e artefato anotado opcional.
- Export de modelo (ONNX/TRT) e seleção de modelo ativo.

## 8) Requisitos Não Funcionais
- Desempenho: P95 inferência < 150 ms (GPU) em 640px; logging de latência/FPS.
- Escalabilidade: Stateless, horizontal com gateway; cache de modelos.
- Segurança: Segredos em runtime; hash bcrypt em produção; HTTPS no edge.
- Observabilidade: logs estruturados, métricas Prometheus, healthcheck.
- Compatibilidade: Python 3.10+, CUDA 12+ (quando GPU), CPU fallback.

## 9) Variáveis de Ambiente (chaves)
- APP_ENV (development|production)
- ADMIN_EMAIL, ADMIN_PASSWORD (opcional), ADMIN_PASSWORD_HASH (recomendado prod)
- SESSION_SECRET, AUTH_TOKEN (opcional para API), CORS_ORIGINS, IP_WHITELIST
- ACTIVE_MODEL, N8N_WEBHOOK_URL, PUBLIC_BASE_URL

## 10) Fluxos Principais
- Login: POST /login valida senha ou bcrypt; cria sessão; redireciona /panel.
- Inferência: POST /infer → valida auth → processa com YOLO → retorna resultados JSON e opcionalmente imagem anotada/artefatos.
- Treino: POST /train inicia job assíncrono; endpoints de controle e status.

## 11) Critérios de Aceite
- /health retorna 200 OK e status do modelo ativo.
- Login funcional em prod com hash bcrypt válido e cookies de sessão.
- /infer responde com boxes/scores/classes corretos e mAP >= meta do modelo.
- Logs e /metrics expostos; variáveis sensíveis não presentes como build-args.

## 12) Métricas de Sucesso
- Disponibilidade > 99.5%, P95 latência inferência, taxa de erro HTTP, mAP/Recall, tempo médio de deploy, tempo para primeira predição.

## 13) Riscos e Mitigações
- Hash corrompido por interpolação ($) no deploy → usar Runtime only e escapar $$.
- Falhas pip/torch em imagens slim → pré-instalar torch CPU ou usar base pytorch.
- OOM na GPU → mixed precision, batch=1, limitar resolução, watchdog.

## 14) Roadmap (MVP → v1.1)
- MVP: Login, /panel, /infer, /health, /metrics, export ONNX, webhook básico.
- v1.0: TRT, tracking, cache de pesos, versão de modelos, audit log.
- v1.1: Multi-câmera RTSP, fila (Redis), escalonamento horizontal.

## 15) Deploy & Operação
- Docker/Coolify: manter segredos como Runtime only; APP_ENV=production em runtime; porta 8000; healthcheck /health.
- Build estável: instalar torch CPU explicitamente no Dockerfile; pip/setuptools/ wheel atualizados; libffi-dev.
- Validação pós-deploy: /health 200, login OK, /infer com amostra.

## 16) Execução Local
- Requisitos: Python 3.10+, opcional GPU (CUDA 12+). Instalar deps via `pip install -r requirements.txt` (ou conforme Dockerfile). Rodar `uvicorn app.main:app --reload`.

## 17) Segurança Operacional
- Forçar ADMIN_PASSWORD_HASH em produção; rotação de SESSION_SECRET; limitar tentativas de login; logs sem segredos; CORS e IP whitelist conforme contexto.

## 18) Personas

- Administrador de Operações (Shopfloor)
  - Objetivos: Garantir produção contínua, acompanhar alertas e taxas de detecção, validar imagens problemáticas.
  - Dores: Interface confusa, latência alta, falsos positivos/negativos.
  - Tarefas-chave: Monitorar painel, executar inferência em lote, exportar logs.
  - KPIs: Disponibilidade, P95 latência, taxa de alerta, FP/FN.

- Engenheiro de Visão Computacional
  - Objetivos: Tunar limiares (conf/iou/imgsz/device), trocar modelos, validar métricas.
  - Dores: Deploy instável, incompatibilidades CUDA/Torch, tempo de iteração longo.
  - Tarefas-chave: Atualizar config, promover modelo, rodar validação/treino.
  - KPIs: mAP, Recall/Precision, tempo de inferência, sucesso de deploy.

- Analista de Qualidade
  - Objetivos: Auditar eventos, revisar amostras, consolidar indicadores.
  - Dores: Falta de rastreabilidade, dados não estruturados.
  - Tarefas-chave: Baixar logs, revisar imagens anotadas, gerar relatórios.
  - KPIs: Tendência de qualidade, taxa de retrabalho, desvio vs meta.

- Integrador/DevOps
  - Objetivos: Automatizar com N8N/REST, manter ambiente seguro.
  - Dores: Variáveis sensíveis no build, CSRF/ CORS, observabilidade.
  - Tarefas-chave: Configurar env, webhooks, monitorar /health e /metrics.
  - KPIs: MTTR, taxas de erro 5xx/4xx, tempo médio de deploy.

## 19) Fluxos de UI (alto nível)

- Login
  1. Usuário acessa /login → backend emite cookie ev_csrf e renderiza formulário.
  2. Envia email/senha + csrf_token → cria sessão e redireciona /panel.

- Painel (Dashboard)
  1. Exibe cartões: Saúde do Sistema (Torch/Ultralytics), Modelo Ativo, Estatísticas de Inferência, Jobs Recentes.
  2. Links rápidos para Inferência, Treino, Config, Webhooks.

- Inferência (UI)
  1. Upload múltiplo ou URLs → define conf/iou/imgsz/device.
  2. Executa, exibe tabela de resultados, miniaturas anotadas, download de artefatos.
  3. Se habilitado, envia resultados ao N8N.

- Treinamento
  1. Inicia job com parâmetros (epochs, imgsz, lr0, batch, device, pretrained, resume, variant).
  2. Acompanha status, pausa/retoma/stop, baixa melhor .pt e promove modelo.

- Configurações
  1. Edita UIConfig (webhooks in/out, defaults de inferência, device).
  2. Salva com CSRF; auditável via logs.

## 20) Wireframes (texto)

- Layout Base
  - Header: título, status (ok/degraded), usuário (menu conta/logout).
  - Sidebar: Dashboard, Inferência, Treino, Configurações, Webhooks, Logs.
  - Conteúdo: cards/tabelas/gráficos.

- Dashboard
  - [Card] Saúde do Sistema: Torch/Ultralytics, CUDA disponível, uptime.
  - [Card] Modelo Ativo: nome/versão, caminho, botões (promover/baixar ONNX/TRT).
  - [Card] Estatísticas: inferences total, P95/P50 latência, jobs ativos.
  - [Tabela] Jobs Recentes: job_id, status, métricas, ações.

- Página Inferência
  - [Upload Dropzone] múltiplas imagens (até 30) + campo URLs.
  - [Parâmetros] conf, iou, imgsz, device; [checkbox] enviar webhook.
  - [Resultados] tabela com: imagem, count, tempo (ms), botões (ver anotada/JSON).

- Página Treino
  - [Form] epochs, imgsz, lr0, batch, device, pretrained, resume, variant.
  - [Lista] jobs: status (running/paused/completed), métricas, ações (pause/resume/stop).

- Página Config
  - [Webhooks In] habilitar/token/path.
  - [Webhooks Out] URL N8N, habilitar, incluir imagem.
  - [Defaults] CONF/IOU/IMGSZ/DEVICE.

## 21) Exemplos de Payloads e cURL

Notas importantes
- CSRF: Em produção, endpoints baseados em sessão exigem CSRF (cookie ev_csrf + campo csrf_token ou header x-csrf-token). Em desenvolvimento, CSRF é bypassado.
- Auth: Você pode autenticar via sessão (login) ou via Bearer token (AUTH_TOKEN). Se usar Bearer, inclua `Authorization: Bearer <token>`.

1) Login (sessão)
- Obter cookie ev_csrf e enviar formulário com token.

```bash
# 1) Obtenha cookie ev_csrf via GET /login
curl -c cookies.txt https://HOST/login -s -o /dev/null
CSRF=$(awk '/\tev_csrf\t/ {print $7}' cookies.txt)

# 2) Envie email/senha + csrf_token
curl -b cookies.txt -c cookies.txt -X POST \
  -F email="admin@example.com" -F password="admin" -F csrf_token="$CSRF" \
  https://HOST/login -i
```

2) Health e Métricas
```bash
curl https://HOST/health
curl https://HOST/metrics
```

3) Inferência (upload de imagens)
```bash
# Sessão + CSRF
echo "$CSRF"  # reusar valor ou refazer GET /login
curl -b cookies.txt -c cookies.txt -X POST \
  -F files=@img1.jpg -F files=@img2.png \
  -F conf=0.25 -F iou=0.45 -F imgsz=640 -F device="cpu" \
  -F send_webhook=true -F csrf_token="$CSRF" \
  https://HOST/infer

# Alternativa via Bearer (sem CSRF)
curl -H "Authorization: Bearer $AUTH_TOKEN" -X POST \
  -F files=@img1.jpg -F files=@img2.png \
  https://HOST/infer
```
Resposta (exemplo simplificado, alinhado com <mcfile name="schemas.py" path="/Users/edbentto/Eletrons_Vision_V1/app/schemas.py"></mcfile>):
```json
{
  "count": 2,
  "results": [
    {
      "image_id": "a1b2c3",
      "source": "/app/data/infer/a1b2c3-img1.jpg",
      "source_url": "https://HOST/i/a1b2c3.jpg",
      "width": 1280,
      "height": 720,
      "detections": [
        {"x1": 100.5, "y1": 80.2, "x2": 300.9, "y2": 260.4, "conf": 0.91, "cls": 0, "label": "person"}
      ],
      "annotated_url": "https://HOST/data/infer/a1b2c3-img1-annotated.jpg"
    }
  ],
  "params": {"conf": 0.25, "iou": 0.45, "imgsz": 640, "device": "cpu"}
}
```

4) Inferência (via URLs)
```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" -X POST \
  -F urls="https://example.com/image1.jpg" -F urls="https://example.com/image2.jpg" \
  https://HOST/infer
```

5) Config (GET/POST)
```bash
# GET atual
curl -H "Authorization: Bearer $AUTH_TOKEN" https://HOST/config

# PATCH (parcial) via POST
curl -H "Authorization: Bearer $AUTH_TOKEN" -H "Content-Type: application/json" \
  -d '{
        "WEBHOOK_INFER_ENABLED": true,
        "N8N_WEBHOOK_URL": "https://n8n.example.com/webhook/abc",
        "CONF_DEFAULT": 0.3,
        "IMGSZ_DEFAULT": 640,
        "DEVICE_DEFAULT": "cpu"
      }' \
  https://HOST/config
```
Resposta (exemplo, conforme UIConfigResponse):
```json
{
  "WEBHOOK_INFER_ENABLED": true,
  "WEBHOOK_INFER_TOKEN": null,
  "WEBHOOK_INFER_PATH": "infer",
  "N8N_WEBHOOK_URL": "https://n8n.example.com/webhook/abc",
  "N8N_WEBHOOK_ENABLED": true,
  "N8N_WEBHOOK_INCLUDE_IMAGE": false,
  "CONF_DEFAULT": 0.3,
  "IOU_DEFAULT": 0.45,
  "IMGSZ_DEFAULT": 640,
  "DEVICE_DEFAULT": "cpu"
}
```

6) Treino
```bash
# Iniciar
curl -H "Authorization: Bearer $AUTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"epochs": 50, "imgsz": 640, "lr0": 0.01, "batch": 16, "device": "cpu", "pretrained": true, "resume": false}' \
  https://HOST/train

# Status de um job
curl -H "Authorization: Bearer $AUTH_TOKEN" https://HOST/train/JOB_ID

# Controle
curl -H "Authorization: Bearer $AUTH_TOKEN" -X POST https://HOST/train/JOB_ID/pause
curl -H "Authorization: Bearer $AUTH_TOKEN" -X POST https://HOST/train/JOB_ID/resume
curl -H "Authorization: Bearer $AUTH_TOKEN" -X POST https://HOST/train/JOB_ID/stop
```
Resposta (exemplo TrainStatus):
```json
{
  "job_id": "20240901-123456",
  "status": "running",
  "best_pt": null,
  "metrics": {"mAP50": 0.71},
  "started_at": "2024-09-01T12:34:56Z",
  "finished_at": null
}
```

7) Webhook (saída para N8N, exemplo de payload)
```json
{
  "timestamp": "2024-09-01T12:35:10Z",
  "job": "infer",
  "count": 2,
  "results": [
    {
      "image_id": "a1b2c3",
      "source_url": "https://HOST/i/a1b2c3.jpg",
      "detections": [
        {"x1":100.5, "y1":80.2, "x2":300.9, "y2":260.4, "conf":0.91, "cls":0, "label":"person"}
      ]
    }
  ],
  "meta": {"registro": 123, "ponto": 45, "sheet_id": "abc"}
}
```

---
Essas adições detalham o uso por perfil, a navegação da UI e fornecem exemplos práticos de integração com as rotas reais do backend (<mcfile name="main.py" path="/Users/edbentto/Eletrons_Vision_V1/app/main.py"></mcfile> e <mcfile name="schemas.py" path="/Users/edbentto/Eletrons_Vision_V1/app/schemas.py"></mcfile>).