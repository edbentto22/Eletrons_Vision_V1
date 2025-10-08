# Configuração para Azure Standard NV4as v4 (AMD)

## Visão Geral da Instância

A instância **Azure Standard NV4as v4** possui as seguintes especificações <mcreference link="https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvv4-series" index="1">1</mcreference>:

- **CPU**: AMD EPYC 7V12 (Rome) - 4 vCPUs
- **Memória**: 14 GB RAM
- **GPU**: AMD Radeon Instinct MI25 (1/8 da GPU = 2 GB VRAM)
- **Sistema Operacional**: Suporta apenas Windows (limitação da série NVv4)

## ⚠️ Limitações Importantes

### 1. **Suporte Apenas Windows**
<mcreference link="https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvv4-series" index="1">1</mcreference> A série NVv4 suporta **apenas Windows** como sistema operacional guest. Para aplicações Linux/Docker, você precisará:
- Usar Windows Server com Docker Desktop
- Ou migrar para uma série diferente (NCasT4_v3, NVadsA10_v5)

### 2. **GPU AMD Limitada**
- Apenas **1/8 da GPU** (2 GB VRAM) <mcreference link="https://learn.microsoft.com/en-us/answers/questions/215930/radeon-instinct-mi25-mxgpu-azure-server-2016-with" index="5">5</mcreference>
- **Não há suporte nativo para CUDA** (GPU AMD)
- **Recomendação**: Executar YOLO em **modo CPU-only**

### 3. **Recursos Limitados**
- **4 vCPUs** e **14 GB RAM** - configuração modesta
- Necessário otimizar para execução eficiente

## 🚨 Série em Descontinuação

**IMPORTANTE**: <mcreference link="https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvv4-series" index="1">1</mcreference> A série NVv4 será **descontinuada em 30 de setembro de 2026**. Considere migrar para:

- **NCasT4_v3**: NVIDIA Tesla T4 (suporta Linux + CUDA)
- **NVadsA10_v5**: NVIDIA A10 (suporta Linux + CUDA)

## 📋 Configurações Implementadas

### Dockerfile Otimizado (`Dockerfile.azure-amd`)

```dockerfile
# Otimizações específicas para AMD EPYC
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    # Forçar execução CPU-only
    CUDA_VISIBLE_DEVICES="" \
    # Configurações de memória para 14GB RAM
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# PyTorch CPU-only otimizado
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu
```

### Docker Compose (`docker-compose.azure-amd.yml`)

```yaml
deploy:
  resources:
    limits:
      cpus: '3.5'      # Deixar 0.5 CPU para sistema
      memory: 12G      # Deixar 2GB para sistema
    reservations:
      cpus: '2.0'
      memory: 8G

environment:
  # Configurações YOLO otimizadas para CPU
  YOLO_BATCH_SIZE: "1"
  YOLO_WORKERS: "2"
  YOLO_IMGSZ: "640"
  YOLO_DEVICE: "cpu"
```

## 🚀 Como Usar

### 1. **Para Windows Server com Docker**

```bash
# Build da imagem otimizada
docker build -f Dockerfile.azure-amd -t eletrons-vision:azure-amd .

# Executar com configurações otimizadas
docker-compose -f docker-compose.azure-amd.yml up -d
```

### 2. **Variáveis de Ambiente Importantes**

```bash
# Configurações obrigatórias
export AZURE_INSTANCE_TYPE=NV4as_v4
export GPU_AVAILABLE=false
export CPU_CORES=4
export MEMORY_GB=14

# Otimizações AMD
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""

# YOLO CPU-only
export YOLO_DEVICE=cpu
export YOLO_BATCH_SIZE=1
export YOLO_WORKERS=2
```

## ⚡ Otimizações de Performance

### 1. **Configurações YOLO**
- **Batch Size**: 1 (evitar sobrecarga de memória)
- **Workers**: 2 (metade dos cores disponíveis)
- **Image Size**: 640 (padrão, balanceado)
- **Device**: CPU (forçado)

### 2. **Configurações de Sistema**
- **Threads**: Limitado a 4 (número de vCPUs)
- **Memória**: Máximo 12GB (reservar 2GB para sistema)
- **CPU**: Máximo 3.5 cores (reservar 0.5 para sistema)

### 3. **Healthcheck Ajustado**
- **Interval**: 45s (mais espaçado)
- **Timeout**: 10s (mais tolerante)
- **Retries**: 3 (reduzido)

## 🔧 Troubleshooting

### Problema: "Out of Memory"
```bash
# Reduzir batch size
export YOLO_BATCH_SIZE=1

# Reduzir workers
export YOLO_WORKERS=1

# Reduzir tamanho da imagem
export YOLO_IMGSZ=416
```

### Problema: Performance Lenta
```bash
# Verificar se está usando CPU
export CUDA_VISIBLE_DEVICES=""

# Otimizar threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Usar modelo menor
export MODEL_VARIANT=yolov8n.pt
```

### Problema: Driver AMD
<mcreference link="https://learn.microsoft.com/en-us/azure/virtual-machines/windows/n-series-amd-driver-setup" index="2">2</mcreference> Para usar a GPU AMD (se necessário):

```bash
# Instalar driver AMD via extensão Azure
az vm extension set \
  --resource-group myResourceGroup \
  --vm-name myVM \
  --name AmdGpuDriverWindows \
  --publisher Microsoft.HpcCompute \
  --version 1.0
```

## 📊 Performance Esperada

### Inferência YOLO (CPU-only)
- **YOLOv8n**: ~2-5 FPS
- **YOLOv8s**: ~1-3 FPS
- **YOLOv8m**: ~0.5-1 FPS

### Recomendações
1. **Use YOLOv8n** para melhor performance
2. **Processe em batch** quando possível
3. **Considere migrar** para NCasT4_v3 para GPU NVIDIA

## 🔄 Migração Recomendada

### Para NCasT4_v3 (NVIDIA Tesla T4)
```bash
# Instância recomendada
Standard_NC4as_T4_v3:
- 4 vCPUs
- 28 GB RAM  
- 1x NVIDIA Tesla T4 (16GB)
- Suporta Linux + CUDA
```

### Benefícios da Migração
- **Suporte Linux nativo**
- **GPU NVIDIA com CUDA**
- **Mais RAM** (28GB vs 14GB)
- **Performance 10-50x melhor** para YOLO

## 📝 Comandos Úteis

```bash
# Verificar recursos da instância
docker stats

# Monitorar CPU/Memória
htop

# Testar inferência
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test_image.jpg" \
  -F "device=cpu"

# Logs detalhados
docker-compose -f docker-compose.azure-amd.yml logs -f
```

## ✅ Checklist de Deploy

- [ ] Confirmar Windows Server como OS
- [ ] Instalar Docker Desktop para Windows
- [ ] Configurar variáveis de ambiente
- [ ] Build da imagem otimizada
- [ ] Testar com imagem pequena
- [ ] Monitorar uso de recursos
- [ ] Planejar migração para NCasT4_v3