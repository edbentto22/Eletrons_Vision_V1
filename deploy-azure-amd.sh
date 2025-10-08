#!/bin/bash

# Script de Deploy para Azure Standard NV4as v4 (AMD)
# Otimizado para execução CPU-only em Windows Server com Docker

set -e

echo "🚀 Deploy para Azure Standard NV4as v4 (AMD)"
echo "============================================="

# Verificar se estamos no diretório correto
if [ ! -f "Dockerfile.azure-amd" ]; then
    echo "❌ Erro: Dockerfile.azure-amd não encontrado!"
    echo "Execute este script no diretório raiz do projeto."
    exit 1
fi

# Configurações da instância
export AZURE_INSTANCE_TYPE="NV4as_v4"
export GPU_AVAILABLE="false"
export CPU_CORES="4"
export MEMORY_GB="14"

# Otimizações AMD
export OMP_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export OPENBLAS_NUM_THREADS="4"
export CUDA_VISIBLE_DEVICES=""

# Configurações YOLO CPU-only
export YOLO_DEVICE="cpu"
export YOLO_BATCH_SIZE="1"
export YOLO_WORKERS="2"
export YOLO_IMGSZ="640"

# Configurações da aplicação
export PORT="${PORT:-8000}"
export APP_ENV="production"

echo "📋 Configurações:"
echo "  - Instância: $AZURE_INSTANCE_TYPE"
echo "  - GPU: $GPU_AVAILABLE"
echo "  - CPU Cores: $CPU_CORES"
echo "  - Memória: ${MEMORY_GB}GB"
echo "  - Porta: $PORT"
echo "  - YOLO Device: $YOLO_DEVICE"
echo ""

# Parar containers existentes
echo "🛑 Parando containers existentes..."
docker-compose -f docker-compose.azure-amd.yml down --remove-orphans || true

# Limpar imagens antigas (opcional)
read -p "🗑️  Limpar imagens antigas? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Limpando imagens antigas..."
    docker image prune -f
    docker rmi eletrons-vision:azure-amd 2>/dev/null || true
fi

# Build da imagem otimizada
echo "🔨 Construindo imagem otimizada para AMD..."
docker build \
    -f Dockerfile.azure-amd \
    -t eletrons-vision:azure-amd \
    --build-arg APP_ENV=production \
    .

if [ $? -ne 0 ]; then
    echo "❌ Erro no build da imagem!"
    exit 1
fi

# Verificar se volumes existem
echo "📁 Verificando volumes..."
docker volume create eletrons-models 2>/dev/null || true
docker volume create eletrons-data 2>/dev/null || true
docker volume create eletrons-runs 2>/dev/null || true

# Iniciar serviços
echo "🚀 Iniciando serviços..."
docker-compose -f docker-compose.azure-amd.yml up -d

if [ $? -ne 0 ]; then
    echo "❌ Erro ao iniciar serviços!"
    exit 1
fi

# Aguardar inicialização
echo "⏳ Aguardando inicialização..."
sleep 10

# Verificar status
echo "🔍 Verificando status dos serviços..."
docker-compose -f docker-compose.azure-amd.yml ps

# Testar healthcheck
echo "🏥 Testando healthcheck..."
for i in {1..6}; do
    if curl -s -f "http://localhost:$PORT/health" > /dev/null; then
        echo "✅ Serviço está funcionando!"
        break
    else
        echo "⏳ Tentativa $i/6 - aguardando..."
        sleep 10
    fi
    
    if [ $i -eq 6 ]; then
        echo "❌ Serviço não respondeu ao healthcheck!"
        echo "📋 Logs do container:"
        docker-compose -f docker-compose.azure-amd.yml logs --tail=20
        exit 1
    fi
done

# Mostrar informações finais
echo ""
echo "✅ Deploy concluído com sucesso!"
echo "============================================="
echo "🌐 URL: http://localhost:$PORT"
echo "🏥 Health: http://localhost:$PORT/health"
echo "📊 Panel: http://localhost:$PORT/panel"
echo ""
echo "📋 Comandos úteis:"
echo "  - Logs: docker-compose -f docker-compose.azure-amd.yml logs -f"
echo "  - Status: docker-compose -f docker-compose.azure-amd.yml ps"
echo "  - Parar: docker-compose -f docker-compose.azure-amd.yml down"
echo "  - Stats: docker stats"
echo ""
echo "⚠️  Lembre-se:"
echo "  - Esta instância roda apenas CPU (sem GPU)"
echo "  - Performance limitada: ~2-5 FPS com YOLOv8n"
echo "  - Considere migrar para NCasT4_v3 para melhor performance"
echo ""

# Mostrar uso de recursos
echo "📊 Uso atual de recursos:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"