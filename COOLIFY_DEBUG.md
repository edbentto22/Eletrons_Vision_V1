# Guia de Debug para Deploy no Coolify

## ⚠️ CONFIGURAÇÃO DE PORTA IMPORTANTE

**A aplicação roda na porta 8012 para evitar conflito com a porta 8000 do Coolify.**

No painel do Coolify, certifique-se de configurar:
- **Port**: 8012 (não 8000)
- **Protocol**: HTTP

## Problemas Identificados e Correções

### 1. Conflito de Porta
**Problema**: Coolify usa porta 8000 internamente, causando conflito.

**Correção**: Aplicação configurada para usar porta 8012.

### 2. Healthcheck Atualizado
**Problema**: Healthcheck estava verificando porta incorreta.

**Correção**: Healthcheck agora verifica `http://127.0.0.1:8012/health`.

### 3. Endpoint de Health Melhorado
**Problema**: O endpoint `/health` não tinha tratamento de erros adequado.

**Correção**: Adicionado try/catch e logging de erros no endpoint.

## Variáveis de Ambiente Necessárias no Coolify

### Obrigatórias
```bash
# Autenticação
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD_HASH=<hash_bcrypt>
SESSION_SECRET=<secret_key_forte>

# Aplicação
APP_NAME="Eletrons Vision Service"
APP_ENV=production
```

### Configuração de Porta (IMPORTANTE)
```bash
# A porta é fixa em 8012 - NÃO altere
PORT=8012
```

### Opcionais (com defaults)
```bash
# Modelo
MODEL_VARIANT=yolov8n.pt
SAVE_ANNOTATIONS=true

# CORS e Segurança
CORS_ORIGINS=*
IP_WHITELIST=

# Webhook
N8N_WEBHOOK_URL=
AUTH_TOKEN=
```

## Comandos de Debug

### 1. Verificar Logs do Container
```bash
# No Coolify, vá para a aba "Logs" do serviço
# Ou via Docker:
docker logs <container_id> --tail 100 -f
```

### 2. Testar Healthcheck Manualmente
```bash
# Dentro do container:
python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8012/health', timeout=5).status==200 else 1)"

# Ou via curl (de fora do container):
curl -f http://localhost:8012/health
```

### 3. Verificar Variáveis de Ambiente
```bash
# Dentro do container:
env | grep -E "(APP_|ADMIN_|SESSION_|PORT)"
```

### 4. Testar Importações Python
```bash
# Dentro do container:
python -c "import torch, ultralytics; print('OK')"
```

## Checklist de Deploy no Coolify

- [ ] **Porta configurada como 8012 no painel do Coolify**
- [ ] Variáveis de ambiente configuradas no Coolify
- [ ] ADMIN_PASSWORD_HASH gerado corretamente
- [ ] SESSION_SECRET definido (não usar default)
- [ ] Healthcheck respondendo em `/health`
- [ ] Logs não mostram erros de importação
- [ ] Endpoint `/` acessível na porta 8012

## Troubleshooting Comum

### Erro: "Connection refused" ou "Port already in use"
- **Causa**: Conflito de porta com Coolify
- **Solução**: Verificar se a porta está configurada como 8012

### Erro: "Health check failed"
- Verificar se a aplicação está rodando na porta 8012
- Testar endpoint `/health` manualmente: `curl http://localhost:8012/health`
- Verificar logs de inicialização

### Erro: "ModuleNotFoundError"
- Verificar se requirements.txt está correto
- Rebuild da imagem Docker

### Erro: "Authentication failed"
- Verificar ADMIN_PASSWORD_HASH
- Gerar novo hash se necessário:
```python
import bcrypt
password = "sua_senha"
hash_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
print(hash_bytes.decode('utf-8'))
```

### Erro: "CORS policy"
- Configurar CORS_ORIGINS adequadamente
- Para desenvolvimento: `CORS_ORIGINS=*`
- Para produção: `CORS_ORIGINS=https://seu-dominio.com`

## Configuração no Painel do Coolify

1. **Service Settings**:
   - Port: `8012`
   - Protocol: `HTTP`

2. **Environment Variables**:
   - Adicionar todas as variáveis listadas acima
   - **Importante**: PORT deve ser 8012

3. **Health Check**:
   - Path: `/health`
   - Port: `8012`

## Logs Importantes para Monitorar

1. **Inicialização da aplicação**
2. **Carregamento do modelo YOLO**
3. **Configuração de variáveis de ambiente**
4. **Resposta do healthcheck na porta 8012**
5. **Erros de autenticação**

## Próximos Passos

1. Fazer commit das correções
2. Rebuild no Coolify
3. **Verificar configuração de porta no painel (8012)**
4. Monitorar logs durante o deploy
5. Testar endpoints principais
6. Verificar funcionalidade completa