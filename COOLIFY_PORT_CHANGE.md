# Como Alterar a Porta do App no Coolify

## Problema
A porta 8000 está ocupada e você precisa usar uma porta diferente durante o deploy no Coolify.

## Solução

### 1. Configuração via Variável de Ambiente no Coolify

No painel do Coolify, vá para as **Environment Variables** do seu projeto e adicione:

```
PORT=8080
```

(ou qualquer outra porta disponível que você deseje usar)

### 2. Alterações Realizadas no Código

As seguintes alterações foram feitas para suportar porta configurável:

#### Dockerfile
- Modificado para usar `${PORT:-8000}` (padrão 8000 se não especificado)
- CMD agora usa shell para expandir a variável de ambiente

#### docker-compose.coolify.yml
- Adicionada variável `PORT: ${PORT:-8000}` no environment
- Healthcheck atualizado para usar a porta configurável

### 3. Como Usar

1. **No Coolify UI:**
   - Vá para seu projeto
   - Acesse "Environment Variables"
   - Adicione: `PORT=8080` (ou sua porta desejada)
   - Faça o redeploy

2. **Via Docker Compose local (para testes):**
   ```bash
   PORT=8080 docker-compose -f docker-compose.coolify.yml up
   ```

3. **Via variável de ambiente no terminal:**
   ```bash
   export PORT=8080
   # Então faça o deploy normalmente
   ```

### 4. Verificação

Após o deploy, o app estará rodando na porta especificada. Você pode verificar:

- Logs do container mostrarão: `Uvicorn running on http://0.0.0.0:8080`
- Healthcheck testará a porta correta
- Coolify proxy redirecionará automaticamente

### 5. Portas Recomendadas

- `8080` - Alternativa comum ao 8000
- `3000` - Porta padrão para muitos apps Node.js
- `5000` - Porta comum para apps Flask/Python
- `9000` - Outra alternativa popular

### Notas Importantes

- O Coolify gerencia automaticamente o proxy/roteamento
- A porta interna do container pode ser diferente da porta externa
- Certifique-se de que a porta escolhida não está em uso por outros serviços
- O healthcheck foi atualizado para usar a porta configurável