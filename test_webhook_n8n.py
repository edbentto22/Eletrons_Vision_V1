#!/usr/bin/env python3
"""
Script para testar o envio do webhook para o N8N
"""
import json
import time
import httpx
import asyncio
from typing import Dict, Any

def create_test_payload() -> Dict[str, Any]:
    """Cria um payload de teste no formato correto para N8N"""
    return {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'job': 'infer',
        'count': 1,
        'latency_ms': 850,
        'results': [
            {
                'image_id': 'test_webhook_001',
                'source': '/app/data/infer/test_webhook_001-test.jpg',
                'source_url': 'http://127.0.0.1:8012/i/test_webhook_001.jpg',
                'width': 640,
                'height': 480,
                'detections': [
                    {
                        'x1': 50.0,
                        'y1': 60.0,
                        'x2': 150.0,
                        'y2': 180.0,
                        'conf': 0.89,
                        'cls': 0,
                        'label': 'person'
                    }
                ],
                'annotated_url': 'http://127.0.0.1:8012/data/infer/test_webhook_001-annotated.jpg'
            }
        ],
        # Meta dados no root level
        'registro': 999,
        'ponto': 888,
        'sheet_id': 'test_sheet_webhook'
    }

async def test_webhook_n8n():
    """Testa o envio do webhook para o N8N"""
    payload = create_test_payload()
    url = "https://eletrons.catalise.me/webhook/infer"
    
    print("🚀 Testando envio do webhook para N8N...")
    print(f"📡 URL: {url}")
    print("=" * 60)
    
    print("📋 Payload a ser enviado:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            print("📤 Enviando webhook...")
            response = await client.post(url, json=payload)
            
            print(f"📊 Status Code: {response.status_code}")
            print(f"📋 Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print("✅ Webhook enviado com sucesso!")
                try:
                    response_data = response.json()
                    print("📥 Resposta do N8N:")
                    print(json.dumps(response_data, indent=2, ensure_ascii=False))
                except:
                    print("📥 Resposta (texto):")
                    print(response.text)
            else:
                print(f"❌ Erro no webhook: {response.status_code}")
                print("📥 Resposta de erro:")
                print(response.text)
                
    except Exception as e:
        print(f"❌ Erro na requisição: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_webhook_n8n())