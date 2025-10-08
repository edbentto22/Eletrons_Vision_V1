#!/usr/bin/env python3
"""
Script para testar o payload do webhook N8N localmente
"""
import json
import time
from typing import Dict, Any

def create_test_payload() -> Dict[str, Any]:
    """Cria um payload de teste no formato correto para N8N"""
    return {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'job': 'infer',
        'count': 2,
        'latency_ms': 1250,
        'results': [
            {
                'image_id': 'test123',
                'source': '/app/data/infer/test123-image1.jpg',
                'source_url': 'http://127.0.0.1:8012/i/test123.jpg',
                'width': 1280,
                'height': 720,
                'detections': [
                    {
                        'x1': 100.5,
                        'y1': 80.2,
                        'x2': 300.9,
                        'y2': 260.4,
                        'conf': 0.91,
                        'cls': 0,
                        'label': 'person'
                    },
                    {
                        'x1': 450.1,
                        'y1': 120.3,
                        'x2': 650.7,
                        'y2': 380.8,
                        'conf': 0.85,
                        'cls': 1,
                        'label': 'car'
                    }
                ],
                'annotated_url': 'http://127.0.0.1:8012/data/infer/test123-annotated.jpg'
            },
            {
                'image_id': 'test456',
                'source': '/app/data/infer/test456-image2.jpg',
                'source_url': 'http://127.0.0.1:8012/i/test456.jpg',
                'width': 1920,
                'height': 1080,
                'detections': [
                    {
                        'x1': 200.0,
                        'y1': 150.0,
                        'x2': 400.0,
                        'y2': 350.0,
                        'conf': 0.78,
                        'cls': 0,
                        'label': 'person'
                    }
                ],
                'annotated_url': 'http://127.0.0.1:8012/data/infer/test456-annotated.jpg'
            }
        ],
        # Meta dados no root level (como esperado pelo N8N)
        'registro': 123,
        'ponto': 456,
        'sheet_id': 'planilha_001'
    }

def test_payload_structure():
    """Testa se a estrutura do payload está correta"""
    payload = create_test_payload()
    
    print("🔍 Testando estrutura do payload do webhook N8N...")
    print("=" * 60)
    
    # Verificar campos obrigatórios
    required_fields = ['timestamp', 'job', 'count', 'results']
    for field in required_fields:
        if field in payload:
            print(f"✅ Campo '{field}': {payload[field]}")
        else:
            print(f"❌ Campo obrigatório '{field}' ausente!")
    
    print("\n📋 Payload completo:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    print(f"\n📊 Estatísticas:")
    print(f"   - Total de imagens: {payload['count']}")
    print(f"   - Total de detecções: {sum(len(r['detections']) for r in payload['results'])}")
    print(f"   - Latência: {payload['latency_ms']}ms")
    
    return payload

if __name__ == "__main__":
    test_payload_structure()