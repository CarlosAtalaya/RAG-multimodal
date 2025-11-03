import requests
import base64
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import time

class HybridEmbedder:
    """
    Estrategia híbrida:
    1. Qwen3-VL genera descripción textual de la imagen
    2. Sentence-BERT convierte descripción a embedding
    """
    
    def __init__(
        self,
        qwen_api_endpoint: str = "http://localhost:8001",
        sentence_model: str = "all-MiniLM-L6-v2",  # ✅ CORREGIDO
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        self.qwen_endpoint = f"{qwen_api_endpoint}/qwen3/chat/completions"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        print(f"🔧 Cargando Sentence-Transformer: {sentence_model}...")
        self.sentence_encoder = SentenceTransformer(sentence_model)
        self.embedding_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        
        print(f"✅ HybridEmbedder inicializado")
        print(f"   - Qwen API: {self.qwen_endpoint}")
        print(f"   - Sentence Model: {sentence_model}")
        print(f"   - Embedding dim: {self.embedding_dim}")
    
    def generate_embedding(
        self,
        image_path: Path,
        text_context: str = "vehicle damage"
    ) -> np.ndarray:
        """
        Genera embedding híbrido de una imagen con retry logic
        """
        
        # 1. Generar descripción con Qwen3-VL (con reintentos)
        description = self._get_description_with_retry(image_path, text_context)
        
        # 2. Generar embedding de la descripción
        embedding = self.sentence_encoder.encode(
            description,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        return embedding.astype(np.float32)
    
    def _get_description_with_retry(
        self, 
        image_path: Path, 
        text_context: str
    ) -> str:
        """
        Obtiene descripción de Qwen3-VL con lógica de reintentos
        """
        # Cargar y codificar imagen
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        description_prompt = f"""Describe the visual characteristics of this {text_context} image in detail.
Focus on:
- Type and severity of damage visible
- Location and spatial extent
- Visual patterns, textures, and edges
- Color variations and lighting
- Any distinctive features

Be objective and detailed."""
        
        payload = {
            "model": "Qwen3-VL-4B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": description_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 256,
            "temperature": 0.1
        }
        
        # Intentar con reintentos
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.qwen_endpoint, 
                    json=payload, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    description = response.json()['choices'][0]['message']['content']
                    return description
                else:
                    raise RuntimeError(f"API returned status {response.status_code}")
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"⚠️  Intento {attempt + 1} falló, reintentando en {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    # Último intento falló, usar descripción fallback
                    print(f"❌ Todos los intentos fallaron para {image_path.name}")
                    return f"Image showing {text_context} with visual features."
        
        # Fallback por si acaso
        return f"Image showing {text_context}."
    
    def generate_batch_embeddings(
        self,
        image_paths: List[Path],
        text_contexts: List[str] = None
    ) -> np.ndarray:
        """
        Genera embeddings en batch
        """
        if text_contexts is None:
            text_contexts = ["vehicle damage"] * len(image_paths)
        
        embeddings = []
        for path, context in zip(image_paths, text_contexts):
            embedding = self.generate_embedding(path, context)
            embeddings.append(embedding)
        
        return np.vstack(embeddings)