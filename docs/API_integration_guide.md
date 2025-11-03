# 🔌 INTEGRACIÓN CON TU API QWEN3-VL EXISTENTE

## 📋 CONTEXTO

Tu API actual en `http://localhost:8001` está optimizada para **generación de texto** (chat completions), pero para el RAG multimodal necesitamos **embeddings vectoriales**.

---

## ⚠️ PROBLEMA: Tu API No Expone Embeddings

### Estado Actual
```python
# Tu endpoint: POST /qwen3/chat/completions
# Retorna: texto generado
# NO retorna: vector embedding del modelo
```

### Necesidad RAG
```python
# Necesitamos: vector numérico [768 dimensiones]
# Para: búsqueda de similitud en FAISS
```

---

## 🎯 SOLUCIONES PROPUESTAS (3 Opciones)

### ✅ **OPCIÓN A: Modificar API para Exponer Embeddings** (RECOMENDADO)

Añadir un nuevo endpoint `/qwen3/embeddings` a tu `api.py`:

```python
# Añadir a api.py

class EmbeddingRequest(BaseModel):
    model: str
    input: str  # Puede ser descripción textual de la imagen
    image_url: Optional[str] = None  # Base64 opcional


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


@app.post("/qwen3/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings from Qwen3-VL's vision encoder.
    
    This extracts the last hidden state from the model, which can be used
    as a dense vector representation for similarity search.
    """
    try:
        if model is None or processor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Process image if provided
        image_path = None
        if request.image_url:
            image = decode_base64_image(request.image_url)
            temp_image = "/tmp/temp_embedding_image.jpg"
            image.save(temp_image, format='JPEG')
            image_path = temp_image
        
        # Prepare messages for processing
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path} if image_path else None,
                    {"type": "text", "text": request.input}
                ]
            }
        ]
        
        # Remove None values
        messages[0]["content"] = [c for c in messages[0]["content"] if c is not None]
        
        # Process inputs
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # No generation for embeddings
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True
        )
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Extract embeddings from model
        with torch.inference_mode():
            # Get model outputs without generation
            outputs = model.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
                image_grid_thw=inputs.image_grid_thw if hasattr(inputs, 'image_grid_thw') else None
            )
            
            # Extract last hidden state and pool
            # Option 1: Mean pooling over sequence
            last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            attention_mask = inputs.attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            
            # Mean pool considering attention mask
            masked_hidden = last_hidden_state * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden_dim]
            sum_mask = attention_mask.sum(dim=1)  # [batch, 1]
            pooled_embedding = sum_hidden / sum_mask  # [batch, hidden_dim]
            
            # Convert to list
            embedding = pooled_embedding[0].cpu().numpy().tolist()
        
        # Build response
        response = EmbeddingResponse(
            object="list",
            data=[
                EmbeddingData(
                    object="embedding",
                    embedding=embedding,
                    index=0
                )
            ],
            model=request.model,
            usage=Usage(
                prompt_tokens=inputs.input_ids.shape[1],
                completion_tokens=0,
                total_tokens=inputs.input_ids.shape[1]
            )
        )
        
        print(f"✅ Embedding generated | Dimension: {len(embedding)}")
        
        return response
        
    except Exception as e:
        print(f"❌ Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Cliente Python Actualizado

```python
# src/core/embeddings/qwen_embedder.py

import requests
import base64
from pathlib import Path
from typing import List, Union
import numpy as np
from PIL import Image

class Qwen3VLEmbedder:
    """
    Cliente para generar embeddings usando tu API Qwen3-VL modificada
    """
    
    def __init__(
        self,
        api_endpoint: str = "http://localhost:8001",
        model_name: str = "Qwen3-VL-4B-Instruct"
    ):
        self.api_endpoint = api_endpoint
        self.embedding_endpoint = f"{api_endpoint}/qwen3/embeddings"
        self.model_name = model_name
        self.embedding_dim = None  # Se detectará en primera llamada
    
    def generate_embedding(
        self,
        image_path: Union[str, Path],
        text_context: str = "Analyze this vehicle damage image"
    ) -> np.ndarray:
        """
        Genera embedding de una imagen
        
        Args:
            image_path: Ruta a la imagen
            text_context: Contexto textual para guiar el embedding
        
        Returns:
            Vector numpy de dimensiones variables (típicamente 768-1536)
        """
        # Cargar y codificar imagen
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Construir payload
        payload = {
            "model": self.model_name,
            "input": text_context,
            "image_url": f"data:image/jpeg;base64,{image_b64}"
        }
        
        # Llamada a API
        response = requests.post(
            self.embedding_endpoint,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
        
        result = response.json()
        embedding = np.array(result['data'][0]['embedding'], dtype=np.float32)
        
        # Detectar dimensión en primera llamada
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
            print(f"✅ Embedding dimension detected: {self.embedding_dim}")
        
        return embedding
    
    def generate_batch_embeddings(
        self,
        image_paths: List[Path],
        text_contexts: List[str] = None,
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Genera embeddings en batch
        
        NOTA: Tu API actual procesa 1 imagen a la vez.
        Este método hace llamadas secuenciales.
        Para verdadero batch processing, modificar API.
        """
        if text_contexts is None:
            text_contexts = ["Analyze this vehicle damage"] * len(image_paths)
        
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_contexts = text_contexts[i:i+batch_size]
            
            for path, context in zip(batch_paths, batch_contexts):
                embedding = self.generate_embedding(path, context)
                embeddings.append(embedding)
        
        return np.vstack(embeddings)
```

---

### 🔧 **OPCIÓN B: Usar Sentence Transformer + Descripciones VLM**

Si no quieres modificar tu API, usa el VLM para generar descripciones y luego embeddings con otro modelo:

```python
# src/core/embeddings/hybrid_embedder.py

import requests
import base64
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

class HybridEmbedder:
    """
    Estrategia híbrida:
    1. Qwen3-VL genera descripción textual de la imagen
    2. Sentence-BERT convierte descripción a embedding
    
    Ventajas:
    - No requiere modificar API
    - Embeddings de alta calidad (sentence-transformers)
    
    Desventajas:
    - 2 pasos (más lento)
    - Pierde información visual directa
    """
    
    def __init__(
        self,
        qwen_api_endpoint: str = "http://localhost:8001",
        sentence_model: str = "all-MiniLM-L6-v2"
    ):
        self.qwen_endpoint = f"{qwen_api_endpoint}/qwen3/chat/completions"
        self.sentence_encoder = SentenceTransformer(sentence_model)
        self.embedding_dim = self.sentence_encoder.get_sentence_embedding_dimension()
    
    def generate_embedding(
        self,
        image_path: Path,
        text_context: str = "vehicle damage"
    ) -> np.ndarray:
        """
        Paso 1: VLM describe imagen
        Paso 2: Sentence-BERT embeddings de descripción
        """
        
        # 1. Generar descripción con Qwen3-VL
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        description_prompt = f"""Describe the visual characteristics of this image showing {text_context}.
Focus on:
- Type of damage visible
- Location and extent
- Visual patterns and textures
- Color and lighting conditions

Be detailed and objective."""
        
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
            "max_tokens": 256
        }
        
        response = requests.post(self.qwen_endpoint, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise RuntimeError(f"Qwen API error: {response.text}")
        
        description = response.json()['choices'][0]['message']['content']
        
        # 2. Generar embedding de la descripción
        embedding = self.sentence_encoder.encode(
            description,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        return embedding.astype(np.float32)
```

**Pros de Opción B**:
- ✅ No requiere modificar API
- ✅ Rápido de implementar
- ✅ Embeddings probados (sentence-transformers)

**Contras de Opción B**:
- ❌ Más lento (2 llamadas por imagen)
- ❌ Pierde información visual pura
- ❌ Dependencia adicional

---

### 🚀 **OPCIÓN C: Usar CLIP Embeddings (Rápido pero Menos Preciso)**

Usar modelo CLIP local para embeddings visuales directos:

```python
# src/core/embeddings/clip_embedder.py

import torch
from PIL import Image
from pathlib import Path
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedder:
    """
    Embeddings visuales usando CLIP
    
    Ventajas:
    - No depende de tu API
    - Rápido (local)
    - Embeddings visuales puros
    
    Desventajas:
    - Menos específico para tu dominio
    - No usa Qwen3-VL
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embedding_dim = self.model.config.projection_dim  # 512 para base
    
    def generate_embedding(
        self,
        image_path: Path,
        text_context: str = None
    ) -> np.ndarray:
        """Genera embedding visual con CLIP"""
        
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Extract image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten().astype(np.float32)
```

---

## 🎯 RECOMENDACIÓN FINAL

### Para POC Rápido (Esta Semana)
**→ Usa OPCIÓN B (Hybrid Embedder)**
- No requiere modificar API
- Fácil de implementar
- Suficientemente bueno para validar arquitectura

### Para Producción (Próxima Iteración)
**→ Implementa OPCIÓN A (Endpoint Embeddings Nativo)**
- Mejor calidad (usa Qwen3-VL directamente)
- Más rápido (1 llamada)
- Escalable

---

## 📝 CÓDIGO ACTUALIZADO PARA TU API

### Test de Integración

```python
# tests/test_qwen_api_integration.py

import requests
import base64
from pathlib import Path

def test_chat_completion():
    """Test tu endpoint actual de chat"""
    
    # Cargar imagen de prueba
    image_path = Path("test_images/sample_damage.jpg")
    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Payload
    payload = {
        "model": "Qwen3-VL-4B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe any damage visible in this vehicle image"
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
        "max_tokens": 512
    }
    
    # Request
    response = requests.post(
        "http://localhost:8001/qwen3/chat/completions",
        json=payload,
        timeout=60
    )
    
    assert response.status_code == 200, f"Error: {response.text}"
    
    result = response.json()
    
    print("\n✅ Chat Completion Test PASSED")
    print(f"Response: {result['choices'][0]['message']['content'][:200]}...")
    print(f"Tokens: {result['usage']['prompt_tokens']} → {result['usage']['completion_tokens']}")
    
    return result


def test_health():
    """Test health endpoint"""
    response = requests.get("http://localhost:8001/health")
    
    assert response.status_code == 200
    health = response.json()
    
    print("\n✅ Health Check PASSED")
    print(f"Model loaded: {health['model_loaded']}")
    print(f"Device: {health['device']}")
    if 'gpu_name' in health:
        print(f"GPU: {health['gpu_name']}")
    
    return health


if __name__ == "__main__":
    print("="*60)
    print("Testing Qwen3-VL API Integration")
    print("="*60)
    
    # Health check
    health = test_health()
    
    # Chat completion
    result = test_chat_completion()
    
    print("\n" + "="*60)
    print("All tests PASSED ✅")
    print("="*60)
```

---

## 🔄 FLUJO DE TRABAJO ACTUALIZADO

### Para POC con Opción B (Híbrida)

```python
# scripts/03_generate_embeddings.py (ACTUALIZADO)

from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from src.core.embeddings.hybrid_embedder import HybridEmbedder

def generate_all_embeddings(
    crops_metadata_path: Path,
    output_dir: Path,
    batch_size: int = 8
):
    """
    Genera embeddings usando estrategia híbrida
    """
    
    # Cargar metadata
    with open(crops_metadata_path) as f:
        crops_metadata = json.load(f)
    
    print(f"📊 Total de crops a procesar: {len(crops_metadata)}")
    
    # Inicializar embedder híbrido
    embedder = HybridEmbedder(
        qwen_api_endpoint="http://localhost:8001"
    )
    
    print(f"🔧 Embedding dimension: {embedder.embedding_dim}")
    
    # Preparar datos
    image_paths = [Path(m['crop_path']) for m in crops_metadata]
    text_contexts = [
        f"{m['damage_type'].replace('_', ' ')} damage"
        for m in crops_metadata
    ]
    
    # Generar embeddings
    print("🧠 Generando embeddings...")
    embeddings = []
    
    for i in tqdm(range(len(image_paths))):
        embedding = embedder.generate_embedding(
            image_paths[i],
            text_contexts[i]
        )
        embeddings.append(embedding)
    
    embeddings = np.vstack(embeddings)
    
    print(f"✅ Generados {embeddings.shape[0]} embeddings de {embeddings.shape[1]} dims")
    
    # Guardar embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    # Guardar metadata enriquecida
    for i, meta in enumerate(crops_metadata):
        meta['embedding_index'] = i
    
    enriched_metadata_path = output_dir / "enriched_crops_metadata.json"
    with open(enriched_metadata_path, 'w') as f:
        json.dump(crops_metadata, f, indent=2)
    
    print(f"💾 Embeddings guardados en: {embeddings_path}")
    print(f"📄 Metadata enriquecida: {enriched_metadata_path}")
    
    return embeddings, crops_metadata


if __name__ == "__main__":
    CROPS_METADATA = Path("data/processed/metadata/crops_metadata.json")
    OUTPUT_DIR = Path("data/processed/embeddings")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    embeddings, metadata = generate_all_embeddings(
        CROPS_METADATA,
        OUTPUT_DIR
    )
```

---

## 📊 COMPARATIVA DE OPCIONES

| Característica | Opción A (Nativo) | Opción B (Híbrido) | Opción C (CLIP) |
|----------------|-------------------|--------------------|--------------------|
| **Requiere modificar API** | ✅ Sí | ❌ No | ❌ No |
| **Velocidad** | ⚡️ Rápido | 🐢 Lento (2 pasos) | ⚡️ Muy rápido |
| **Calidad embeddings** | 🏆 Excelente | ✅ Buena | ⚠️ Básica |
| **Usa Qwen3-VL** | ✅ Directo | ⚠️ Solo descripción | ❌ No |
| **Complejidad** | 🔧 Media | 🎯 Baja | 🎯 Muy baja |
| **Ideal para** | Producción | POC rápido | Prototipo inicial |
| **Tiempo implementación** | 2-3 horas | 30 min | 15 min |

---

## ✅ PLAN DE ACCIÓN RECOMENDADO

### ESTA SEMANA (POC)
1. **Implementar Opción B (Híbrida)** para validar arquitectura
2. Generar embeddings de 100 imágenes
3. Construir índice FAISS
4. Probar retrieval

### PRÓXIMA ITERACIÓN (Producción)
1. **Implementar Opción A** (modificar `api.py`)
2. Re-generar embeddings con calidad nativa
3. Comparar métricas vs Opción B
4. Escalar a dataset completo

---

## 🚀 INICIO RÁPIDO

```bash
# 1. Instalar dependencia adicional
pip install sentence-transformers

# 2. Test de tu API
python tests/test_qwen_api_integration.py

# 3. Generar embeddings (Opción B)
python scripts/03_generate_embeddings.py

# 4. Construir índice
python scripts/04_build_indices.py

# 5. Probar RAG
python scripts/05_run_inference.py
```

---

## 📞 SIGUIENTE PASO

**¿Cuál opción prefieres para el POC?**

- **Opción B (Híbrida)**: Empezar YA sin tocar la API ✅
- **Opción A (Nativa)**: Modificar API primero para mejor calidad

Te recomiendo **empezar con B** para validar rápido, y migrar a A si los resultados son prometedores.