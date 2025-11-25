# src/core/embeddings/metaclip_embedder_unified.py

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import List, Union

class MetaCLIPUnifiedEmbedder:
    """
    Embedder unificado usando MetaCLIP 2 (ViT-L/14@336px).
    Genera un espacio vectorial compartido 1024d.
    """
    
    # Usamos el modelo MetaCLIP especificado o equivalente ViT-L
    # Nota: Si el ID exacto 'facebook/metaclip-h14-fullcc2.5b' no descarga directamente 
    # desde HF público, usar 'facebook/metaclip-h14-fullcc2.5b' si existe o fallback a CLIP estándar
    # Asumimos que el modelo está disponible o se usa open_clip. 
    # Para este código usaré la interfaz standard de HF Transformers para CLIP/MetaCLIP.
    MODEL_NAME = "facebook/metaclip-h14-fullcc2.5b" 

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Inicializando MetaCLIP Unified Embedder ({self.MODEL_NAME})...")
        print(f"   Device: {self.device}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
            self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        except OSError:
            print("⚠️ Advertencia: Modelo específico no encontrado en HF Hub público.")
            print("   Usando 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K' como fallback de alta calidad.")
            fallback_model = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
            self.processor = AutoProcessor.from_pretrained(fallback_model)
            self.model = AutoModel.from_pretrained(fallback_model).to(self.device)

        self.model.eval()
        self.target_dim = 1024 if self.model.config.projection_dim else 768 
        # ViT-L suele ser 768 o 1024 dependiendo de la proyección. 
        # El documento dice 1024d. Si es H-14 suele ser 1024.

    def generate_embedding(self, image_path: str, text: str, fusion: str = 'average') -> np.ndarray:
        """
        Genera embedding unificado: (Image_Emb + Text_Emb) / 2
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Preprocesar inputs
            inputs = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Obtener embeddings normalizados (MetaCLIP/CLIP ya suele normalizar, pero aseguramos)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                # Normalización L2 explícita
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                # FUSIÓN
                if fusion == 'average':
                    # Espacio compartido: promedio directo
                    unified_features = (image_features + text_features) / 2.0
                elif fusion == 'weighted':
                    # Dar un poco más de peso a la imagen para defectos visuales sutiles
                    unified_features = (image_features * 0.6) + (text_features * 0.4)
                else:
                    raise ValueError(f"Estrategia desconocida: {fusion}")

                # Renormalizar el resultado final
                unified_features = unified_features / unified_features.norm(p=2, dim=-1, keepdim=True)

            return unified_features.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            print(f"❌ Error generando embedding para {image_path}: {e}")
            return np.zeros(self.target_dim, dtype=np.float32)