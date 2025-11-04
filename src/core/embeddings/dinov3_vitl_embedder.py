# src/core/embeddings/dinov3_vitl_embedder.py

import torch
import os
from PIL import Image
from pathlib import Path
from huggingface_hub import login
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from typing import List, Union, Optional
from tqdm import tqdm

class DINOv3ViTLEmbedder:
    """
    Embedder específico para DINOv3-ViT-L/16 (300M params, 1024 dims)
    
    Modelo: facebook/dinov3-vitl16-pretrain-lvd1689m
    
    Características:
    - 300M parámetros
    - 1024 dimensiones de embedding
    - Patch size: 16x16
    - Entrenado en LVD-1689M (1.689 billones de imágenes)
    - State-of-the-art en tareas densas
    
    Ventajas para detección de defectos:
    - Características visuales de alta calidad
    - Excelente para detección de anomalías
    - Balance óptimo calidad/velocidad
    - No requiere fine-tuning
    """
    
    MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    EMBEDDING_DIM = 1024
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_bfloat16: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Inicializa el embedder
        
        Args:
            device: 'cuda', 'cpu', o None (auto-detect)
            use_bfloat16: Usar bfloat16 para reducir memoria (recomendado en GPU)
            cache_dir: Directorio para cachear el modelo descargado
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        token = os.getenv("HF_TOKEN")
        login(token=token)
        
        # Determinar dtype óptimo
        if use_bfloat16 and self.device == "cuda":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        print(f"\n{'='*70}")
        print(f"🔧 INICIALIZANDO DINOv3-ViT-L/16")
        print(f"{'='*70}")
        print(f"Modelo: {self.MODEL_NAME}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")
        print(f"Embedding dim: {self.EMBEDDING_DIM}")
        print(f"{'='*70}\n")
        
        # Cargar image processor
        print("📥 Descargando processor...")
        self.processor = AutoImageProcessor.from_pretrained(
            self.MODEL_NAME,
            cache_dir=cache_dir
        )
        
        # Cargar modelo
        print("📥 Descargando modelo (puede tardar varios minutos la primera vez)...")
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=self.dtype,
            cache_dir=cache_dir
        )
        
        # Mover a device
        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Obtener información del modelo
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n✅ Modelo cargado exitosamente")
        print(f"   Total parámetros: {total_params / 1e6:.1f}M")
        print(f"   Parámetros entrenables: {trainable_params / 1e6:.1f}M")
        print(f"   Memoria GPU asignada: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if self.device == "cuda" else "")
        print(f"{'='*70}\n")
    
    def generate_embedding(
        self,
        image_path: Union[str, Path],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Genera embedding de una imagen
        
        Args:
            image_path: Ruta a la imagen
            normalize: Si True, normaliza el embedding (recomendado para FAISS)
        
        Returns:
            Vector numpy de dimensión 1024
        """
        # Cargar imagen
        image = Image.open(image_path).convert('RGB')
        
        # Procesar imagen
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        
        # Mover a device correcto
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generar embedding
        with torch.inference_mode():
            outputs = self.model(**inputs)
            
            # Usar pooled output (CLS token)
            embedding = outputs.pooler_output  # Shape: [1, 1024]
            
            # Normalizar si se solicita
            if normalize:
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        # Convertir a numpy
        return embedding.cpu().numpy().flatten().astype(np.float32)
    
    def generate_batch_embeddings(
        self,
        image_paths: List[Path],
        batch_size: int = 16,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Genera embeddings en batch (más eficiente)
        
        Args:
            image_paths: Lista de rutas a imágenes
            batch_size: Tamaño del batch (16 recomendado para ViT-L)
            normalize: Si True, normaliza los embeddings
            show_progress: Mostrar barra de progreso
        
        Returns:
            Array numpy de shape (len(image_paths), 1024)
        """
        all_embeddings = []
        
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator, 
                desc="Generando embeddings",
                total=(len(image_paths) + batch_size - 1) // batch_size
            )
        
        for i in iterator:
            batch_paths = image_paths[i:i+batch_size]
            
            # Cargar imágenes del batch
            images = []
            valid_indices = []
            
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                    valid_indices.append(i + idx)
                except Exception as e:
                    print(f"\n⚠️  Error cargando {path.name}: {e}")
                    # Imagen placeholder (gris)
                    images.append(Image.new('RGB', (224, 224), (114, 114, 114)))
                    valid_indices.append(i + idx)
            
            # Procesar batch completo
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            )
            
            # Mover a device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generar embeddings
            with torch.inference_mode():
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output  # Shape: [batch_size, 1024]
                
                # Normalizar si se solicita
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            # Guardar embeddings del batch
            all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def get_detailed_embeddings(
        self,
        image_path: Union[str, Path],
        normalize: bool = True
    ) -> dict:
        """
        Obtiene embeddings detallados incluyendo patch-level features
        
        Útil para análisis más detallados o visualizaciones
        
        Returns:
            dict con:
            - 'pooled': embedding global (CLS token) [1024]
            - 'cls_token': solo CLS token [1024]
            - 'patch_tokens': embeddings de patches [num_patches, 1024]
            - 'register_tokens': tokens de registro [4, 1024]
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            
            # Embedding global (pooled)
            pooled = outputs.pooler_output
            
            # Hidden states completos
            last_hidden = outputs.last_hidden_state  # [1, seq_len, 1024]
            
            # Separar tokens
            # Formato DINOv3: [CLS] + [4 REGISTERS] + [PATCHES]
            num_registers = 4
            
            cls_token = last_hidden[:, 0]  # [1, 1024]
            register_tokens = last_hidden[:, 1:1+num_registers]  # [1, 4, 1024]
            patch_tokens = last_hidden[:, 1+num_registers:]  # [1, num_patches, 1024]
            
            # Normalizar si se solicita
            if normalize:
                pooled = pooled / pooled.norm(dim=-1, keepdim=True)
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
        
        return {
            'pooled': pooled.cpu().numpy().flatten().astype(np.float32),
            'cls_token': cls_token.cpu().numpy().flatten().astype(np.float32),
            'patch_tokens': patch_tokens.cpu().numpy().squeeze(0).astype(np.float32),
            'register_tokens': register_tokens.cpu().numpy().squeeze(0).astype(np.float32)
        }
    
    def get_model_info(self) -> dict:
        """Retorna información del modelo"""
        return {
            'model_name': self.MODEL_NAME,
            'embedding_dim': self.EMBEDDING_DIM,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'patch_size': self.model.config.patch_size,
            'num_registers': self.model.config.num_register_tokens,
            'hidden_size': self.model.config.hidden_size,
        }


# Test rápido
if __name__ == "__main__":
    print("🧪 Test DINOv3-ViT-L/16 Embedder\n")
    
    # Inicializar
    embedder = DINOv3ViTLEmbedder()
    
    # Mostrar info del modelo
    info = embedder.get_model_info()
    print("\n📋 Información del modelo:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Embedder listo para usar!")