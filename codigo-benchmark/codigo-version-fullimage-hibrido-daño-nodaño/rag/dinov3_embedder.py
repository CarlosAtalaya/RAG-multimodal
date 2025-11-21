# rag/dinov3_embedder.py

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
    Embedder para DINOv3-ViT-L/16 (300M params, 1024 dims)
    
    Modelo: facebook/dinov3-vitl16-pretrain-lvd1689m
    Optimizado para herramienta de benchmarking
    """
    
    MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    EMBEDDING_DIM = 1024
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_bfloat16: bool = False,
        cache_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Inicializa el embedder
        
        Args:
            device: 'cuda', 'cpu', o None (auto-detect)
            use_bfloat16: Usar bfloat16 (no recomendado)
            cache_dir: Directorio para cachear modelo
            verbose: Si True, imprime información detallada
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Login HF (si hay token)
        token = os.getenv("HF_TOKEN")
        if token:
            try:
                login(token=token, add_to_git_credential=False)
            except:
                pass
        
        # Dtype (mantener float32 para estabilidad)
        self.dtype = torch.float32
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🔧 INICIALIZANDO DINOv3-ViT-L/16")
            print(f"{'='*70}")
            print(f"Modelo: {self.MODEL_NAME}")
            print(f"Device: {self.device}")
            print(f"Dtype: {self.dtype}")
            print(f"Embedding dim: {self.EMBEDDING_DIM}")
            print(f"{'='*70}\n")
        else:
            print(f"🔧 Inicializando DINOv3-ViT-L/16...")
            print(f"   Device: {self.device}")
            print(f"   Embedding dim: {self.EMBEDDING_DIM}")
        
        # Cargar image processor
        if self.verbose:
            print("📥 Descargando processor...")
        
        self.processor = AutoImageProcessor.from_pretrained(
            self.MODEL_NAME,
            cache_dir=cache_dir
        )
        
        # Cargar modelo
        if self.verbose:
            print("📥 Descargando modelo (puede tardar varios minutos)...")
        
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=self.dtype,
            cache_dir=cache_dir
        )
        
        # Mover a device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\n✅ Modelo cargado exitosamente")
            print(f"   Total parámetros: {total_params / 1e6:.1f}M")
            if self.device == "cuda":
                print(f"   Memoria GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"{'='*70}\n")
        else:
            print(f"   ✅ Modelo cargado\n")
    
    def generate_embedding(
        self,
        image_path: Union[str, Path],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Genera embedding visual de una imagen
        
        Args:
            image_path: Ruta a imagen
            normalize: Normalizar embedding
        
        Returns:
            Vector numpy [1024]
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output  # [1, 1024]
            
            if normalize:
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().flatten().astype(np.float32)
    
    def generate_batch_embeddings(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 16,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Genera embeddings en batch
        
        Returns:
            Array numpy [len(image_paths), 1024]
        """
        all_embeddings = []
        
        image_paths = [Path(p) if isinstance(p, str) else p for p in image_paths]
        
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Generando embeddings",
                total=(len(image_paths) + batch_size - 1) // batch_size
            )
        
        for i in iterator:
            batch_paths = image_paths[i:i+batch_size]
            
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    if self.verbose:
                        print(f"\n⚠️  Error cargando {path.name}: {e}")
                    images.append(Image.new('RGB', (224, 224), (114, 114, 114)))
            
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output
                
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def get_model_info(self) -> dict:
        """Retorna información del modelo"""
        return {
            'model_name': self.MODEL_NAME,
            'embedding_dim': self.EMBEDDING_DIM,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'total_params': sum(p.numel() for p in self.model.parameters()),
        }


# Test rápido
if __name__ == "__main__":
    print("🧪 Test DINOv3-ViT-L/16 Embedder\n")
    
    embedder = DINOv3ViTLEmbedder(verbose=True)
    
    info = embedder.get_model_info()
    print("\n📋 Información del modelo:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Embedder listo!")