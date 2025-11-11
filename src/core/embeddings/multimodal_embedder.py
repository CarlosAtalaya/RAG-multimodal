# src/core/embeddings/multimodal_embedder.py

"""
🧠 MULTIMODAL EMBEDDER - Visual + Textual

Combina embeddings visuales (DINOv3) con semántica textual (SentenceTransformer)
para mejorar la capacidad de retrieval en casos donde la similitud visual es baja.

Arquitectura:
- Visual: DINOv3-ViT-L/16 (1024 dims)
- Text: all-MiniLM-L6-v2 (384 dims)
- Fusion: Weighted concatenation (1408 dims total)
"""

from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch

from .dinov3_vitl_embedder import DINOv3ViTLEmbedder


class MultimodalEmbedder:
    """Embedder híbrido visual + textual"""
    
    def __init__(
        self,
        visual_weight: float = 0.6,
        text_weight: float = 0.4,
        text_model: str = "all-MiniLM-L6-v2",
        use_bfloat16: bool = True
    ):
        """
        Args:
            visual_weight: Peso para embedding visual (default: 0.6)
            text_weight: Peso para embedding textual (default: 0.4)
            text_model: Modelo de Sentence-BERT
            use_bfloat16: Usar bfloat16 para DINOv3
        """
        assert abs(visual_weight + text_weight - 1.0) < 1e-6, \
            "visual_weight + text_weight debe ser 1.0"
        
        self.visual_weight = visual_weight
        self.text_weight = text_weight
        
        print(f"🔧 Inicializando MultimodalEmbedder...")
        print(f"   - Visual weight: {visual_weight}")
        print(f"   - Text weight: {text_weight}")
        
        # Visual embedder
        print(f"\n📸 Cargando visual encoder (DINOv3)...")
        self.visual_embedder = DINOv3ViTLEmbedder(use_bfloat16=use_bfloat16)
        
        # Text embedder
        print(f"\n📝 Cargando text encoder ({text_model})...")
        from sentence_transformers import SentenceTransformer
        self.text_embedder = SentenceTransformer(text_model)
        
        # Dimensiones
        self.visual_dim = 1024
        self.text_dim = self.text_embedder.get_sentence_embedding_dimension()
        self.total_dim = self.visual_dim + self.text_dim
        
        print(f"\n✅ MultimodalEmbedder inicializado")
        print(f"   - Visual dim: {self.visual_dim}")
        print(f"   - Text dim: {self.text_dim}")
        print(f"   - Total dim: {self.total_dim}\n")
    
    def build_text_description(self, metadata: Dict) -> str:
        """
        Construye descripción textual rica desde metadata
        
        Formato optimizado para retrieval:
        "N damage_type1, M damage_type2 on vehicle_zone (area)"
        
        Ejemplo:
        "3 surface scratches, 1 dent on hood center (frontal area)"
        """
        from collections import Counter
        
        # Tipos de daño con conteo
        defect_types = metadata.get('defect_types', [])
        if not defect_types:
            return "no visible damage"
        
        type_counts = Counter(defect_types)
        
        # Construir parte de daños
        damage_parts = []
        for dtype, count in type_counts.most_common():
            plural = "s" if count > 1 else ""
            damage_parts.append(f"{count} {dtype}{plural}")
        
        damage_desc = ", ".join(damage_parts)
        
        # Info de zona
        zone_desc = metadata.get('zone_description', 'unknown area')
        zone_area = metadata.get('zone_area', 'unknown')
        
        # Descripción completa
        full_desc = f"{damage_desc} on {zone_desc} ({zone_area} area)"
        
        return full_desc
    
    def generate_visual_embedding(
        self,
        image_path: Path,
        normalize: bool = True
    ) -> np.ndarray:
        """Genera embedding visual (DINOv3)"""
        return self.visual_embedder.generate_embedding(
            image_path=image_path,
            normalize=normalize
        )
    
    def generate_text_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """Genera embedding textual (Sentence-BERT)"""
        embedding = self.text_embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embedding.astype('float32')
    
    def generate_hybrid_embedding(
        self,
        image_path: Path,
        metadata: Dict,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Genera embedding híbrido (visual + text)
        
        Returns:
            hybrid_embedding: Vector de 1408 dims
            debug_info: Dict con info de componentes
        """
        # 1. Visual embedding
        visual_emb = self.generate_visual_embedding(image_path, normalize=True)
        
        # 2. Text embedding
        text_desc = self.build_text_description(metadata)
        text_emb = self.generate_text_embedding(text_desc, normalize=True)
        
        # 3. Weighted concatenation
        visual_weighted = visual_emb * self.visual_weight
        text_weighted = text_emb * self.text_weight
        
        # 4. Concatenar
        hybrid = np.concatenate([visual_weighted, text_weighted])
        
        # 5. Normalizar embedding final
        if normalize:
            norm = np.linalg.norm(hybrid)
            if norm > 0:
                hybrid = hybrid / norm
        
        debug_info = {
            'text_description': text_desc,
            'visual_norm': float(np.linalg.norm(visual_emb)),
            'text_norm': float(np.linalg.norm(text_emb)),
            'hybrid_norm': float(np.linalg.norm(hybrid))
        }
        
        return hybrid.astype('float32'), debug_info
    
    def generate_batch_embeddings(
        self,
        image_paths: List[Path],
        metadata_list: List[Dict],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Genera embeddings híbridos en batch
        
        Returns:
            embeddings: Array (N, 1408)
            debug_info_list: Lista de debug info
        """
        assert len(image_paths) == len(metadata_list), \
            "image_paths y metadata_list deben tener mismo tamaño"
        
        n_samples = len(image_paths)
        embeddings = np.zeros((n_samples, self.total_dim), dtype='float32')
        debug_info_list = []
        
        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=n_samples, desc="Generating hybrid embeddings")
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_paths = image_paths[i:batch_end]
            batch_metadata = metadata_list[i:batch_end]
            
            for j, (img_path, meta) in enumerate(zip(batch_paths, batch_metadata)):
                idx = i + j
                
                try:
                    emb, debug = self.generate_hybrid_embedding(
                        image_path=img_path,
                        metadata=meta,
                        normalize=True
                    )
                    embeddings[idx] = emb
                    debug_info_list.append(debug)
                
                except Exception as e:
                    print(f"\n❌ Error en imagen {img_path}: {e}")
                    # Embedding cero si falla
                    embeddings[idx] = np.zeros(self.total_dim, dtype='float32')
                    debug_info_list.append({'error': str(e)})
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        return embeddings, debug_info_list
    
    def get_model_info(self) -> Dict:
        """Info del modelo"""
        return {
            'type': 'multimodal_hybrid',
            'visual_model': 'dinov3-vitl16',
            'text_model': self.text_embedder._modules['0'].auto_model.config.name_or_path,
            'visual_dim': self.visual_dim,
            'text_dim': self.text_dim,
            'total_dim': self.total_dim,
            'visual_weight': self.visual_weight,
            'text_weight': self.text_weight
        }


# ============================================
# TEST UNITARIO
# ============================================

def test_multimodal_embedder():
    """Test básico del embedder multimodal"""
    print("\n" + "="*70)
    print("🧪 TEST: MultimodalEmbedder")
    print("="*70 + "\n")
    
    # Inicializar
    embedder = MultimodalEmbedder(
        visual_weight=0.6,
        text_weight=0.4
    )
    
    # Metadata de ejemplo
    metadata = {
        'defect_types': ['surface_scratch', 'surface_scratch', 'dent'],
        'zone_description': 'hood center',
        'zone_area': 'frontal'
    }
    
    # Descripción textual
    text_desc = embedder.build_text_description(metadata)
    print(f"📝 Descripción textual generada:")
    print(f"   '{text_desc}'\n")
    
    # Text embedding
    text_emb = embedder.generate_text_embedding(text_desc)
    print(f"📊 Text embedding:")
    print(f"   Shape: {text_emb.shape}")
    print(f"   Norm: {np.linalg.norm(text_emb):.4f}\n")
    
    print("✅ Test completado correctamente\n")


if __name__ == "__main__":
    test_multimodal_embedder()