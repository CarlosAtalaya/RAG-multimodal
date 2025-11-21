# rag/multimodal_embedder.py

"""
🧠 MULTIMODAL EMBEDDER - Visual + Textual

Combina embeddings visuales (DINOv3) con semántica textual (SentenceTransformer)

✨ Ahora con soporte para imágenes SIN daño:
- Detecta flag 'has_damage' en metadata
- Genera descripciones optimizadas según caso
- Mantiene dimensión fija de 1408 dims
"""

from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter

from .dinov3_embedder import DINOv3ViTLEmbedder


class MultimodalEmbedder:
    """Embedder híbrido visual + textual con soporte para imágenes sin daño"""
    
    def __init__(
        self,
        visual_weight: float = 0.6,
        text_weight: float = 0.4,
        text_model: str = "all-MiniLM-L6-v2",
        use_bfloat16: bool = False,
        verbose: bool = False
    ):
        """
        Args:
            visual_weight: Peso para embedding visual (default: 0.6)
            text_weight: Peso para embedding textual (default: 0.4)
            text_model: Modelo Sentence-BERT
            use_bfloat16: Usar bfloat16 para DINOv3
            verbose: Verbosidad
        """
        assert abs(visual_weight + text_weight - 1.0) < 1e-6, \
            "visual_weight + text_weight debe ser 1.0"
        
        self.visual_weight = visual_weight
        self.text_weight = text_weight
        self.verbose = verbose
        
        if verbose:
            print(f"🔧 Inicializando MultimodalEmbedder...")
            print(f"   - Visual weight: {visual_weight}")
            print(f"   - Text weight: {text_weight}")
        
        # Visual embedder
        print(f"📸 Cargando visual encoder (DINOv3)...")
        self.visual_embedder = DINOv3ViTLEmbedder(
            use_bfloat16=use_bfloat16,
            verbose=verbose
        )
        
        # Text embedder
        print(f"📝 Cargando text encoder ({text_model})...")
        from sentence_transformers import SentenceTransformer
        self.text_embedder = SentenceTransformer(text_model)
        
        # Dimensiones
        self.visual_dim = 1024
        self.text_dim = self.text_embedder.get_sentence_embedding_dimension()
        self.total_dim = self.visual_dim + self.text_dim
        
        if verbose:
            print(f"\n✅ MultimodalEmbedder inicializado")
            print(f"   - Visual dim: {self.visual_dim}")
            print(f"   - Text dim: {self.text_dim}")
            print(f"   - Total dim: {self.total_dim}")
            print(f"   - Supports no damage: ✅\n")
    
    def build_text_description(self, metadata: Dict) -> str:
        """
        Construye descripción textual desde metadata
        
        Soporta:
        - Imágenes CON daño (has_damage=True)
        - Imágenes SIN daño (has_damage=False)
        
        Formato CON daño: "N damage_type1, M damage_type2 on vehicle_zone (area)"
        Formato SIN daño: "no visible damage on vehicle_zone (area)"
        """
        has_damage = metadata.get('has_damage', True)
        defect_types = metadata.get('defect_types', [])
        
        # ✨ CASO 1: SIN DAÑO
        if not has_damage or not defect_types:
            zone_desc = metadata.get('zone_description', 'vehicle')
            zone_area = metadata.get('zone_area', 'unknown')
            return f"no visible damage on {zone_desc} ({zone_area} area)"
        
        # ✨ CASO 2: CON DAÑO
        type_counts = Counter(defect_types)
        
        damage_parts = []
        for dtype, count in type_counts.most_common():
            plural = "s" if count > 1 else ""
            # Reemplazar underscores para mejor legibilidad
            dtype_readable = dtype.replace('_', ' ')
            damage_parts.append(f"{count} {dtype_readable}{plural}")
        
        damage_desc = ", ".join(damage_parts)
        
        zone_desc = metadata.get('zone_description', 'unknown area')
        zone_area = metadata.get('zone_area', 'unknown')
        
        full_desc = f"{damage_desc} on {zone_desc} ({zone_area} area)"
        
        return full_desc
    
    def generate_visual_embedding(
        self,
        image_path: Union[str, Path],
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
        image_path: Union[str, Path],
        metadata: Dict,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Genera embedding híbrido (visual + text)
        
        Returns:
            hybrid_embedding: Vector de 1408 dims
            debug_info: Dict con info de componentes
        """
        # Convertir a Path si es str
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        # 1. Visual embedding
        visual_emb = self.generate_visual_embedding(image_path, normalize=True)
        
        # 2. Text embedding (con soporte para sin daño)
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
            'has_damage': metadata.get('has_damage', True),  # ✨ NUEVO
            'visual_norm': float(np.linalg.norm(visual_emb)),
            'text_norm': float(np.linalg.norm(text_emb)),
            'hybrid_norm': float(np.linalg.norm(hybrid))
        }
        
        return hybrid.astype('float32'), debug_info
    
    def generate_embedding(
        self,
        image_path: Union[str, Path],
        normalize: bool = True,
        metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Genera embedding híbrido (SIEMPRE 1408 dims)
        
        Args:
            image_path: Ruta a imagen
            normalize: Normalizar embedding
            metadata: Metadata para descripción textual
                     Si None, usa metadata dummy CON daño
        
        Returns:
            Vector numpy [1408] SIEMPRE
        """
        if metadata is None:
            # ✅ Usar metadata dummy para mantener dimensión
            if self.verbose:
                print("     ℹ️  No metadata provided, using dummy metadata (with damage)")
            
            metadata = {
                'has_damage': True,  # ✨ Asumir daño por defecto
                'defect_types': ['unknown'],
                'zone_description': 'unknown area',
                'zone_area': 'unknown'
            }
        
        # ✅ SIEMPRE generar embedding híbrido
        hybrid_emb, _ = self.generate_hybrid_embedding(
            image_path=image_path,
            metadata=metadata,
            normalize=normalize
        )
        return hybrid_emb
    
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
            'version': '1.1',  # ✨ Incrementado
            'visual_model': 'dinov3-vitl16',
            'text_model': self.text_embedder._modules['0'].auto_model.config.name_or_path,
            'visual_dim': self.visual_dim,
            'text_dim': self.text_dim,
            'total_dim': self.total_dim,
            'visual_weight': self.visual_weight,
            'text_weight': self.text_weight,
            'supports_no_damage': True  # ✨ NUEVO flag
        }


# Test unitario
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TEST: MultimodalEmbedder con soporte sin daño")
    print("="*70 + "\n")
    
    embedder = MultimodalEmbedder(
        visual_weight=0.6,
        text_weight=0.4,
        verbose=True
    )
    
    # Test con daño
    metadata_damage = {
        'has_damage': True,
        'defect_types': ['scratch', 'dent'],
        'zone_description': 'hood center',
        'zone_area': 'frontal'
    }
    
    text_desc_damage = embedder.build_text_description(metadata_damage)
    print(f"📝 Descripción CON daño: '{text_desc_damage}'")
    
    # Test sin daño
    metadata_no_damage = {
        'has_damage': False,
        'defect_types': [],
        'zone_description': 'hood center',
        'zone_area': 'frontal'
    }
    
    text_desc_no_damage = embedder.build_text_description(metadata_no_damage)
    print(f"📝 Descripción SIN daño: '{text_desc_no_damage}'")
    
    print("\n✅ Test completado\n")