# 🚗 RAG Multimodal MetaCLIP 2 - Dataset Balanceado con Contexto Enriquecido

## 📋 Índice

1. [Visión General](#-visión-general)
2. [Arquitectura del Sistema](#-arquitectura-del-sistema)
3. [Dataset](#-dataset)
4. [Módulos del Sistema](#-módulos-del-sistema)
5. [Pipeline de Implementación](#-pipeline-de-implementación)
6. [Estructura de Archivos](#-estructura-de-archivos)
7. [Esquema de Metadata](#-esquema-de-metadata)
8. [Configuración y Parámetros](#-configuración-y-parámetros)
9. [Resultados Esperados](#-resultados-esperados)
10. [Próximos Pasos](#-próximos-pasos)

---

## 🎯 Visión General

### Objetivo

Implementar un sistema RAG (Retrieval-Augmented Generation) multimodal que procese un dataset balanceado de imágenes vehiculares con y sin daños, utilizando **MetaCLIP 2** para generar embeddings unificados (visual + textual) que preserven la máxima información y contexto de las imágenes originales.

### Características Principales

- ✅ **Dataset Balanceado**: 50% imágenes con daño (ko) + 50% sin daño (ok)
- ✅ **Embeddings Unificados con MetaCLIP 2**: Visual + Textual en espacio compartido (1024d)
- ✅ **Resolución Optimizada**: 448×448 con estrategia multi-patch para preservar detalles
- ✅ **Sin Pérdida de Información**: Eliminados procesados que degradan calidad visual
- ✅ **Contexto Enriquecido**: Descripciones textuales ricas con zona, parte específica y tipos de daño
- ✅ **Índice Único FAISS**: Unificación de crops damaged + clean con filtros avanzados
- ✅ **Crops Inteligentes**: Clusterizados para daños, grid adaptativo para imágenes limpias

### Cambios Clave vs. Arquitectura Anterior

| Aspecto | Antes (DINOv3+BERT) | Ahora (MetaCLIP 2) |
|---------|--------------------|--------------------|
| **Modelo de embeddings** | 2 modelos separados | ✅ **1 modelo unificado** |
| **Dimensión** | 1408d (concatenados) | ✅ **1024d (espacio compartido)** |
| **Alineación visual-texto** | Manual (weights 50/50) | ✅ **Nativa end-to-end** |
| **Velocidad** | Lenta (2 forward pass) | ✅ **2× más rápida** |
| **Resolución crops** | 448×448 fijo | ✅ **448×448 + multi-patch** |
| **Pérdida información** | Padding gris | ✅ **Sin artifacts** |
| **VRAM requerida** | ~8GB | ✅ **~6GB** |

---

## 🏗️ Arquitectura del Sistema

```
Dataset Split 2 (552 imágenes)
           ↓
    ¿Tiene daño?
     ↙         ↘
   SÍ          NO
(276 ko)    (276 ok)
    ↓            ↓
Clustered    Vehicle Detector
  Crop       + Grid Crop
Generator      Generator
    ↓            ↓
~850 crops   ~1,500-2,000 crops
(448×448)       (448×448)
    ↓            ↓
Damage       Clean
Contextualizer  Contextualizer
    ↓            ↓
Contexto     Contexto
CON daño     SIN daño
    ↓            ↓
    └─────┬──────┘
          ↓
   MetaCLIP 2 Embedder
   (Fusion: Average)
          ↓
   Embeddings Unificados
      (1024 dims)
          ↓
   Índice FAISS Unificado
    (IndexHNSWFlat)
          ↓
  Retriever con Filtros
  (zona + has_damage)
```

---

## 📊 Dataset

### Estructura del Dataset Split 2

```
data/raw/dataset_split_2/
├── Imágenes CON daño (276 imágenes)
│   ├── zona1_ko_2_3_1554817134014_zona_4_imageDANO_original.jpg
│   ├── zona1_ko_2_3_1554817134014_zona_4_imageDANO_original.json          # Ground truth
│   └── zona1_ko_2_3_1554817134014_zona_4_labelDANO_modificado.json        # Segmentación
│
└── Imágenes SIN daño (276 imágenes)
    ├── zona1_ok_2_3_1554373063646_zona_6_imageDANO_original.jpg
    └── zona1_ok_2_3_1554373063646_zona_6_imageDANO_original.json          # Ground truth "No damage"
```

### Características

| Aspecto | Valor |
|---------|-------|
| **Total imágenes** | 552 |
| **Con daño (ko)** | 276 (50%) |
| **Sin daño (ok)** | 276 (50%) |
| **Formato** | JPG |
| **Anotaciones** | JSON (LabelMe format) |

### Naming Convention

```
zona1_{ko|ok}_2_3_{timestamp}_zona_{1-10}_imageDANO_original.jpg
  │     │                         │
  │     │                         └─ Zona del vehículo (1-10)
  │     └─ ko: con daño | ok: sin daño
  └─ Identificador del dataset
```

### Zonas del Vehículo

| Zona | Nombre Técnico | Descripción | Área |
|------|----------------|-------------|------|
| 1 | front_left_fender | Guardabarros delantero izquierdo | frontal |
| 2 | hood_center | Capó central | frontal |
| 3 | front_right_fender | Guardabarros delantero derecho | frontal |
| 4 | rear_left_quarter | Panel trasero izquierdo | posterior |
| 5 | rear_bumper | Parachoques trasero | posterior |
| 6 | rear_right_quarter | Panel trasero derecho | posterior |
| 7 | driver_side_door | Puerta del conductor (izquierda) | lateral_left |
| 8 | driver_side_rocker | Panel lateral del conductor | lateral_left |
| 9 | passenger_side_door | Puerta del pasajero (derecha) | lateral_right |
| 10 | passenger_side_rocker | Panel lateral del pasajero | lateral_right |

### Tipos de Daño

| Label | Nombre Canónico | Descripción |
|-------|-----------------|-------------|
| 1 | surface_scratch | Arañazo superficial |
| 2 | dent | Abolladura |
| 3 | paint_peeling | Desprendimiento de pintura |
| 4 | deep_scratch | Arañazo profundo |
| 5 | crack | Grieta |
| 6 | missing_part | Parte faltante |
| 7 | missing_accessory | Accesorio faltante |
| 8 | misaligned_part | Parte desalineada |

---

## 🧩 Módulos del Sistema

### 1. Vehicle Detector 🚗

**Ubicación**: `src/core/preprocessing/vehicle_detector.py`

**Función**: Detectar el coche principal en imágenes sin daño para centrar los crops.

**Tecnología**: YOLOv8 pre-entrenado en COCO (clase 'car')

**Proceso**:
1. Detectar todos los coches en la imagen
2. Seleccionar el bbox más grande (coche principal)
3. Expandir bbox con margen (10-15%) para contexto
4. Retornar coordenadas ajustadas

**Output**:
```python
{
    'bbox': [x1, y1, x2, y2],
    'confidence': 0.95,
    'vehicle_area_ratio': 0.68  # % de imagen ocupado
}
```

**Ventajas**:
- ✅ Centra crops en el vehículo
- ✅ Minimiza fondo innecesario
- ✅ Mejora calidad de embeddings visuales

---

### 2. Grid Crop Generator 📐

**Ubicación**: `src/core/preprocessing/grid_crop_generator.py`

**Función**: Generar grid de crops 448×448 para imágenes sin daño.

**Estrategia**:
- Sliding window dentro del bbox del vehículo
- Overlap inteligente (20-30%) para cobertura completa
- Filtrado: solo crops con >70% área del coche

**Parámetros**:
```python
crop_size = 448  # Óptimo para MetaCLIP
overlap = 0.25   # 25%
min_vehicle_ratio = 0.70  # Mínimo 70% del crop debe ser coche
```

**¿Por qué 448×448?**
- ✅ Compatible con MetaCLIP H/14 (336×336 nativo)
- ✅ Preserva más detalles que 224×224
- ✅ Balance óptimo: calidad vs. eficiencia computacional
- ✅ Permite multi-patch sin overhead excesivo

**Output Esperado por Imagen**:
- Imágenes grandes: 6-10 crops
- Imágenes medianas: 4-6 crops
- Imágenes pequeñas: 2-4 crops

**Total Estimado**: ~1,500-2,000 crops para 276 imágenes ok

---

### 3. Clustered Crop Generator (Optimizado) 🔧

**Ubicación**: `src/core/preprocessing/clustered_crop_generator_optimized.py`

**Cambios Críticos para Preservar Información**:

#### ❌ **ELIMINADO**: Padding Adaptativo con Color Gris

```python
# ❌ ANTES (causaba artifacts y pérdida de información)
canvas = np.full((448, 448, 3), [114, 114, 114], dtype=np.uint8)
canvas[y:y+h, x:x+w] = crop  # Insertar crop centrado
```

**Problemas del padding gris**:
- 🚫 Artifacts visuales confunden al modelo
- 🚫 Pierde contexto espacial real
- 🚫 Reduce área efectiva del vehículo
- 🚫 Embeddings de menor calidad

#### ✅ **NUEVO**: Resize Proporcional Sin Padding

```python
# ✅ AHORA (preserva toda la información real)
def generate_crop_optimized(self, bbox, image):
    """
    Genera crop sin padding artificial
    """
    x, y, w, h = bbox
    crop = image[y:y+h, x:x+w]
    
    # Opción 1: Si crop es más pequeño que 448×448
    if w <= 448 and h <= 448:
        # NO hacer padding, usar multi-patch en embedder
        return crop  # Retornar tamaño original
    
    # Opción 2: Si crop es más grande que 448×448
    else:
        # Resize proporcional manteniendo aspecto
        scale = min(448 / w, 448 / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        crop_resized = cv2.resize(
            crop, 
            (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4  # Mejor calidad
        )
        
        return crop_resized  # Retornar sin padding
```

**Ventajas**:
- ✅ Solo píxeles reales del vehículo
- ✅ Sin artifacts artificiales
- ✅ Mejor calidad visual para MetaCLIP
- ✅ Preserva toda la información de la imagen original

**Manejo de Crops Pequeños**:
Si crop < 448×448, se pasa directamente al embedder que usa **estrategia multi-patch** (ver sección MetaCLIP Embedder).

**Output Esperado**: ~850 crops para 276 imágenes ko

---

### 4. Damage Contextualizer 📝

**Ubicación**: `src/core/embeddings/damage_contextualizer.py`

**Función**: Generar descripciones textuales enriquecidas para cada crop.

#### Para Imágenes CON Daño

**Método**: `build_damage_context(metadata: Dict) -> str`

**Estructura**:
1. **Zona del vehículo** (del naming de archivo)
2. **Parte específica** (inferida con heurística o VLM)
3. **Tipos de daño** con descripción breve
4. **Relación espacial** entre daños

**Implementación**:

```python
def build_damage_context(self, metadata: Dict) -> str:
    """
    Genera contexto rico para crops con daño
    """
    zone_desc = metadata['zone_description']  # "rear_left_quarter"
    zone_area = metadata['zone_area']         # "posterior"
    
    # Inferir parte específica del vehículo
    specific_part = self.infer_specific_part(
        zone=zone_desc,
        bbox_center=metadata.get('bbox_center', (0.5, 0.5)),
        damage_types=metadata.get('damage_types', [])
    )
    # Ejemplo: "Rear left corner panel near bumper junction"
    
    # Describir tipos de daño
    from collections import Counter
    type_counts = Counter(metadata['damage_types'])
    
    damage_descriptions = []
    for dtype, count in type_counts.items():
        friendly = self.get_friendly_description(dtype)
        damage_descriptions.append(
            f"{dtype.replace('_', ' ')} ({friendly}, {count} instance{'s' if count > 1 else ''})"
        )
    
    # Analizar patrón espacial
    spatial_pattern = self.analyze_spatial_pattern(metadata['defects'])
    
    # Ensamblar contexto
    context = (
        f"Vehicle zone: {zone_desc} ({zone_area} area). "
        f"Affected part: {specific_part}. "
        f"Damage types: {', '.join(damage_descriptions)}. "
        f"Spatial pattern: {spatial_pattern}."
    )
    
    return context

def get_friendly_description(self, damage_type: str) -> str:
    """Descripciones amigables para tipos de daño"""
    descriptions = {
        'surface_scratch': 'minor surface abrasion',
        'dent': 'metal deformation',
        'paint_peeling': 'paint layer detachment',
        'deep_scratch': 'deep paint penetration',
        'crack': 'structural fracture',
        'missing_part': 'component absence',
        'missing_accessory': 'accessory detachment',
        'misaligned_part': 'panel misalignment'
    }
    return descriptions.get(damage_type, 'unspecified damage')

def analyze_spatial_pattern(self, defects: list) -> str:
    """Analiza patrón espacial entre defectos"""
    if len(defects) <= 1:
        return "isolated damage"
    
    # Calcular distancias entre defectos
    centers = [
        ((d['bbox'][0] + d['bbox'][2]) / 2, 
         (d['bbox'][1] + d['bbox'][3]) / 2)
        for d in defects
    ]
    
    # Heurística simple de clustering
    avg_distance = np.mean([
        np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
        for i in range(len(centers))
        for j in range(i+1, len(centers))
    ])
    
    if avg_distance < 100:  # Píxeles
        return "defects clustered together, suggesting single impact event"
    else:
        return "defects distributed across area, multiple impact points"

def infer_specific_part(
    self, 
    zone: str, 
    bbox_center: tuple,
    damage_types: list
) -> str:
    """
    Infiere parte específica del vehículo usando heurística
    
    Alternativa: Integrar VLM ligero como Florence-2 o PaliGemma
    """
    # Mapa zona → partes posibles
    part_mapping = {
        'front_left_fender': {
            'top': 'Upper front fender panel',
            'center': 'Front fender mid-section',
            'bottom': 'Front fender lower edge near wheel arch'
        },
        'hood_center': {
            'front': 'Front hood edge near grille',
            'center': 'Central hood panel',
            'rear': 'Hood rear section near windshield'
        },
        'rear_left_quarter': {
            'top': 'Upper rear quarter panel',
            'center': 'Rear left corner panel near bumper junction',
            'bottom': 'Lower rear quarter panel near wheel arch'
        },
        # ... (mapeo completo para todas las zonas)
    }
    
    # Determinar posición relativa
    cx, cy = bbox_center
    if cy < 0.33:
        position = 'top'
    elif cy < 0.67:
        position = 'center'
    else:
        position = 'bottom'
    
    parts = part_mapping.get(zone, {})
    return parts.get(position, f"{zone.replace('_', ' ')} panel")
```

**Ejemplo Output**:
```
Vehicle zone: rear_left_quarter (posterior area). 
Affected part: Rear left corner panel near bumper junction. 
Damage types: surface scratch (minor surface abrasion, 2 instances), 
              dent (metal deformation, 1 instance). 
Spatial pattern: defects clustered together, suggesting single impact event.
```

**Longitud**: ~150-200 caracteres

#### Para Imágenes SIN Daño

**Método**: `build_clean_context(metadata: Dict) -> str`

**Estructura**:
1. Zona del vehículo
2. Parte específica
3. Condición superficie (minimalista)

**Implementación**:

```python
def build_clean_context(self, metadata: Dict) -> str:
    """
    Contexto minimalista para imágenes limpias
    """
    zone_desc = metadata['zone_description']
    zone_area = metadata['zone_area']
    
    # Inferir parte específica usando posición del grid
    specific_part = self.infer_specific_part(
        zone=zone_desc,
        bbox_center=metadata.get('grid_center', (0.5, 0.5))
    )
    
    context = (
        f"Vehicle zone: {zone_desc} ({zone_area} area). "
        f"Inspected part: {specific_part}. "
        f"Surface condition: Clean paint, no scratches or dents detected. "
        f"Panel integrity: Normal alignment, intact surface."
    )
    
    return context
```

**Ejemplo Output**:
```
Vehicle zone: hood_center (frontal area). 
Inspected part: Central hood panel. 
Surface condition: Clean paint, no scratches or dents detected. 
Panel integrity: Normal alignment, intact surface.
```

**Longitud**: ~120-150 caracteres

---

### 5. MetaCLIP 2 Embedder (Unificado) 🧠

**Ubicación**: `src/core/embeddings/metaclip_embedder_unified.py`

**Función**: Generar embeddings unificados (visual + textual) usando MetaCLIP 2.

#### Ventajas de MetaCLIP 2 sobre DINOv3 + BERT

| Característica | DINOv3 + BERT | MetaCLIP 2 |
|----------------|---------------|------------|
| **Modelos** | 2 separados | ✅ **1 unificado** |
| **Alineación** | Manual (weights) | ✅ **Nativa end-to-end** |
| **Espacio embedding** | Concatenación forzada | ✅ **Compartido naturalmente** |
| **Dimensión** | 1408d (1024+384) | ✅ **1024d** (más eficiente) |
| **Velocidad** | 2 forward pass | ✅ **1 forward pass** |
| **VRAM** | ~8GB | ✅ **~6GB** |
| **Retrieval** | Bueno | ✅ **Superior** |
| **Mantenimiento** | Complejo | ✅ **Simple** |

#### Implementación Completa

```python
# src/core/embeddings/metaclip_embedder_unified.py

from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
from pathlib import Path

class MetaCLIPUnifiedEmbedder:
    """
    Embedder unificado con MetaCLIP 2 para dataset balanceado
    
    Características:
    - Visual + Textual en espacio compartido
    - Multi-patch para preservar detalles
    - Fusión optimizada (average o weighted)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/metaclip-h14-fullcc2.5b",
        use_multipatch: bool = True,
        patch_size: int = 336,
        patch_stride: int = 224,
        device: str = None
    ):
        """
        Args:
            model_name: MetaCLIP model
                - "facebook/metaclip-b32-400m" → 512d (rápido)
                - "facebook/metaclip-h14-fullcc2.5b" → 1024d ✅ (recomendado)
            use_multipatch: Si True, usa estrategia multi-patch para crops grandes
            patch_size: Tamaño de patch para multi-patch (336 recomendado)
            patch_stride: Stride para sliding window (224 → 33% overlap)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*70}")
        print(f"🔧 INICIALIZANDO MetaCLIP Unified Embedder")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Multi-patch: {use_multipatch}")
        if use_multipatch:
            print(f"Patch size: {patch_size}×{patch_size}")
            print(f"Patch stride: {patch_stride} (overlap: {(1 - patch_stride/patch_size)*100:.0f}%)")
        print(f"{'='*70}\n")
        
        # Cargar modelo
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Configuración
        self.embedding_dim = 1024 if "h14" in model_name else 512
        self.use_multipatch = use_multipatch
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        print(f"✅ Modelo cargado")
        print(f"   - Embedding dim: {self.embedding_dim}")
        print(f"   - Total params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M\n")
    
    def generate_unified_embedding(
        self,
        image_path: Path,
        text_description: str,
        fusion_strategy: str = "average"
    ) -> np.ndarray:
        """
        Genera embedding unificado (imagen + texto)
        
        Args:
            image_path: Ruta al crop
            text_description: Contexto textual rico
            fusion_strategy: 
                - "average": (img_emb + text_emb) / 2 → 1024d ✅ RECOMENDADO
                - "weighted": α*img + β*text → 1024d
                - "concat": [img | text] → 2048d (NO recomendado)
        
        Returns:
            Embedding normalizado (1024,) float32
        """
        image = Image.open(image_path).convert('RGB')
        W, H = image.size
        
        # Decidir si usar multi-patch
        if self.use_multipatch and (W > self.patch_size or H > self.patch_size):
            return self._generate_multipatch_embedding(
                image, text_description, fusion_strategy
            )
        else:
            return self._generate_single_embedding(
                image, text_description, fusion_strategy
            )
    
    def _generate_single_embedding(
        self,
        image: Image.Image,
        text_description: str,
        fusion_strategy: str
    ) -> np.ndarray:
        """Genera embedding de imagen única (crop pequeño)"""
        
        # Procesamiento conjunto
        inputs = self.processor(
            text=[text_description],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Mover a device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Embeddings alineados nativamente en espacio compartido
            image_emb = outputs.image_embeds  # [1, 1024]
            text_emb = outputs.text_embeds    # [1, 1024]
        
        # Fusión
        if fusion_strategy == "average":
            # Promedio simple → Balance perfecto
            combined = (image_emb + text_emb) / 2
        
        elif fusion_strategy == "weighted":
            # Fusión ponderada
            α = 0.5  # peso imagen
            β = 0.5  # peso texto
            combined = α * image_emb + β * text_emb
        
        elif fusion_strategy == "concat":
            # Concatenación → 2048d (NO recomendado)
            combined = torch.cat([image_emb, text_emb], dim=1)
        
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Normalizar L2
        combined = combined / combined.norm(dim=-1, keepdim=True)
        
        return combined.cpu().numpy().flatten().astype(np.float32)
    
    def _generate_multipatch_embedding(
        self,
        image: Image.Image,
        text_description: str,
        fusion_strategy: str
    ) -> np.ndarray:
        """
        Genera embedding multi-patch para preservar detalles
        
        Estrategia:
        1. Embedding global (imagen completa con texto)
        2. Embeddings de patches (solo visual)
        3. Fusión: 60% global + 40% patches
        """
        W, H = image.size
        
        # 1. Embedding global (imagen + texto)
        global_emb = self._generate_single_embedding(
            image, text_description, fusion_strategy
        )
        
        # 2. Generar patches con sliding window
        patches = []
        patch_coords = []
        
        for y in range(0, max(H - self.patch_size + 1, 1), self.patch_stride):
            for x in range(0, max(W - self.patch_size + 1, 1), self.patch_stride):
                # Extraer patch
                patch = image.crop((
                    x, y, 
                    min(x + self.patch_size, W), 
                    min(y + self.patch_size, H)
                ))
                
                # Si patch es más pequeño que patch_size, hacer resize
                if patch.size != (self.patch_size, self.patch_size):
                    patch = patch.resize(
                        (self.patch_size, self.patch_size),
                        Image.LANCZOS
                    )
                
                patches.append(patch)
                patch_coords.append((x, y))
        
        # Si no hay patches (imagen muy pequeña), retornar solo global
        if not patches:
            return global_emb
        
        # 3. Embeddings de patches (solo visual, sin texto)
        patch_embeddings = []
        
        for patch in patches:
            inputs = self.processor(images=patch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                patch_embeddings.append(outputs.image_embeds)
        
        # 4. Fusionar patches (promedio)
        patch_tensor = torch.cat(patch_embeddings, dim=0)  # [N_patches, 1024]
        fused_patches = torch.mean(patch_tensor, dim=0, keepdim=True)  # [1, 1024]
        
        # 5. Combinar global + patches
        # Global tiene contexto textual → más peso
        # Patches tienen detalles visuales → menos peso
        global_tensor = torch.from_numpy(global_emb).unsqueeze(0).to(self.device)
        combined = 0.6 * global_tensor + 0.4 * fused_patches
        
        # Normalizar
        combined = combined / combined.norm(dim=-1, keepdim=True)
        
        return combined.cpu().numpy().flatten().astype(np.float32)
    
    def generate_batch_embeddings(
        self,
        image_paths: List[Path],
        text_descriptions: List[str],
        batch_size: int = 8,
        fusion_strategy: str = "average",
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Genera embeddings en batch
        
        Returns:
            embeddings: np.ndarray (N, 1024)
            debug_info: List[dict] con info de cada embedding
        """
        assert len(image_paths) == len(text_descriptions), \
            "image_paths y text_descriptions deben tener mismo tamaño"
        
        n_samples = len(image_paths)
        all_embeddings = []
        debug_info = []
        
        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=n_samples, desc="Generating MetaCLIP embeddings")
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_paths = image_paths[i:batch_end]
            batch_texts = text_descriptions[i:batch_end]
            
            for img_path, text in zip(batch_paths, batch_texts):
                try:
                    emb = self.generate_unified_embedding(
                        image_path=img_path,
                        text_description=text,
                        fusion_strategy=fusion_strategy
                    )
                    
                    all_embeddings.append(emb)
                    
                    debug_info.append({
                        'image_path': str(img_path),
                        'text_description': text,
                        'embedding_norm': float(np.linalg.norm(emb)),
                        'fusion_strategy': fusion_strategy,
                        'multi_patch_used': self.use_multipatch
                    })
                
                except Exception as e:
                    print(f"\n❌ Error en {img_path.name}: {e}")
                    # Embedding cero si falla
                    all_embeddings.append(
                        np.zeros(self.embedding_dim, dtype=np.float32)
                    )
                    debug_info.append({
                        'error': str(e),
                        'image_path': str(img_path)
                    })
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        embeddings = np.vstack(all_embeddings)
        
        print(f"\n{'='*70}")
        print(f"✅ BATCH EMBEDDINGS GENERADOS")
        print(f"{'='*70}")
        print(f"Shape: {embeddings.shape}")
        print(f"Dtype: {embeddings.dtype}")
        print(f"Norma promedio: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
        print(f"{'='*70}\n")
        
        return embeddings, debug_info
    
    def get_model_info(self) -> Dict:
        """Retorna información del modelo"""
        return {
            'model_type': 'metaclip-2',
            'model_name': self.model.config.name_or_path,
            'embedding_dim': self.embedding_dim,
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'supports_multipatch': self.use_multipatch,
            'patch_size': self.patch_size if self.use_multipatch else None,
            'patch_stride': self.patch_stride if self.use_multipatch else None
        }
```

#### Comparativa de Estrategias de Fusión

| Estrategia | Dimensión | Pros | Contras | Uso |
|------------|-----------|------|---------|-----|
| **average** | 1024d | Balance perfecto, eficiente | - | ✅ **RECOMENDADO** |
| **weighted** | 1024d | Ajustable (α, β) | Requiere tuning | Alternativa |
| **concat** | 2048d | Preserva todo | Pesado para FAISS | ❌ NO usar |

**Recomendación**: Usar `fusion_strategy="average"` → **1024d**

---

### 6. Unified FAISS Index Builder 🗄️

**Ubicación**: `src/core/vector_store/unified_faiss_builder.py`

**Función**: Construir índice FAISS único con todos los crops (damaged + clean).

**Configuración Optimizada**:

```python
# Para ~2,500 vectores de 1024 dims: IndexHNSWFlat
index = faiss.IndexHNSWFlat(1024, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64
```

**Parámetros HNSW**:
- **M**: 32 (conectividad del grafo)
- **efConstruction**: 200 (calidad durante construcción)
- **efSearch**: 64 (calidad durante búsqueda)

**Tamaño Estimado**:
- ~2,500 vectores × 1024 dims × 4 bytes ≈ **10 MB**
- Con overhead HNSW: **~15-18 MB** ✅

**Ventaja vs. Arquitectura Anterior**:
- Antes: 1408 dims → ~20-25 MB
- Ahora: 1024 dims → **~15-18 MB** (28% reducción)

---

### 7. Unified Retriever con Filtros 🔍

**Ubicación**: `src/core/rag/retriever_unified.py`

**Función**: Búsqueda semántica con filtros pre-FAISS.

#### Filtros Soportados

| Filtro | Tipo | Descripción |
|--------|------|-------------|
| `vehicle_zone` | str o List[str] | Zona(s) del vehículo (1-10) |
| `has_damage` | bool | Con daño (True) o sin daño (False) |
| `damage_type` | str o List[str] | Tipo(s) de daño específico(s) |

#### Ejemplo de Uso

```python
# Buscar solo zona 4 con daño
results = retriever.search(
    query_embedding=query_emb,
    k=5,
    filters={
        'vehicle_zone': '4',
        'has_damage': True
    }
)

# Buscar imágenes limpias en zonas frontales
results = retriever.search(
    query_embedding=query_emb,
    k=10,
    filters={
        'vehicle_zone': ['1', '2', '3'],
        'has_damage': False
    }
)

# Buscar arañazos superficiales
results = retriever.search(
    query_embedding=query_emb,
    k=5,
    filters={
        'damage_type': 'surface_scratch'
    }
)
```

---

## 🚀 Pipeline de Implementación

### FASE 1: Generación de Crops Optimizados 📸

**Script**: `scripts/phase1_generate_crops_optimized.py`  
**Duración Estimada**: 15-20 minutos

#### Proceso

```
Dataset Split 2 (552 imágenes)
           ↓
    ¿Tipo de imagen?
     ↙         ↘
   ko          ok
(276)         (276)
    ↓            ↓
Clustered    Vehicle Detector
  Crop       + Grid Crop
Generator      Generator
(sin padding)  (448×448)
    ↓            ↓
~850 crops   ~1,500-2,000 crops
    ↓            ↓
damaged/     clean/
```

#### Implementación

```python
#!/usr/bin/env python3
# scripts/phase1_generate_crops_optimized.py

from pathlib import Path
from src.core.preprocessing.vehicle_detector import VehicleDetector
from src.core.preprocessing.grid_crop_generator import GridCropGenerator
from src.core.preprocessing.clustered_crop_generator_optimized import ClusteredCropGeneratorOptimized
import json
from tqdm import tqdm

def main():
    # Configuración
    dataset_dir = Path("data/raw/dataset_split_2")
    output_dir = Path("data/processed/crops/balanced_optimized")
    
    output_dir_damaged = output_dir / "damaged"
    output_dir_clean = output_dir / "clean"
    
    output_dir_damaged.mkdir(parents=True, exist_ok=True)
    output_dir_clean.mkdir(parents=True, exist_ok=True)
    
    # Inicializar componentes
    print("🔧 Inicializando componentes...")
    vehicle_detector = VehicleDetector()
    grid_generator = GridCropGenerator(
        crop_size=448,
        overlap=0.25,
        min_vehicle_ratio=0.70
    )
    cluster_generator = ClusteredCropGeneratorOptimized(
        target_size=448,
        use_padding=False  # ← SIN PADDING
    )
    
    # Escanear dataset
    all_images = list(dataset_dir.glob("*_imageDANO_original.jpg"))
    
    ko_images = [img for img in all_images if "_ko_" in img.name]
    ok_images = [img for img in all_images if "_ok_" in img.name]
    
    print(f"\n📊 Dataset:")
    print(f"   - Imágenes CON daño: {len(ko_images)}")
    print(f"   - Imágenes SIN daño: {len(ok_images)}")
    print(f"   - Total: {len(all_images)}\n")
    
    metadata_list = []
    
    # Procesar imágenes CON daño
    print("🔧 Procesando imágenes CON daño (clustered crops)...")
    for img_path in tqdm(ko_images, desc="Crops damaged"):
        # Buscar JSON de segmentación
        json_path = img_path.parent / img_path.name.replace(
            '_imageDANO_original.jpg',
            '_labelDANO_modificado.json'
        )
        
        if not json_path.exists():
            continue
        
        # Generar crops
        crops = cluster_generator.generate_crops(
            image_path=img_path,
            json_path=json_path
        )
        
        # Guardar crops
        for i, crop_data in enumerate(crops):
            crop_id = img_path.stem + f"_cluster_{i:03d}"
            crop_path = output_dir_damaged / f"{crop_id}.jpg"
            
            cv2.imwrite(str(crop_path), crop_data['crop'])
            
            # Metadata preliminar
            metadata_list.append({
                'crop_id': crop_id,
                'crop_path': str(crop_path),
                'source_image': img_path.name,
                'has_damage': True,
                'vehicle_zone': extract_zone_from_filename(img_path.name),
                'crop_type': 'clustered',
                'crop_size': crop_data['crop'].shape[:2],
                'defects': crop_data.get('defects', [])
            })
    
    # Procesar imágenes SIN daño
    print("\n🔧 Procesando imágenes SIN daño (grid crops)...")
    for img_path in tqdm(ok_images, desc="Crops clean"):
        # Detectar vehículo
        detection = vehicle_detector.detect(img_path)
        
        if detection is None:
            continue
        
        # Generar grid crops
        crops = grid_generator.generate_crops(
            image_path=img_path,
            vehicle_bbox=detection['bbox']
        )
        
        # Guardar crops
        for i, crop_data in enumerate(crops):
            crop_id = img_path.stem + f"_grid_{crop_data['grid_x']}_{crop_data['grid_y']}"
            crop_path = output_dir_clean / f"{crop_id}.jpg"
            
            cv2.imwrite(str(crop_path), crop_data['crop'])
            
            # Metadata preliminar
            metadata_list.append({
                'crop_id': crop_id,
                'crop_path': str(crop_path),
                'source_image': img_path.name,
                'has_damage': False,
                'vehicle_zone': extract_zone_from_filename(img_path.name),
                'crop_type': 'grid',
                'crop_size': crop_data['crop'].shape[:2],
                'grid_position': [crop_data['grid_x'], crop_data['grid_y']]
            })
    
    # Guardar metadata preliminar
    metadata_path = Path("data/processed/metadata/balanced_crops_preliminary.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ FASE 1 COMPLETADA")
    print(f"{'='*70}")
    print(f"Crops damaged: {len([m for m in metadata_list if m['has_damage']])}")
    print(f"Crops clean: {len([m for m in metadata_list if not m['has_damage']])}")
    print(f"Total crops: {len(metadata_list)}")
    print(f"Metadata guardada: {metadata_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
```

#### Output

```
data/processed/crops/balanced_optimized/
├── damaged/
│   ├── zona1_ko_..._cluster_000.jpg
│   ├── zona1_ko_..._cluster_001.jpg
│   └── ... (~850 crops, sin padding)
└── clean/
    ├── zona1_ok_..._grid_0_0.jpg
    ├── zona1_ok_..._grid_0_1.jpg
    └── ... (~1,500-2,000 crops)

data/processed/metadata/
└── balanced_crops_preliminary.json
```

#### Métricas de Éxito

- ✅ ~850 crops damaged (sin padding artificial)
- ✅ ~1,500-2,000 crops clean
- ✅ Todos los crops preservan información original
- ✅ Crops clean con >70% área del vehículo

---

### FASE 2: Generación de Contextos Enriquecidos 📝

**Script**: `scripts/phase2_generate_contexts.py`  
**Duración Estimada**: 20-30 minutos

#### Proceso

```
Metadata Preliminar
        ↓
  ¿Tiene daño?
   ↙         ↘
 SÍ          NO
  ↓            ↓
Cargar JSON  Contexto
segmentación  limpio
  ↓            ↓
Extraer tipos   ↓
+ relaciones    ↓
espaciales      ↓
  ↓            ↓
  └─────┬──────┘
        ↓
DamageContextualizer
        ↓
text_description
        ↓
Metadata Enriquecida
```

#### Implementación

```python
#!/usr/bin/env python3
# scripts/phase2_generate_contexts.py

from pathlib import Path
from src.core.embeddings.damage_contextualizer import DamageContextualizer
import json
from tqdm import tqdm

def main():
    # Cargar metadata preliminar
    metadata_path = Path("data/processed/metadata/balanced_crops_preliminary.json")
    
    with open(metadata_path) as f:
        metadata_list = json.load(f)
    
    print(f"📊 Metadata cargada: {len(metadata_list)} crops\n")
    
    # Inicializar contextualizador
    contextualizer = DamageContextualizer()
    
    # Enriquecer metadata
    print("📝 Generando contextos textuales...")
    
    for meta in tqdm(metadata_list, desc="Generating contexts"):
        if meta['has_damage']:
            # Cargar JSON de segmentación
            source_img = meta['source_image']
            json_path = Path("data/raw/dataset_split_2") / source_img.replace(
                '_imageDANO_original.jpg',
                '_labelDANO_modificado.json'
            )
            
            with open(json_path) as f:
                seg_data = json.load(f)
            
            # Extraer info de daños
            meta['damage_types'] = [
                shape['label'] for shape in seg_data['shapes']
                if shape['label'] != '9'
            ]
            
            # Generar contexto
            text_desc = contextualizer.build_damage_context(meta)
        
        else:
            # Generar contexto limpio
            text_desc = contextualizer.build_clean_context(meta)
        
        # Actualizar metadata
        meta['text_description'] = text_desc
    
    # Guardar metadata enriquecida
    output_path = Path("data/processed/metadata/balanced_crops_enriched.json")
    
    with open(output_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ FASE 2 COMPLETADA")
    print(f"{'='*70}")
    print(f"Metadata enriquecida guardada: {output_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
```

#### Output

```
data/processed/metadata/
└── balanced_crops_enriched.json  # Con text_description
```

#### Métricas de Éxito

- ✅ 100% crops con `text_description`
- ✅ Longitud promedio: 150-180 caracteres
- ✅ Contextos coherentes y descriptivos

---

### FASE 3: Generación de Embeddings MetaCLIP 🧠

**Script**: `scripts/phase3_generate_metaclip_embeddings.py`  
**Duración Estimada**: 25-30 minutos

#### Proceso

```
Metadata Enriquecida
        ↓
MetaCLIP 2 Embedder
        ↓
    ┌───┴───┐
    ↓       ↓
Visual   Text
(aligned natively)
    ↓       ↓
    └───┬───┘
        ↓
Fusion (average)
        ↓
Normalización L2
        ↓
Embeddings 1024d
```

#### Implementación

```python
#!/usr/bin/env python3
# scripts/phase3_generate_metaclip_embeddings.py

from pathlib import Path
from src.core.embeddings.metaclip_embedder_unified import MetaCLIPUnifiedEmbedder
import json
import numpy as np
from datetime import datetime

def main():
    # Cargar metadata enriquecida
    metadata_path = Path("data/processed/metadata/balanced_crops_enriched.json")
    
    with open(metadata_path) as f:
        metadata_list = json.load(f)
    
    print(f"📊 Metadata cargada: {len(metadata_list)} crops\n")
    
    # Inicializar embedder
    embedder = MetaCLIPUnifiedEmbedder(
        model_name="facebook/metaclip-h14-fullcc2.5b",
        use_multipatch=True,
        patch_size=336,
        patch_stride=224
    )
    
    # Preparar inputs
    image_paths = [Path(m['crop_path']) for m in metadata_list]
    text_descriptions = [m['text_description'] for m in metadata_list]
    
    # Generar embeddings
    embeddings, debug_info = embedder.generate_batch_embeddings(
        image_paths=image_paths,
        text_descriptions=text_descriptions,
        batch_size=8,
        fusion_strategy="average",
        show_progress=True
    )
    
    # Enriquecer metadata final
    for i, meta in enumerate(metadata_list):
        meta['embedding_index'] = i
        meta['embedding_model'] = 'metaclip-h14-fullcc2.5b'
        meta['embedding_dim'] = int(embeddings.shape[1])
        meta['embedding_norm'] = float(np.linalg.norm(embeddings[i]))
        meta['fusion_strategy'] = 'average'
    
    # Guardar
    output_dir = Path("data/processed/embeddings/metaclip_balanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Embeddings
    embeddings_path = output_dir / "embeddings_metaclip.npy"
    np.save(embeddings_path, embeddings)
    print(f"💾 Embeddings guardados: {embeddings_path}")
    
    # Metadata final
    metadata_final_path = output_dir / "metadata_final.json"
    with open(metadata_final_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    print(f"💾 Metadata final guardada: {metadata_final_path}")
    
    # Info del proceso
    process_info = {
        'timestamp': datetime.now().isoformat(),
        'model': embedder.get_model_info(),
        'dataset': {
            'total_crops': len(metadata_list),
            'damage_crops': sum(1 for m in metadata_list if m['has_damage']),
            'clean_crops': sum(1 for m in metadata_list if not m['has_damage'])
        },
        'embeddings': {
            'shape': list(embeddings.shape),
            'dtype': str(embeddings.dtype),
            'norm_mean': float(np.linalg.norm(embeddings, axis=1).mean()),
            'norm_std': float(np.linalg.norm(embeddings, axis=1).std())
        }
    }
    
    info_path = output_dir / "generation_info.json"
    with open(info_path, 'w') as f:
        json.dump(process_info, f, indent=2)
    print(f"💾 Info del proceso: {info_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ FASE 3 COMPLETADA")
    print(f"{'='*70}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Norma promedio: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
```

#### Output

```
data/processed/embeddings/metaclip_balanced/
├── embeddings_metaclip.npy      # (2500, 1024) float32
├── metadata_final.json           # Con embedding_index
└── generation_info.json          # Info del proceso
```

#### Estadísticas Esperadas

```
Shape: (2500, 1024)
Dtype: float32
Norma promedio: 1.0000 (±0.0001)
Tiempo total: ~25-30 minutos
Tiempo/crop: ~0.6-0.7s
```

#### Métricas de Éxito

- ✅ Embeddings shape: (N, 1024)
- ✅ Normas promedio: ~1.0
- ✅ Sin NaN o Inf
- ✅ 28% más eficiente que 1408d

---

### FASE 4: Construcción Índice FAISS 🗄️

**Script**: `scripts/phase4_build_unified_faiss_index.py`  
**Duración Estimada**: 1-2 minutos

#### Implementación

```python
#!/usr/bin/env python3
# scripts/phase4_build_unified_faiss_index.py

from pathlib import Path
import json
import numpy as np
import faiss
import pickle

def main():
    # Cargar embeddings y metadata
    embeddings_path = Path("data/processed/embeddings/metaclip_balanced/embeddings_metaclip.npy")
    metadata_path = Path("data/processed/embeddings/metaclip_balanced/metadata_final.json")
    
    embeddings = np.load(embeddings_path).astype('float32')
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    n_vectors, dim = embeddings.shape
    
    print(f"\n{'='*70}")
    print(f"🏗️  CONSTRUCCIÓN ÍNDICE FAISS")
    print(f"{'='*70}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Dimensión: {dim}")
    print(f"Metadata: {len(metadata)} entries\n")
    
    # Estadísticas
    damage_count = sum(1 for m in metadata if m['has_damage'])
    clean_count = len(metadata) - damage_count
    
    print(f"📊 Dataset:")
    print(f"   - CON daño: {damage_count} ({damage_count/n_vectors*100:.1f}%)")
    print(f"   - SIN daño: {clean_count} ({clean_count/n_vectors*100:.1f}%)\n")
    
    # Construir índice HNSW
    M = 32
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    
    print(f"🔧 Índice: IndexHNSWFlat")
    print(f"   - M: {M}")
    print(f"   - efConstruction: 200")
    print(f"   - efSearch: 64\n")
    
    # Añadir vectores
    index.add(embeddings)
    print(f"✅ Vectores añadidos: {index.ntotal}\n")
    
    # Validación
    distances, indices = index.search(embeddings[0:1], k=5)
    
    print(f"🔍 Validación (self-similarity):")
    print(f"   - Top-1 index: {indices[0][0]} (esperado: 0)")
    print(f"   - Top-1 distance: {distances[0][0]:.4f} (esperado: ~0.0)")
    
    assert indices[0][0] == 0, "Error: primer resultado no es él mismo"
    assert distances[0][0] < 0.01, "Error: distancia a sí mismo > 0.01"
    print(f"   ✅ Validación exitosa\n")
    
    # Guardar
    output_dir = Path("outputs/vector_indices/metaclip_balanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Índice FAISS
    index_path = output_dir / "index_metaclip_balanced.faiss"
    faiss.write_index(index, str(index_path))
    print(f"💾 Índice FAISS: {index_path}")
    
    # Metadata (pickle)
    metadata_pkl_path = output_dir / "metadata_balanced.pkl"
    with open(metadata_pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"💾 Metadata (pickle): {metadata_pkl_path}")
    
    # Configuración
    config = {
        'index_type': 'IndexHNSWFlat',
        'embedding_model': 'metaclip-h14-fullcc2.5b',
        'embedding_dim': dim,
        'fusion_strategy': 'average',
        'n_vectors': n_vectors,
        'M': M,
        'efConstruction': 200,
        'efSearch': 64,
        'damage_crops': damage_count,
        'clean_crops': clean_count,
        'index_size_mb': index_path.stat().st_size / (1024 * 1024)
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Config: {config_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ FASE 4 COMPLETADA")
    print(f"{'='*70}")
    print(f"Índice: {index.ntotal} vectores")
    print(f"Tamaño: {config['index_size_mb']:.2f} MB")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
```

#### Output

```
outputs/vector_indices/metaclip_balanced/
├── index_metaclip_balanced.faiss  # ~15-18 MB
├── metadata_balanced.pkl           # ~15-20 MB
└── config.json                     # Configuración
```

#### Métricas de Éxito

- ✅ Índice construido: 2,500 vectores
- ✅ Tamaño: ~15-18 MB (28% más eficiente que 1408d)
- ✅ Validación exitosa

---

### FASE 5: Validación y Testing ✅

**Script**: `scripts/phase5_validate_retrieval.py`  
**Duración Estimada**: 5-10 minutos

#### Tests Principales

1. **Filtros básicos** (zona, has_damage)
2. **Cobertura de retrieval** (damaged → damaged, clean → clean)
3. **Calidad de contextos** (text_description coherente)
4. **Visualización manual** (top-K resultados)

#### Implementación

```python
#!/usr/bin/env python3
# scripts/phase5_validate_retrieval.py

from pathlib import Path
from src.core.rag.retriever_unified import MetaCLIPUnifiedRetriever
import numpy as np
import random

def main():
    # Cargar retriever
    index_path = Path("outputs/vector_indices/metaclip_balanced/index_metaclip_balanced.faiss")
    metadata_path = Path("outputs/vector_indices/metaclip_balanced/metadata_balanced.pkl")
    
    retriever = MetaCLIPUnifiedRetriever(
        index_path=str(index_path),
        metadata_path=str(metadata_path)
    )
    
    # Cargar embeddings para testing
    embeddings = np.load("data/processed/embeddings/metaclip_balanced/embeddings_metaclip.npy")
    
    print(f"\n{'='*70}")
    print(f"🧪 VALIDACIÓN DEL RETRIEVER")
    print(f"{'='*70}\n")
    
    # Test 1: Filtros básicos
    print("📋 TEST 1: Filtros Básicos\n")
    
    # Test 1.1: Zona 4 con daño
    test_emb = embeddings[0]
    results = retriever.search(
        query_embedding=test_emb,
        k=5,
        filters={'vehicle_zone': '4', 'has_damage': True}
    )
    
    print("Test 1.1: Zona 4 con daño")
    for r in results:
        assert r['vehicle_zone'] == '4'
        assert r['has_damage'] == True
        print(f"   ✅ {r['crop_id']} - Zone: {r['zone_description']}")
    
    # Test 1.2: Zonas frontales sin daño
    results = retriever.search(
        query_embedding=test_emb,
        k=5,
        filters={'vehicle_zone': ['1', '2', '3'], 'has_damage': False}
    )
    
    print("\nTest 1.2: Zonas frontales sin daño")
    for r in results:
        assert r['vehicle_zone'] in ['1', '2', '3']
        assert r['has_damage'] == False
        print(f"   ✅ {r['crop_id']} - Zone: {r['zone_description']}")
    
    # Test 2: Cobertura
    print(f"\n{'='*70}")
    print("📋 TEST 2: Cobertura de Retrieval\n")
    
    # Query damaged → recuperar damaged
    damaged_indices = [i for i, m in enumerate(retriever.metadata) if m['has_damage']]
    sample_damaged = random.sample(damaged_indices, min(10, len(damaged_indices)))
    
    damage_coverage = []
    for idx in sample_damaged:
        query_emb = embeddings[idx]
        results = retriever.search(query_emb, k=5)
        
        damaged_count = sum(1 for r in results if r['has_damage'])
        damage_coverage.append(damaged_count / 5)
    
    avg_damage_coverage = np.mean(damage_coverage)
    print(f"Query damaged → Damaged retrieved: {avg_damage_coverage*100:.1f}%")
    assert avg_damage_coverage >= 0.6, "Cobertura de damaged < 60%"
    print(f"   ✅ Cobertura adecuada (≥60%)")
    
    # Query clean → recuperar clean
    clean_indices = [i for i, m in enumerate(retriever.metadata) if not m['has_damage']]
    sample_clean = random.sample(clean_indices, min(10, len(clean_indices)))
    
    clean_coverage = []
    for idx in sample_clean:
        query_emb = embeddings[idx]
        results = retriever.search(query_emb, k=5)
        
        clean_count = sum(1 for r in results if not r['has_damage'])
        clean_coverage.append(clean_count / 5)
    
    avg_clean_coverage = np.mean(clean_coverage)
    print(f"Query clean → Clean retrieved: {avg_clean_coverage*100:.1f}%")
    assert avg_clean_coverage >= 0.6, "Cobertura de clean < 60%"
    print(f"   ✅ Cobertura adecuada (≥60%)")
    
    print(f"\n{'='*70}")
    print(f"✅ TODAS LAS VALIDACIONES EXITOSAS")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
```

#### Métricas de Éxito

- ✅ Filtros funcionan correctamente
- ✅ Cobertura damaged ≥ 60%
- ✅ Cobertura clean ≥ 60%
- ✅ Contextos coherentes

---

## 📁 Estructura de Archivos

```
RAG-multimodal/
│
├── data/
│   ├── raw/
│   │   └── dataset_split_2/                    # Dataset original (552 imágenes)
│   │
│   └── processed/
│       ├── crops/
│       │   └── balanced_optimized/
│       │       ├── damaged/                    # ~850 crops (sin padding)
│       │       └── clean/                      # ~1,500-2,000 crops
│       │
│       ├── metadata/
│       │   ├── balanced_crops_preliminary.json
│       │   └── balanced_crops_enriched.json
│       │
│       └── embeddings/
│           └── metaclip_balanced/
│               ├── embeddings_metaclip.npy     # (2500, 1024)
│               ├── metadata_final.json
│               └── generation_info.json
│
├── src/
│   └── core/
│       ├── preprocessing/
│       │   ├── vehicle_detector.py
│       │   ├── grid_crop_generator.py
│       │   └── clustered_crop_generator_optimized.py  # SIN padding
│       │
│       ├── embeddings/
│       │   ├── damage_contextualizer.py
│       │   └── metaclip_embedder_unified.py   # ✨ NUEVO
│       │
│       ├── vector_store/
│       │   └── unified_faiss_builder.py
│       │
│       └── rag/
│           └── retriever_unified.py
│
├── scripts/
│   ├── phase1_generate_crops_optimized.py     # ✨ ACTUALIZADO
│   ├── phase2_generate_contexts.py
│   ├── phase3_generate_metaclip_embeddings.py # ✨ NUEVO
│   ├── phase4_build_unified_faiss_index.py    # ✨ ACTUALIZADO
│   └── phase5_validate_retrieval.py
│
├── outputs/
│   └── vector_indices/
│       └── metaclip_balanced/
│           ├── index_metaclip_balanced.faiss  # ~15-18 MB
│           ├── metadata_balanced.pkl
│           └── config.json
│
├── config/
│   └── crop_strategy_config.yaml
│
├── docs/
│   └── BALANCED_DATASET_ARCHITECTURE_METACLIP2.md  # 📄 ESTE DOCUMENTO
│
└── requirements.txt
```

---

## 📊 Esquema de Metadata

### Metadata Final (JSON)

```json
{
  "crop_id": "zona1_ko_2_3_1554817134014_zona_4_cluster_003",
  "crop_path": "data/processed/crops/balanced_optimized/damaged/zona1_ko_..._cluster_003.jpg",
  "source_image": "zona1_ko_2_3_1554817134014_zona_4_imageDANO_original.jpg",
  
  "has_damage": true,
  "vehicle_zone": "4",
  "zone_description": "rear_left_quarter",
  "zone_area": "posterior",
  
  "specific_part": "Rear left corner panel near bumper junction",
  "text_description": "Vehicle zone: rear_left_quarter (posterior area). Affected part: Rear left corner panel near bumper junction. Damage types: surface scratch (minor surface abrasion, 2 instances), dent (metal deformation, 1 instance). Spatial pattern: defects clustered together, suggesting single impact event.",
  
  "damage_types": ["surface_scratch", "dent"],
  "damage_count": 3,
  
  "crop_type": "clustered",
  "crop_size": [380, 420],
  
  "embedding_index": 42,
  "embedding_model": "metaclip-h14-fullcc2.5b",
  "embedding_dim": 1024,
  "embedding_norm": 1.0000,
  "fusion_strategy": "average"
}
```

---

## ⚙️ Configuración y Parámetros

### Parámetros Clave

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **Crop Size** | 448×448 | Balance calidad/eficiencia |
| **MetaCLIP Patch** | 336×336 | Resolución nativa H/14 |
| **Multi-patch Stride** | 224 | 33% overlap |
| **Grid Overlap** | 25% | Cobertura completa |
| **Min Vehicle Ratio** | 70% | Minimizar fondo |
| **Fusion Strategy** | average | Balance perfecto |
| **Embedding Dim** | 1024 | Espacio unificado |
| **FAISS Index** | IndexHNSWFlat | Óptimo <10K |
| **HNSW M** | 32 | Conectividad |
| **efConstruction** | 200 | Calidad construcción |
| **efSearch** | 64 | Calidad búsqueda |

---

## 📈 Resultados Esperados

### Comparativa Final: DINOv3+BERT vs. MetaCLIP 2

| Métrica | DINOv3+BERT (50/50) | MetaCLIP 2 (average) | Mejora |
|---------|---------------------|----------------------|--------|
| **Dimensión** | 1408d | 1024d | ✅ 27% reducción |
| **Índice FAISS** | ~20-25 MB | ~15-18 MB | ✅ 28% reducción |
| **Velocidad embedding** | ~0.8s/crop | ~0.6s/crop | ✅ 25% más rápido |
| **VRAM** | ~8GB | ~6GB | ✅ 25% reducción |
| **Alineación** | Manual | Nativa | ✅ Superior |
| **Retrieval esperado** | Bueno | Excelente | ✅ +10-15% |
| **Mantenimiento** | Complejo | Simple | ✅ 1 modelo |

### Estadísticas del Pipeline

| Fase | Input | Output | Tiempo |
|------|-------|--------|--------|
| Fase 1 | 552 imágenes | ~2,500 crops | 15-20 min |
| Fase 2 | 2,500 crops | Contextos | 20-30 min |
| Fase 3 | 2,500 crops | Embeddings 1024d | 25-30 min |
| Fase 4 | 2,500 embeddings | Índice FAISS | 1-2 min |
| Fase 5 | Índice | Validación | 5-10 min |
| **TOTAL** | - | - | **~1.5 horas** |

---

## 🚀 Próximos Pasos

### Implementación Inmediata

1. ✅ **Fase 1**: Generar crops optimizados (sin padding)
2. ✅ **Fase 2**: Contextos enriquecidos
3. ✅ **Fase 3**: Embeddings MetaCLIP 2
4. ✅ **Fase 4**: Índice FAISS unificado
5. ✅ **Fase 5**: Validación

### Mejoras Futuras

#### Corto Plazo
- ☐ Experimentar con `fusion_strategy="weighted"` (α/β ajustables)
- ☐ Agregar más filtros (severidad, área afectada)
- ☐ Implementar re-ranking con cross-attention
- ☐ A/B testing vs. arquitectura anterior

#### Medio Plazo
- ☐ Fine-tuning de MetaCLIP en dataset vehicular
- ☐ Integrar VLM ligero para `specific_part` (Florence-2, PaliGemma)
- ☐ Hard negative mining
- ☐ Explorar MetaCLIP 2 Worldwide (multilenguaje)

#### Largo Plazo
- ☐ Multi-modal fusion con attention weights dinámicos
- ☐ Integrar SAM para segmentación automática
- ☐ Sistema de active learning
- ☐ Deploy en producción con API REST

---

## 📚 Referencias

### Papers Científicos

1. **MetaCLIP**: "Demystifying CLIP Data" (2023)
   - https://arxiv.org/abs/2309.16671

2. **MetaCLIP 2**: "A Worldwide Scaling Recipe" (2024)
   - https://arxiv.org/abs/2507.22062

3. **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision" (2021)
   - https://arxiv.org/abs/2103.00020

4. **HNSW**: "Efficient and robust approximate nearest neighbor search" (2018)
   - https://arxiv.org/abs/1603.09320

5. **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
   - https://arxiv.org/abs/2005.11401

### Herramientas

- **FAISS**: https://github.com/facebookresearch/faiss
- **Transformers**: https://huggingface.co/docs/transformers
- **MetaCLIP Models**: https://huggingface.co/facebook
- **YOLOv8**: https://docs.ultralytics.com/

---

## 📝 Changelog

### v4.0 (MetaCLIP 2 - Current)
- ✨ Migración completa a MetaCLIP 2
- ✨ Embeddings unificados 1024d (espacio compartido)
- ✨ Multi-patch strategy para preservar detalles
- ✨ Eliminado padding artificial
- ✨ Fusión nativa (average strategy)
- 📉 27% reducción dimensionalidad
- 📉 28% reducción tamaño índice
- ⚡ 25% más rápido
- 💾 25% menos VRAM

### v3.0 (Balanced Hybrid)
- ✨ Dataset balanceado 50/50
- ✨ Grid crops para clean
- ✨ Contexto enriquecido
- ✨ Embeddings 50/50 (DINOv3+BERT)

### v2.0 (Hybrid Fullimages)
- ✨ Embeddings híbridos 60/40
- ✨ Full images
- ✨ Contexto básico

### v1.0 (Crops Only)
- ✨ Crops clusterizados
- ✨ DINOv3 puro
- ✨ Índice básico

---

**Última actualización**: Noviembre 2024  
**Versión del documento**: 4.0 (MetaCLIP 2)  
**Estado**: Listo para implementación  
**Autor**: [Tu nombre/equipo]

---

## 🎯 Resumen Ejecutivo

Este documento describe una arquitectura RAG multimodal optimizada que:

1. **Preserva información máxima** mediante crops sin padding y estrategia multi-patch
2. **Unifica visual + textual** nativamente con MetaCLIP 2 (1024d)
3. **Balancea dataset** 50/50 (damaged/clean) con contextos enriquecidos
4. **Reduce complejidad** 27% (dimensión), 28% (índice), 25% (VRAM)
5. **Mejora retrieval** mediante alineación nativa end-to-end

**Resultado esperado**: Sistema RAG de alta calidad, eficiente y escalable para detección de defectos vehiculares.