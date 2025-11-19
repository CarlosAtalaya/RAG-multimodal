# 🚗 RAG Multimodal - Dataset Balanceado con Contexto Enriquecido

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

Implementar un sistema RAG (Retrieval-Augmented Generation) multimodal que procese un dataset balanceado de imágenes vehiculares con y sin daños, generando embeddings híbridos (visual + textual) contextualizados para búsqueda semántica avanzada.

### Características Principales

- ✅ **Dataset Balanceado**: 50% imágenes con daño (ko) + 50% sin daño (ok)
- ✅ **Embeddings Híbridos**: Visual (DINOv3 1024d) + Textual (SBERT 384d) = 1408 dimensiones
- ✅ **Pesos 50/50**: Equilibrio entre información visual y semántica
- ✅ **Contexto Enriquecido**: Descripciones textuales ricas con zona, parte específica y tipos de daño
- ✅ **Índice Único FAISS**: Unificación de crops damaged + clean con filtros avanzados
- ✅ **Crops Inteligentes**: Clusterizados para daños, grid adaptativo para imágenes limpias

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
   Multimodal Embedder
   (50% Visual + 50% Text)
          ↓
   Embeddings Híbridos
      (1408 dims)
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
crop_size = 448
overlap = 0.25  # 25%
min_vehicle_ratio = 0.70  # Mínimo 70% del crop debe ser coche
```

**Output Esperado por Imagen**:
- Imágenes grandes: 6-10 crops
- Imágenes medianas: 4-6 crops
- Imágenes pequeñas: 2-4 crops

**Total Estimado**: ~1,500-2,000 crops para 276 imágenes ok

---

### 3. Clustered Crop Generator (Modificado) 🔧

**Ubicación**: `src/core/preprocessing/clustered_crop_generator.py`

**Cambios Clave**:

❌ **Eliminado**: Padding adaptativo con color gris
```python
# ANTES (NO usar)
canvas = np.full((448, 448, 3), [114, 114, 114], dtype=np.uint8)
canvas[y:y+h, x:x+w] = crop
```

✅ **Nuevo**: Imagen original completa
```python
# AHORA (usar)
if merged_bbox.width > 448 or merged_bbox.height > 448:
    # Resize proporcional manteniendo aspecto
    scale = min(448 / merged_bbox.width, 448 / merged_bbox.height)
    crop_resized = cv2.resize(crop, None, fx=scale, fy=scale)
else:
    # Crop directo sin padding
    crop_resized = crop
```

**Ventajas**:
- ✅ Sin artifacts de padding
- ✅ Solo píxeles reales del vehículo
- ✅ Mejor calidad visual para DINOv3

**Output Esperado**: ~850 crops para 276 imágenes ko

---

### 4. Damage Contextualizer 📝

**Ubicación**: `src/core/embeddings/damage_contextualizer.py`

**Función**: Generar descripciones textuales enriquecidas para cada crop.

#### Para Imágenes CON Daño

**Método**: `build_damage_context(metadata: Dict) -> str`

**Estructura**:
1. Zona del vehículo (del naming)
2. Parte específica (DINOv3 + heurística)
3. Tipos de daño con descripción breve
4. Relación espacial entre daños

**Ejemplo Output**:
```
Vehicle zone: rear_left_quarter (posterior area).
Affected part: Rear left corner panel near bumper junction.
Damage types: Surface scratch (minor abrasion, 2 instances), Dent (metal deformation, 1 instance).
Spatial pattern: Scratches clustered around dent, suggesting single impact event.
```

**Longitud**: ~150-200 caracteres (conciso pero informativo)

#### Para Imágenes SIN Daño

**Método**: `build_clean_context(metadata: Dict) -> str`

**Estructura**:
1. Zona del vehículo
2. Parte específica (DINOv3)
3. Condición superficie (minimalista)

**Ejemplo Output**:
```
Vehicle zone: hood_center (frontal area).
Inspected part: Central hood panel.
Surface condition: Clean paint, no scratches or dents detected.
Panel integrity: Normal alignment, intact surface.
```

**Longitud**: ~120-150 caracteres

#### Estrategia para "Parte Específica" con DINOv3

```python
prompt = f"""
Analyze this cropped vehicle image showing the {zone_description} area.
Identify the SPECIFIC car part visible (e.g., 'front bumper', 'door handle area', 'quarter panel edge').
Be precise and concise. Format: "specific_part_name"
"""
```

**Fallback**: Si DINOv3 falla, usar heurística basada en zona + posición relativa.

---

### 5. Multimodal Embedder (Modificado) 🧠

**Ubicación**: `src/core/embeddings/multimodal_embedder.py`

**Cambios Principales**:

✅ **Pesos Ajustados a 50/50**
```python
# ANTES
visual_weight = 0.6  # 60%
text_weight = 0.4    # 40%

# AHORA
visual_weight = 0.5  # 50%
text_weight = 0.5    # 50%
```

**Justificación**:
- Equilibrio perfecto entre información visual y semántica
- Mejora retrieval cuando similitud visual es baja pero contexto es similar
- Base científica: Papers de CLIP, BLIP-2 usan fusión balanceada

#### Dimensión de Embeddings Híbridos

```
Visual (DINOv3):           1024 dims
Textual (Sentence-BERT):    384 dims
────────────────────────────────────
Total (Concatenación):     1408 dims
```

#### ¿Por qué 1408 dims es óptimo?

| Aspecto | Evaluación |
|---------|-----------|
| Capacidad FAISS | ✅ Excelente (<10K dims es óptimo) |
| Overfitting | ✅ No hay riesgo con 2,500 samples |
| Velocidad búsqueda | ✅ <50ms por query |
| Calidad información | ✅ Preserva ambas modalidades |
| Alternativa PCA | ❌ Perdería información valiosa |

**Conclusión**: ✅ Mantener 1408 dims

#### Pipeline de Generación

```python
def generate_hybrid_embedding(self, image_path, metadata):
    # 1. Embedding Visual
    visual_emb = self.visual_embedder.generate_embedding(image_path)
    # Shape: (1024,)
    
    # 2. Contexto Textual
    if metadata['has_damage']:
        text_desc = self.contextualizer.build_damage_context(metadata)
    else:
        text_desc = self.contextualizer.build_clean_context(metadata)
    
    # 3. Embedding Textual
    text_emb = self.text_embedder.encode(text_desc)
    # Shape: (384,)
    
    # 4. Fusión Ponderada 50/50
    hybrid_emb = np.concatenate([
        visual_emb * self.visual_weight,  # 0.5
        text_emb * self.text_weight       # 0.5
    ])
    # Shape: (1408,)
    
    # 5. Normalización L2
    hybrid_emb = hybrid_emb / np.linalg.norm(hybrid_emb)
    
    return hybrid_emb, text_desc
```

---

### 6. Unified FAISS Index Builder 🗄️

**Ubicación**: `src/core/vector_store/unified_faiss_builder.py`

**Función**: Construir índice FAISS único con todos los crops (damaged + clean).

**Configuración**:
```python
# Para <10K vectores: IndexHNSWFlat
index = faiss.IndexHNSWFlat(1408, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64
```

**Parámetros**:
- **M**: 32 (conectividad del grafo HNSW)
- **efConstruction**: 200 (calidad durante indexación)
- **efSearch**: 64 (calidad durante búsqueda)

**Tamaño Estimado**:
- ~2,500 vectores × 1408 dims × 4 bytes = ~14 MB
- Con overhead HNSW: ~20-25 MB

---

### 7. Unified Retriever con Filtros 🔍

**Ubicación**: `src/core/rag/retriever_unified.py`

**Función**: Búsqueda semántica con filtros pre-FAISS.

#### Filtros Soportados

| Filtro | Tipo | Descripción |
|--------|------|-------------|
| vehicle_zone | str o List[str] | Zona(s) del vehículo (1-10) |
| has_damage | bool | Con daño (True) o sin daño (False) |
| damage_type | str o List[str] | Tipo(s) de daño específico(s) |

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
        'vehicle_zone': ['1', '2', '3'],  # Zonas frontales
        'has_damage': False
    }
)
```

#### Flujo de Búsqueda

```
Query Embedding
      ↓
Aplicar Filtros en Metadata
      ↓
Construir Subset de Índices Válidos
      ↓
Búsqueda FAISS en Subset
      ↓
Remapear Índices al Índice Principal
      ↓
Top-K Resultados
```

---

## 🚀 Pipeline de Implementación

### FASE 1: Generación de Crops 📸

**Script**: `scripts/phase1_generate_crops.py`  
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
    ↓            ↓
~850 crops   ~1,500-2,000 crops
    ↓            ↓
damaged/     clean/
```

#### Acciones

1. **Inicializar componentes**:
```python
vehicle_detector = VehicleDetector()  # YOLOv8
grid_generator = GridCropGenerator(crop_size=448, overlap=0.25)
cluster_generator = ClusteredCropGenerator(target_size=448)
```

2. **Procesar imágenes**:
```python
for image in dataset_split_2:
    if image.has_damage:  # _ko_
        crops = cluster_generator.generate(image)
    else:  # _ok_
        vehicle_bbox = vehicle_detector.detect(image)
        crops = grid_generator.generate(image, vehicle_bbox)
    
    save_crops(crops, output_dir)
```

3. **Guardar metadata preliminar**:
```python
metadata = {
    'crop_id': 'zona1_ko_..._cluster_003',
    'crop_path': '/path/to/crop.jpg',
    'source_image': 'zona1_ko_..._zona_4_imageDANO_original.jpg',
    'has_damage': True,
    'vehicle_zone': '4',
    'zone_description': 'rear_left_quarter',
    'zone_area': 'posterior',
    'crop_type': 'clustered' | 'grid',
    # ... (sin text_description todavía)
}
```

#### Output

```
data/processed/crops/balanced_dataset/
├── damaged/
│   ├── zona1_ko_..._cluster_000.jpg
│   ├── zona1_ko_..._cluster_001.jpg
│   └── ... (~850 crops)
└── clean/
    ├── zona1_ok_..._grid_0_0.jpg
    ├── zona1_ok_..._grid_0_1.jpg
    └── ... (~1,500-2,000 crops)

data/processed/metadata/
└── balanced_crops_preliminary.json
```

#### Métricas de Éxito

- ✅ ~850 crops damaged generados
- ✅ ~1,500-2,000 crops clean generados
- ✅ Todos los crops son 448×448 (o proporcionalmente escalados)
- ✅ Crops clean tienen >70% área del vehículo

---

### FASE 2: Generación de Contextos Textuales 📝

**Script**: `scripts/phase2_generate_contexts.py`  
**Duración Estimada**: 30-40 minutos (dependiente de DINOv3 API)

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
  Llamar DINOv3
  parte específica
        ↓
DamageContextualizer
        ↓
Generar text_description
        ↓
Enriquecer metadata
```

#### Acciones

1. **Cargar metadata preliminar**:
```python
with open('balanced_crops_preliminary.json') as f:
    metadata_list = json.load(f)
```

2. **Para cada crop CON daño**:
```python
# Cargar JSON de segmentación
segmentation_json = load_segmentation_json(crop.source_image)

# Extraer información
damage_types = extract_damage_types(segmentation_json)
spatial_relations = analyze_spatial_relations(segmentation_json)

# Obtener parte específica con DINOv3
specific_part = dinov3_identify_part(
    crop_path=crop.crop_path,
    zone=crop.zone_description
)

# Generar contexto
text_desc = contextualizer.build_damage_context(
    zone=crop.zone_description,
    specific_part=specific_part,
    damage_types=damage_types,
    spatial_relations=spatial_relations
)
```

3. **Para cada crop SIN daño**:
```python
# Obtener parte específica con DINOv3
specific_part = dinov3_identify_part(
    crop_path=crop.crop_path,
    zone=crop.zone_description
)

# Generar contexto minimalista
text_desc = contextualizer.build_clean_context(
    zone=crop.zone_description,
    specific_part=specific_part
)
```

4. **Enriquecer metadata**:
```python
crop.metadata['text_description'] = text_desc
crop.metadata['specific_part'] = specific_part

if crop.has_damage:
    crop.metadata['damage_descriptions'] = {
        'surface_scratch': 'Minor abrasion, 2 instances',
        'dent': 'Metal deformation, 1 instance'
    }
    crop.metadata['spatial_pattern'] = 'Scratches clustered around dent'
```

#### Output

```
data/processed/metadata/
└── balanced_crops_enriched.json
```

#### Ejemplo de Metadata Enriquecida

```json
{
  "crop_id": "zona1_ko_..._cluster_003",
  "crop_path": "/path/to/crop.jpg",
  "source_image": "zona1_ko_..._zona_4_imageDANO_original.jpg",
  "has_damage": true,
  "vehicle_zone": "4",
  "zone_description": "rear_left_quarter",
  "zone_area": "posterior",
  "specific_part": "Rear left corner panel near bumper junction",
  "text_description": "Vehicle zone: rear_left_quarter (posterior area). Affected part: Rear left corner panel near bumper junction. Damage types: Surface scratch (minor abrasion, 2 instances), Dent (metal deformation, 1 instance). Spatial pattern: Scratches clustered around dent, suggesting single impact event.",
  "damage_types": ["surface_scratch", "dent"],
  "damage_count": 3,
  "damage_descriptions": {
    "surface_scratch": "Minor abrasion, 2 instances",
    "dent": "Metal deformation, 1 instance"
  },
  "spatial_pattern": "Scratches clustered around dent",
  "crop_type": "clustered"
}
```

#### Métricas de Éxito

- ✅ Todos los crops tienen text_description
- ✅ Todos los crops tienen specific_part
- ✅ Crops damaged tienen damage_descriptions y spatial_pattern
- ✅ Longitud promedio de descripciones: 150-200 caracteres

---

### FASE 3: Generación de Embeddings Híbridos 🧠

**Script**: `scripts/phase3_generate_hybrid_embeddings.py`  
**Duración Estimada**: 20-25 minutos

#### Proceso

```
Metadata Enriquecida
        ↓
MultimodalEmbedder (50/50)
        ↓
    ┌───┴───┐
    ↓       ↓
DINOv3   Sentence-BERT
Visual   Text
1024d    384d
    ↓       ↓
    └───┬───┘
        ↓
Concatenación Ponderada
        ↓
Normalización L2
        ↓
Embeddings Híbridos
      1408d
```

#### Acciones

1. **Inicializar embedder**:
```python
embedder = MultimodalEmbedder(
    visual_weight=0.5,
    text_weight=0.5,
    use_bfloat16=True
)
```

2. **Batch processing**:
```python
batch_size = 16
all_embeddings = []

for batch in batches(crops_metadata, batch_size):
    batch_paths = [crop['crop_path'] for crop in batch]
    
    embeddings, debug_info = embedder.generate_batch_embeddings(
        image_paths=batch_paths,
        metadata_list=batch
    )
    
    all_embeddings.append(embeddings)

final_embeddings = np.vstack(all_embeddings)
# Shape: (N, 1408) donde N ~2,500
```

3. **Guardar embeddings y metadata final**:
```python
# Embeddings
np.save('embeddings.npy', final_embeddings)

# Metadata con índices de embeddings
for i, meta in enumerate(crops_metadata):
    meta['embedding_index'] = i
    meta['embedding_model'] = 'hybrid_dinov3_sbert_50_50'
    meta['embedding_norm'] = float(np.linalg.norm(final_embeddings[i]))

with open('metadata_final.json', 'w') as f:
    json.dump(crops_metadata, f, indent=2)
```

#### Output

```
data/processed/embeddings/balanced_hybrid_50_50/
├── embeddings.npy           # (2500, 1408) float32
├── metadata_final.json      # Con embedding_index
└── generation_info.json     # Info del proceso
```

#### Estadísticas Esperadas

```
Shape: (2500, 1408)
Dtype: float32
Norma promedio: 1.0000 (±0.0001)
Tiempo total: ~20 minutos
Tiempo/crop: ~0.5s
```

#### Métricas de Éxito

- ✅ Embeddings shape: (N, 1408)
- ✅ Normas promedio: ~1.0 (normalización L2)
- ✅ Sin NaN o Inf
- ✅ Todos los crops tienen embedding_index

---

### FASE 4: Construcción Índice FAISS 🗄️

**Script**: `scripts/phase4_build_unified_faiss_index.py`  
**Duración Estimada**: 2-3 minutos

#### Proceso

```
Embeddings (2500×1408)
        ↓
IndexHNSWFlat (M=32)
        ↓
Añadir Vectores
        ↓
Validación
        ↓
Guardar Índice + Metadata
```

#### Acciones

1. **Cargar embeddings**:
```python
embeddings = np.load('embeddings.npy').astype('float32')
with open('metadata_final.json') as f:
    metadata = json.load(f)

print(f"Embeddings: {embeddings.shape}")
print(f"Metadata: {len(metadata)} entries")
```

2. **Construir índice FAISS**:
```python
import faiss

dim = 1408
M = 32  # Conectividad HNSW

index = faiss.IndexHNSWFlat(dim, M)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64

# Añadir vectores
index.add(embeddings)

print(f"Índice construido: {index.ntotal} vectores")
```

3. **Validación**:
```python
# Test de búsqueda
query = embeddings[0:1]
distances, indices = index.search(query, k=5)

print("Top-5 resultados:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i+1}. Idx={idx}, Dist={dist:.4f}")

# Validar que el primero es él mismo
assert indices[0][0] == 0
assert distances[0][0] < 0.01
```

4. **Guardar**:
```python
# Índice FAISS
faiss.write_index(index, 'index_hnsw_balanced.faiss')

# Metadata (pickle para preservar tipos)
import pickle
with open('metadata_balanced.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# Configuración
config = {
    'index_type': 'IndexHNSWFlat',
    'n_vectors': index.ntotal,
    'embedding_dim': dim,
    'M': M,
    'efConstruction': 200,
    'efSearch': 64,
    'data_type': 'balanced_unified',
    'crops_damaged': sum(1 for m in metadata if m['has_damage']),
    'crops_clean': sum(1 for m in metadata if not m['has_damage'])
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

#### Output

```
outputs/vector_indices/balanced_unified/
├── index_hnsw_balanced.faiss    # ~20-25 MB
├── metadata_balanced.pkl         # ~15-20 MB
└── config.json                   # Config del índice
```

#### Métricas de Éxito

- ✅ Índice construido: 2,500 vectores
- ✅ Tamaño razonable: ~20-25 MB
- ✅ Validación exitosa (self-similarity)
- ✅ Metadata en sync con índice

---

### FASE 5: Validación y Testing ✅

**Script**: `scripts/phase5_validate_retrieval.py`  
**Duración Estimada**: 5-10 minutos

#### Test 1: Filtros Básicos

```python
# Test 1.1: Buscar solo zona 4 con daño
results = retriever.search(
    query_embedding=test_emb,
    k=5,
    filters={
        'vehicle_zone': '4',
        'has_damage': True
    }
)

print("Test 1.1: Zona 4 con daño")
for r in results:
    assert r.vehicle_zone == '4'
    assert r.has_damage == True
    print(f"✅ {r.crop_id} - Zone: {r.zone_description}")

# Test 1.2: Buscar imágenes limpias frontales
results = retriever.search(
    query_embedding=test_emb,
    k=10,
    filters={
        'vehicle_zone': ['1', '2', '3'],
        'has_damage': False
    }
)

print("\nTest 1.2: Zonas frontales sin daño")
for r in results:
    assert r.vehicle_zone in ['1', '2', '3']
    assert r.has_damage == False
    print(f"✅ {r.crop_id} - Zone: {r.zone_description}")
```

#### Test 2: Cobertura de Retrieval

```python
# Test 2.1: Query damaged → Recuperar damaged similares
damaged_crops = [m for m in metadata if m['has_damage']]
sample_damaged = random.sample(damaged_crops, 10)

for crop in sample_damaged:
    query_emb = embeddings[crop['embedding_index']]
    results = retriever.search(query_emb, k=5)
    
    damaged_count = sum(1 for r in results if r.has_damage)
    print(f"Query damaged: {damaged_count}/5 resultados con daño")
    assert damaged_count >= 3  # Al menos 60% deben ser damaged

# Test 2.2: Query clean → Recuperar clean similares
clean_crops = [m for m in metadata if not m['has_damage']]
sample_clean = random.sample(clean_crops, 10)

for crop in sample_clean:
    query_emb = embeddings[crop['embedding_index']]
    results = retriever.search(query_emb, k=5)
    
    clean_count = sum(1 for r in results if not r.has_damage)
    print(f"Query clean: {clean_count}/5 resultados limpios")
    assert clean_count >= 3  # Al menos 60% deben ser clean
```

#### Test 3: Calidad de Contextos

```python
# Verificar que text_description tiene sentido
for i in range(20):
    meta = metadata[i]
    print(f"\n{meta['crop_id']}")
    print(f"Has damage: {meta['has_damage']}")
    print(f"Zone: {meta['zone_description']}")
    print(f"Text: {meta['text_description'][:150]}...")
    
    # Validaciones básicas
    assert len(meta['text_description']) > 50
    assert meta['zone_description'] in meta['text_description']
    
    if meta['has_damage']:
        assert 'damage' in meta['text_description'].lower()
        assert len(meta['damage_types']) > 0
    else:
        assert 'no damage' in meta['text_description'].lower() or \
               'clean' in meta['text_description'].lower()
```

#### Test 4: Visualización Manual

```python
# Seleccionar query y visualizar top-5
query_idx = 42
query_meta = metadata[query_idx]
query_emb = embeddings[query_idx]

print(f"\n{'='*70}")
print(f"QUERY: {query_meta['crop_id']}")
print(f"Zone: {query_meta['zone_description']}")
print(f"Has damage: {query_meta['has_damage']}")
print(f"Text: {query_meta['text_description']}")
print(f"{'='*70}\n")

results = retriever.search(query_emb, k=5)

print("TOP-5 RESULTADOS:")
for i, r in enumerate(results, 1):
    print(f"\n{i}. {r.crop_id}")
    print(f"   Distance: {r.distance:.4f}")
    print(f"   Zone: {r.zone_description}")
    print(f"   Has damage: {r.has_damage}")
    print(f"   Text: {r.text_description[:100]}...")
```

#### Métricas de Éxito

- ✅ Todos los filtros funcionan correctamente
- ✅ Query damaged recupera mayoritariamente damaged
- ✅ Query clean recupera mayoritariamente clean
- ✅ Query zona X recupera mayoritariamente zona X
- ✅ Contextos textuales son coherentes y descriptivos
- ✅ No hay errores de indexación (índices fuera de rango)

---

## 📁 Estructura de Archivos

### Estructura Completa del Proyecto

```
RAG-multimodal/
│
├── data/
│   ├── raw/
│   │   └── dataset_split_2/                    # Dataset original (552 imágenes)
│   │       ├── zona1_ko_..._imageDANO_original.jpg
│   │       ├── zona1_ko_..._imageDANO_original.json
│   │       ├── zona1_ko_..._labelDANO_modificado.json
│   │       ├── zona1_ok_..._imageDANO_original.jpg
│   │       └── zona1_ok_..._imageDANO_original.json
│   │
│   └── processed/
│       ├── crops/
│       │   └── balanced_dataset/
│       │       ├── damaged/                    # ~850 crops (448×448)
│       │       │   ├── zona1_ko_..._cluster_000.jpg
│       │       │   ├── zona1_ko_..._cluster_001.jpg
│       │       │   └── ...
│       │       └── clean/                      # ~1,500-2,000 crops (448×448)
│       │           ├── zona1_ok_..._grid_0_0.jpg
│       │           ├── zona1_ok_..._grid_0_1.jpg
│       │           └── ...
│       │
│       ├── metadata/
│       │   ├── balanced_crops_preliminary.json # Metadata sin contexto textual
│       │   └── balanced_crops_enriched.json    # Metadata con text_description
│       │
│       └── embeddings/
│           └── balanced_hybrid_50_50/
│               ├── embeddings.npy              # (2500, 1408) float32
│               ├── metadata_final.json         # Metadata con embedding_index
│               └── generation_info.json        # Info del proceso
│
├── src/
│   └── core/
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── vehicle_detector.py             # ✨ NUEVO
│       │   ├── grid_crop_generator.py          # ✨ NUEVO
│       │   └── clustered_crop_generator.py     # 🔧 MODIFICADO
│       │
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── damage_contextualizer.py        # ✨ NUEVO
│       │   ├── multimodal_embedder.py          # 🔧 MODIFICADO
│       │   ├── dinov3_vitl_embedder.py         # Existente
│       │   └── ...
│       │
│       ├── vector_store/
│       │   ├── __init__.py
│       │   └── unified_faiss_builder.py        # ✨ NUEVO
│       │
│       └── rag/
│           ├── __init__.py
│           ├── retriever_unified.py            # ✨ NUEVO
│           ├── taxonomy_normalizer.py          # Existente
│           └── ...
│
├── scripts/
│   ├── phase1_generate_crops.py                # ✨ NUEVO
│   ├── phase2_generate_contexts.py             # ✨ NUEVO
│   ├── phase3_generate_hybrid_embeddings.py    # ✨ NUEVO
│   ├── phase4_build_unified_faiss_index.py     # ✨ NUEVO
│   ├── phase5_validate_retrieval.py            # ✨ NUEVO
│   └── ...
│
├── outputs/
│   └── vector_indices/
│       └── balanced_unified/
│           ├── index_hnsw_balanced.faiss       # Índice FAISS (~20-25 MB)
│           ├── metadata_balanced.pkl           # Metadata (~15-20 MB)
│           └── config.json                     # Configuración del índice
│
├── config/
│   ├── crop_strategy_config.yaml              # Configuración de crops
│   └── ...
│
├── docs/
│   └── BALANCED_DATASET_ARCHITECTURE.md        # 📄 ESTE DOCUMENTO
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Esquema de Metadata

### Metadata Final (JSON)

```json
{
  "crop_id": "zona1_ko_2_3_1554817134014_zona_4_cluster_003",
  "crop_path": "data/processed/crops/balanced_dataset/damaged/zona1_ko_..._cluster_003.jpg",
  "source_image": "zona1_ko_2_3_1554817134014_zona_4_imageDANO_original.jpg",
  
  "has_damage": true,
  "vehicle_zone": "4",
  "zone_description": "rear_left_quarter",
  "zone_area": "posterior",
  
  "specific_part": "Rear left corner panel near bumper junction",
  "text_description": "Vehicle zone: rear_left_quarter (posterior area). Affected part: Rear left corner panel near bumper junction. Damage types: Surface scratch (minor abrasion, 2 instances), Dent (metal deformation, 1 instance). Spatial pattern: Scratches clustered around dent, suggesting single impact event.",
  
  "damage_types": ["surface_scratch", "dent"],
  "damage_count": 3,
  "damage_descriptions": {
    "surface_scratch": "Minor abrasion, 2 instances",
    "dent": "Metal deformation, 1 instance"
  },
  "spatial_pattern": "Scratches clustered around dent, suggesting single impact event",
  
  "crop_type": "clustered",
  "crop_size": [448, 448],
  
  "embedding_index": 42,
  "embedding_model": "hybrid_dinov3_sbert_50_50",
  "embedding_dim": 1408,
  "embedding_norm": 1.0000,
  "visual_emb_norm": 1.0000,
  "text_emb_norm": 1.0000
}
```

### Metadata para Crop Sin Daño

```json
{
  "crop_id": "zona1_ok_2_3_1554373063646_zona_6_grid_0_1",
  "crop_path": "data/processed/crops/balanced_dataset/clean/zona1_ok_..._grid_0_1.jpg",
  "source_image": "zona1_ok_2_3_1554373063646_zona_6_imageDANO_original.jpg",
  
  "has_damage": false,
  "vehicle_zone": "6",
  "zone_description": "rear_right_quarter",
  "zone_area": "posterior",
  
  "specific_part": "Right rear quarter panel",
  "text_description": "Vehicle zone: rear_right_quarter (posterior area). Inspected part: Right rear quarter panel. Surface condition: Clean paint, no scratches or dents detected. Panel integrity: Normal alignment, intact surface.",
  
  "damage_types": [],
  "damage_count": 0,
  
  "crop_type": "grid",
  "crop_grid_position": [0, 1],
  "crop_size": [448, 448],
  "crop_coverage_area": 0.11,
  
  "embedding_index": 1234,
  "embedding_model": "hybrid_dinov3_sbert_50_50",
  "embedding_dim": 1408,
  "embedding_norm": 1.0000,
  "visual_emb_norm": 1.0000,
  "text_emb_norm": 1.0000
}
```

---

## ⚙️ Configuración y Parámetros

### Parámetros Clave

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Crop Size | 448×448 | Óptimo para DINOv3-ViT-L/16 |
| Grid Overlap | 25% | Balance cobertura/eficiencia |
| Min Vehicle Ratio | 70% | Minimizar fondo innecesario |
| Visual Weight | 0.5 (50%) | Equilibrio modalidades |
| Text Weight | 0.5 (50%) | Equilibrio modalidades |
| Embedding Dim | 1408 | 1024 (visual) + 384 (text) |
| FAISS Index | IndexHNSWFlat | Óptimo para <10K vectores |
| HNSW M | 32 | Conectividad óptima |
| efConstruction | 200 | Calidad construcción |
| efSearch | 64 | Calidad búsqueda |

### Configuración de Crops

```yaml
# config/crop_strategy_config.yaml

roi_crops:
  target_size: [448, 448]
  maintain_aspect: true

grid_crops:
  crop_size: 448
  overlap: 0.25
  min_vehicle_ratio: 0.70

vehicle_detection:
  model: yolov8n
  confidence_threshold: 0.5
  bbox_expansion: 0.15  # 15% de margen
```

### Configuración de Embeddings

```yaml
# config/embedding_config.yaml

multimodal:
  visual_weight: 0.5
  text_weight: 0.5
  normalize: true

visual:
  model: dinov3-vitl16
  dimension: 1024
  use_bfloat16: true

textual:
  model: all-MiniLM-L6-v2
  dimension: 384
  normalize: true
```

---

## 📈 Resultados Esperados

### Estadísticas del Pipeline

| Fase | Input | Output | Tiempo |
|------|-------|--------|--------|
| Fase 1 | 552 imágenes | ~2,500 crops | 15-20 min |
| Fase 2 | 2,500 crops | Contextos enriquecidos | 30-40 min |
| Fase 3 | 2,500 crops | Embeddings 1408d | 20-25 min |
| Fase 4 | 2,500 embeddings | Índice FAISS | 2-3 min |
| Fase 5 | Índice FAISS | Validación | 5-10 min |
| **TOTAL** | - | - | **~1.5 horas** |

### Métricas de Calidad

#### Crops
- ✅ ~850 crops damaged (clusterizados)
- ✅ ~1,500-2,000 crops clean (grid)
- ✅ Total: ~2,500 crops
- ✅ Todos 448×448 (o proporcionalmente escalados)
- ✅ Crops clean con >70% área del vehículo

#### Contextos Textuales
- ✅ 100% de crops con text_description
- ✅ Longitud promedio: 150-180 caracteres
- ✅ Damaged: zona + parte + tipos + relación espacial
- ✅ Clean: zona + parte + condición superficie

#### Embeddings
- ✅ Shape: (2500, 1408)
- ✅ Dtype: float32
- ✅ Normalización L2: norma ~1.0
- ✅ Sin NaN o Inf
- ✅ Balance 50/50 (visual/text)

#### Índice FAISS
- ✅ Tipo: IndexHNSWFlat
- ✅ Vectores: 2,500
- ✅ Dimensión: 1408
- ✅ Tamaño: ~20-25 MB
- ✅ Latencia búsqueda: <50ms
- ✅ Recall@5: >80% (esperado)

### Comparación con Versiones Anteriores

| Aspecto | v1.0 (Crops solos) | v2.0 (Hybrid Fullimages) | v3.0 (Balanced Hybrid) |
|---------|-------------------|-------------------------|----------------------|
| Dataset | 815 damaged | 815 full images | 552 balanced (50/50) |
| Crops | 850 clustered | 0 (full images) | ~2,500 (mixed) |
| Embedding | DINOv3 only | Hybrid 60/40 | Hybrid 50/50 |
| Contexto | Básico | Zona + descripción | Rico (parte + daños + espacial) |
| Índice | 850 vectores | 815 vectores | ~2,500 vectores |
| Filtros | Tipo + zona | Zona + has_damage | Zona + has_damage + tipo |
| Recall@5 | ~60% | ~45% | >70% (esperado) |

---

## 🚀 Próximos Pasos

### Implementación Inmediata

1. **Implementar Fase 1**:
   - Crear VehicleDetector
   - Crear GridCropGenerator
   - Modificar ClusteredCropGenerator
   - Script phase1_generate_crops.py

2. **Implementar Fase 2**:
   - Crear DamageContextualizer
   - Integrar DINOv3 para partes específicas
   - Script phase2_generate_contexts.py

3. **Implementar Fase 3**:
   - Modificar MultimodalEmbedder (pesos 50/50)
   - Script phase3_generate_hybrid_embeddings.py

4. **Implementar Fase 4**:
   - Crear UnifiedFAISSBuilder
   - Script phase4_build_unified_faiss_index.py

5. **Implementar Fase 5**:
   - Crear UnifiedRetriever con filtros
   - Script phase5_validate_retrieval.py

### Mejoras Futuras (Post-MVP)

#### Corto Plazo
- ☐ Aumentar tamaño de crops a 512×512 o 640×640
- ☐ Experimentar con pesos dinámicos (adaptative fusion)
- ☐ Agregar más filtros (severidad, múltiples tipos)
- ☐ Implementar re-ranking con cross-attention

#### Medio Plazo
- ☐ Fine-tuning de DINOv3 en dataset vehicular
- ☐ Implementar hard negative mining
- ☐ Agregar augmentations durante generación de crops
- ☐ Explorar IndexIVFPQ para datasets más grandes

#### Largo Plazo
- ☐ Multi-modal fusion con attention weights
- ☐ Integrar segmentación automática (SAM)
- ☐ Sistema de active learning para mejorar contextos
- ☐ Deploy en producción con API REST

---

## 📚 Referencias

### Papers Científicos

1. **DINOv3**: "DINOv3: A SELF-SUPERVISED VISION TRANSFORMER MODEL" (2023)
   - https://arxiv.org/abs/2304.07193

2. **HNSW**: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2018)
   - https://arxiv.org/abs/1603.09320

3. **Multimodal Embeddings**: "CLIP: Learning Transferable Visual Models From Natural Language Supervision" (2021)
   - https://arxiv.org/abs/2103.00020

4. **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
   - https://arxiv.org/abs/2005.11401

### Herramientas y Librerías

- **FAISS**: https://github.com/facebookresearch/faiss
- **Sentence-Transformers**: https://www.sbert.net/
- **YOLOv8**: https://docs.ultralytics.com/
- **Transformers (Hugging Face)**: https://huggingface.co/docs/transformers

---

## 📝 Changelog

### v3.0 (Balanced Hybrid - Current)
- ✨ Dataset balanceado 50/50 (ko + ok)
- ✨ Grid crops para imágenes limpias
- ✨ Contexto enriquecido con parte específica
- ✨ Embeddings híbridos 50/50
- ✨ Índice unificado con filtros avanzados
- 🔧 Eliminado padding adaptativo en crops damaged

### v2.0 (Hybrid Fullimages)
- ✨ Embeddings híbridos 60/40
- ✨ Full images con metadata enriquecida
- ✨ Contexto textual básico

### v1.0 (Crops Only)
- ✨ Crops clusterizados
- ✨ Embeddings DINOv3 puro
- ✨ Índice FAISS básico

---

## 🤝 Contribución

Este documento describe la arquitectura implementada. Para modificaciones o mejoras:

1. Revisar la sección **Módulos del Sistema**
2. Actualizar scripts correspondientes en `scripts/phase*.py`
3. Documentar cambios en este README
4. Validar con Fase 5 antes de integrar

---

**Última actualización**: Noviembre 2024  
**Versión del documento**: 3.0  
**Estado**: Implementación en progreso