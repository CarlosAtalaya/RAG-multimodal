# 🤖 PROMPT PARA NUEVA CONVERSACIÓN - PROYECTO RAG MULTIMODAL

## 📋 CONTEXTO DEL PROYECTO

Estoy desarrollando un **sistema RAG (Retrieval-Augmented Generation) multimodal** para detección y análisis de defectos en vehículos usando Vision-Language Models (VLM).

### 🎯 Objetivo Principal
Construir un sistema que:
1. Procesa imágenes de vehículos con defectos etiquetados (polígonos)
2. Genera embeddings visuales de los defectos usando Qwen3-VL
3. Indexa vectores en FAISS para búsqueda de similitud
4. Utiliza RAG para responder preguntas sobre daños vehiculares con contexto de ejemplos similares

### 🛠️ Stack Tecnológico
- **VLM**: Qwen3-VL-4B-Instruct (via API Docker en `localhost:8001`)
- **Embeddings**: Estrategia híbrida (Qwen3-VL + Sentence-Transformers)
- **Vector DB**: FAISS (IndexHNSWFlat para POC)
- **Lenguaje**: Python 3.12
- **Dataset**: 100 imágenes POC (~2,143 crops de defectos)

---

## 📂 DOCUMENTOS DE CONTEXTO ESENCIALES

Por favor, lee estos documentos en el siguiente orden para entender el proyecto:

### 1️⃣ **Diseño Técnico Completo** (CRÍTICO)
```
docs/RAG_MULTIMODAL_TECHNICAL_DESIGN.md
```
- Decisiones de arquitectura fundamentadas científicamente
- Estrategia de embeddings jerárquicos
- Configuración FAISS por escala
- Taxonomía de daños normalizada
- Referencias bibliográficas

### 2️⃣ **Plan de Implementación POC** (CRÍTICO)
```
docs/POC_IMPLEMENTATION_PLAN.md
```
- Roadmap de 7 días implementación
- Scripts de cada fase con código completo
- Outputs esperados de cada etapa
- Checklist de progreso

### 3️⃣ **Guía de Integración API** (IMPORTANTE)
```
docs/API_integration_guide.md
```
- Explicación de las 3 opciones de embeddings
- Estrategia híbrida actual (Opción B)
- Modificaciones futuras para embeddings nativos (Opción A)
- Tests de integración

### 4️⃣ **README del Proyecto** (REFERENCIA)
```
README.md
```
- Overview general
- Instrucciones de instalación
- Quick start guide

---

## 📊 ESTADO ACTUAL DEL PROYECTO

### ✅ **COMPLETADO** (Fases 1-2)

#### FASE 1: Preparación Dataset ✅
- [x] Selección de 100 imágenes balanceadas
- [x] Generación de `poc_manifest.json`
- [x] Estadísticas del dataset:
  - 100 imágenes
  - 2,155 defectos totales
  - Distribución por tipos de daño validada

#### FASE 2: Generación de Crops ✅
- [x] Implementado `AdaptiveCropGenerator` con:
  - Padding adaptativo por tipo de daño y tamaño
  - Umbrales dinámicos basados en percentiles (P99 × 1.5)
  - Metadata completa con contexto espacial
- [x] Resultados obtenidos:
  - **2,143 crops generados** (99.4% aprovechamiento)
  - 12 crops descartados (extremos anómalos)
  - Metadata enriquecida guardada en:
    ```
    data/processed/metadata/crops_metadata.json
    ```

**Distribución de Crops Generados:**
```
surface_scratch: 1,911 (89.2%)
dent: 77 (3.6%)
crack: 30 (1.4%)
missing_part: 29 (1.4%)
missing_accessory: 29 (1.4%)
paint_peeling: 23 (1.1%)
misaligned_part: 22 (1.0%)
deep_scratch: 22 (1.0%)
```

**Distribución Espacial:**
```
middle_center: 658 crops (30.7%)
bottom_center: 520 crops (24.3%)
middle_left: 312 crops (14.6%)
middle_right: 271 crops (12.6%)
bottom_right: 195 crops (9.1%)
bottom_left: 180 crops (8.4%)
top_center: 4 crops (0.2%)
top_right: 2 crops (0.1%)
top_left: 1 crops (0.0%)
```

**Tamaños Relativos:**
```
very_small: 2,018 crops (94.2%)
small: 103 crops (4.8%)
medium: 22 crops (1.0%)
```

---

### ⏳ **EN PROGRESO** (Fase 3)

#### FASE 3: Generación de Embeddings 🔄
- [x] Implementado `HybridEmbedder`:
  - Qwen3-VL genera descripciones visuales
  - Sentence-BERT (all-MiniLM-L6-v2) convierte a embeddings
  - Dimensión: 384
  - Retry logic con fallback
- [x] Script `03_generate_embeddings.py` preparado
- [ ] **PRÓXIMO PASO INMEDIATO**: Ejecutar generación de embeddings
  - Comando: `python scripts/03_generate_embeddings.py`
  - Tiempo estimado: 15-20 minutos
  - Output esperado: `data/processed/embeddings/embeddings.npy` (2143 × 384)

---

### ⏹️ **PENDIENTE** (Fases 4-7)

#### FASE 4: Construcción Índice FAISS
- [ ] Script `04_build_faiss_index.py` (preparar durante ejecución de fase 3)
- [ ] Construcción IndexHNSWFlat
- [ ] Validación búsqueda k-NN

#### FASE 5: RAG Retriever
- [ ] Implementar `DamageRAGRetriever`
- [ ] Sistema de filtros y búsqueda
- [ ] Construcción de contexto RAG

#### FASE 6: Análisis Completo con RAG
- [ ] Implementar `RAGDamageAnalyzer`
- [ ] Integración completa con API Qwen3-VL
- [ ] Pipeline end-to-end

#### FASE 7: Evaluación y Métricas
- [ ] Métricas de retrieval (Recall@k, Precision@k)
- [ ] Casos de prueba
- [ ] Documentación de resultados

---

## 🗂️ ESTRUCTURA DEL PROYECTO ACTUAL

```
RAG-multimodal/
├── data/
│   ├── raw/
│   │   └── 100_samples/              # Dataset POC (100 imágenes + JSONs)
│   │       └── poc_manifest.json     # Manifest con estadísticas
│   └── processed/
│       ├── crops/
│       │   └── roi/                  # 2,143 crops generados ✅
│       ├── metadata/
│       │   └── crops_metadata.json   # Metadata completa ✅
│       └── embeddings/               # (vacío - siguiente paso)
│
├── src/
│   ├── core/
│   │   ├── preprocessing/
│   │   │   └── crop_generator.py     # ✅ AdaptiveCropGenerator v2
│   │   ├── embeddings/
│   │   │   └── hybrid_embedder.py    # ✅ HybridEmbedder
│   │   ├── rag/                      # (vacío - fase 5)
│   │   └── vector_store/             # (vacío - fase 4)
│   └── utils/
│
├── scripts/
│   ├── 01_prepare_dataset.py         # ✅ Usado
│   ├── 02_generate_crops.py          # ✅ Usado
│   ├── 03_generate_embeddings.py     # ⏳ PRÓXIMO
│   ├── 04_build_faiss_index.py       # ⏹️ Pendiente
│   └── 05_run_inference.py           # ⏹️ Pendiente
│
├── config/
│   └── crop_strategy_config.yaml     # ✅ Configuración crops
│
├── docs/
│   ├── RAG_MULTIMODAL_TECHNICAL_DESIGN.md
│   ├── POC_IMPLEMENTATION_PLAN.md
│   ├── API_integration_guide.md
│   └── notes/
│
├── outputs/                           # (vacío - fases 4-7)
│   ├── vector_indices/
│   ├── evaluation/
│   └── logs/
│
├── requirements.txt                   # ✅ Dependencias
├── README.md                          # ✅ Documentación
└── .gitignore                         # ✅ Configurado
```

---

## 🎯 PRÓXIMOS PASOS INMEDIATOS

### 1. **Verificar API Qwen3-VL** (2 min)
```bash
# Verificar que Docker está corriendo
docker ps | grep qwen3vl

# Test de salud
curl http://localhost:8001/health
```

### 2. **Generar Embeddings** (15-20 min)
```bash
python scripts/03_generate_embeddings.py
```

**Output esperado:**
- Archivo: `data/processed/embeddings/embeddings.npy`
- Shape: `(2143, 384)`
- Metadata enriquecida: `data/processed/embeddings/enriched_crops_metadata.json`

### 3. **Construir Índice FAISS** (10 min)
```bash
python scripts/04_build_faiss_index.py
```

### 4. **Implementar RAG Retriever** (1-2 horas)

### 5. **Pipeline Completo** (2-3 horas)

---

## ❓ PREGUNTAS FRECUENTES

### ¿Por qué estrategia híbrida en lugar de embeddings nativos?
- **Decisión pragmática**: No requiere modificar la API Qwen3-VL en producción
- **Validación rápida**: Permite probar arquitectura RAG completa en días
- **Plan de migración**: Opción A (embeddings nativos) implementable después si resultados son buenos

### ¿Por qué 94% de crops son "very_small"?
- **Realista**: Surface scratches ocupan poco % del área total de la imagen
- **Esperado**: La mayoría de defectos vehiculares son pequeños rasguños
- **Justifica padding**: Necesidad de contexto espacial alrededor del defecto

### ¿Desbalance de clases es problema?
- **No para POC**: Refleja distribución real del mundo (scratches >> otros)
- **Estrategia RAG**: Filtros por tipo de daño permiten búsqueda específica
- **Evaluación**: Métricas por clase individual, no solo accuracy global

---

## 🔧 CONFIGURACIÓN DEL ENTORNO

### Dependencias Principales
```
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
PyYAML>=6.0
torch>=2.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
requests>=2.31.0
tqdm>=4.65.0
```

### API Qwen3-VL
- **Endpoint**: `http://localhost:8001/qwen3/chat/completions`
- **Formato**: Compatible OpenAI
- **Modelo**: Qwen3-VL-4B-Instruct
- **Docker**: Corriendo en puerto 8001

---

## 📝 NOTAS IMPORTANTES

1. **Metadata Rica**: Cada crop tiene 18 campos incluyendo:
   - Posición espacial relativa (x, y) en imagen original
   - Zona espacial (grilla 3×3)
   - Tamaño relativo a imagen
   - Flag de "edge defect"
   - Padding aplicado adaptativo

2. **Umbrales Adaptativos**: El sistema calcula automáticamente umbrales basándose en percentiles del dataset (P99 × 1.5 para máximo)

3. **Tasa de Aprovechamiento**: 99.4% (solo 12 de 2,155 defectos descartados)

4. **Desbalance Natural**: Surface scratches dominan (89%) - esto es correcto y esperado

---

## 🎨 EJEMPLO DE USO (Futuro - Fase 6)

```python
from src.core.rag.retriever import DamageRAGRetriever
from src.inference.analyzer import RAGDamageAnalyzer

# Inicializar sistema
analyzer = RAGDamageAnalyzer(
    retriever=retriever,
    embedder=embedder
)

# Analizar imagen
result = analyzer.analyze_image(
    image_path="test_images/scratch_door.jpg",
    question="¿Qué tipo de daño tiene esta puerta?",
    k_examples=3
)

# Resultado incluye:
# - Tipo de daño detectado
# - 3 ejemplos similares del dataset
# - Análisis comparativo del VLM
# - Score de confianza
```

---

## 🚀 TIMELINE RESTANTE

| Fase | Tarea | Tiempo Estimado | Estado |
|------|-------|-----------------|--------|
| **3** | Generar embeddings | 20 min | ⏳ Próximo |
| **4** | Construir índice FAISS | 10 min | ⏹️ |
| **5** | Implementar RAG Retriever | 2 horas | ⏹️ |
| **6** | Pipeline completo | 2 horas | ⏹️ |
| **7** | Evaluación y métricas | 2 horas | ⏹️ |

**Tiempo total restante estimado**: 6-8 horas de desarrollo activo

---

## 💡 SUGERENCIAS PARA LA IA

Al continuar este proyecto, por favor:

1. **Sigue el plan POC**: Está diseñado científicamente con papers de 2024-2025
2. **Mantén modularidad**: Cada componente debe ser testeable independientemente
3. **Prioriza métricas**: Toda decisión debe validarse con números
4. **Documenta cambios**: Actualiza este contexto si modificas arquitectura
5. **Piensa en escalado**: Código debe funcionar para 100 o 10,000 imágenes

---

## 📧 INFORMACIÓN ADICIONAL

- **Python Version**: 3.12
- **OS**: Ubuntu 24 (via Docker para Qwen3-VL)
- **GPU**: Recomendada (opcional para POC)
- **Última actualización**: 2025-11-03

---

## ✅ CHECKLIST RÁPIDO

```
✅ Dataset preparado (100 imágenes)
✅ Crops generados (2,143 ROIs)
✅ Metadata completa con contexto espacial
⏳ Embeddings (siguiente paso)
⏹️ Índice FAISS
⏹️ RAG Retriever
⏹️ Pipeline E2E
⏹️ Evaluación
```

---

**¿Listo para continuar? El próximo comando es:**
```bash
python scripts/03_generate_embeddings.py
```