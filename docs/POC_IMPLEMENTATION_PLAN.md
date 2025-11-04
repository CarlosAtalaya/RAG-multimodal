# 🤖 RAG MULTIMODAL - Detección de Defectos en Vehículos

Sistema RAG (Retrieval-Augmented Generation) multimodal para análisis inteligente de defectos vehiculares usando Vision-Language Models (VLM).

---

## 🎯 Objetivo del Proyecto

Construir un sistema que:
1. **Procesa** imágenes de vehículos con defectos etiquetados (polígonos)
2. **Genera** embeddings visuales de los defectos usando Qwen3-VL
3. **Indexa** vectores en FAISS para búsqueda de similitud
4. **Utiliza** RAG para responder preguntas sobre daños vehiculares con contexto de ejemplos similares

---

## 🛠️ Stack Tecnológico

- **VLM**: Qwen3-VL-4B-Instruct (via API Docker en `localhost:8001`)
- **Embeddings**: Estrategia híbrida (Qwen3-VL + Sentence-Transformers)
- **Vector DB**: FAISS (IndexHNSWFlat para POC)
- **Lenguaje**: Python 3.12
- **Dataset**: 60 imágenes POC 20/20/20 (high, medium and low defects density): high -> 1024 crops; medium -> 239 crops; low -> 17 crops

---

## 📊 Estado Actual del Proyecto
```
PROGRESO GLOBAL: ████████████████░░░░░░░░░░░░ 60% (Fase 5/7 completada)

✅ FASE 1: Preparación Dataset         [100%] ━━━━━━━━━━ COMPLETADO
✅ FASE 2: Generación Crops             [100%] ━━━━━━━━━━ COMPLETADO
✅ FASE 3: Generación Embeddings        [100%] ━━━━━━━━━━ COMPLETADO
✅ FASE 4: Construcción Índice FAISS   [100%] ━━━━━━━━━━ COMPLETADO
✅ FASE 5: RAG Retriever               [100%] ━━━━━━━━━━ COMPLETADO
⏹️ FASE 6: Análisis Completo            [  0%] ░░░░░░░░░░ PENDIENTE
⏹️ FASE 7: Evaluación y Métricas        [  0%] ░░░░░░░░░░ PENDIENTE
```

### ✅ Resultados Obtenidos

#### **Fase 1: Dataset POC**
- 100 imágenes seleccionadas estratégicamente
- 2,155 defectos totales etiquetados
- Promedio: 21.55 defectos/imagen
- Distribución balanceada por tipos de daño y zonas del vehículo

#### **Fase 2: Generación de Crops**
- 2,143 crops generados con padding adaptativo
- 99.4% tasa de aprovechamiento (12 descartados)
- Metadata enriquecida con 18 campos por crop
- Distribución espacial: 30.7% middle_center, 24.3% bottom_center

#### **Fase 3: Embeddings (Mini-POC)**
- 100 crops procesados inicialmente para validación
- Dimensión: 384 (all-MiniLM-L6-v2)
- Estrategia híbrida: Qwen3-VL → descripciones → Sentence-BERT
- Archivos generados:
  - `embeddings_mini_100.npy` (100 × 384)
  - `enriched_crops_metadata_mini_100.json`

#### **Fase 4: Índice FAISS**
- Índice construido: IndexHNSWFlat
- 100 vectores indexados (M=32, efConstruction=200, efSearch=64)
- Tamaño en disco: ~0.15 MB
- Validación exitosa: búsqueda k-NN funcional

---

## 🗂️ Estructura del Proyecto
```
RAG-multimodal/
├── data/
│   ├── raw/
│   │   └── 100_samples/              # Dataset POC (100 imágenes + JSONs)
│   └── processed/
│       ├── crops/roi/                # 2,143 crops generados
│       ├── metadata/
│       │   └── crops_metadata.json   # Metadata completa
│       └── embeddings/               # Embeddings generados
│
├── src/
│   ├── core/
│   │   ├── preprocessing/
│   │   │   └── crop_generator.py     # AdaptiveCropGenerator
│   │   ├── embeddings/
│   │   │   └── hybrid_embedder.py    # HybridEmbedder
│   │   ├── rag/                      # (en desarrollo)
│   │   └── vector_store/             # (en desarrollo)
│   └── utils/
│
├── scripts/
│   ├── 01_prepare_dataset.py         # ✅ Completado
│   ├── 02_generate_crops.py          # ✅ Completado
│   ├── 03_generate_embeddings.py     # ✅ Completado
│   ├── 04_build_faiss_index.py       # ✅ Completado
│   ├── test_retriever.py             # ⏳ Siguiente
│   └── 05_run_inference.py           # ⏹️ Pendiente
│
├── outputs/
│   └── vector_indices/
│       ├── indexhnswflat.index       # ✅ Índice FAISS
│       ├── metadata.pkl              # ✅ Metadata asociada
│       └── index_config.json         # ✅ Configuración
│
├── docs/
│   ├── RAG_MULTIMODAL_TECHNICAL_DESIGN.md
│   ├── POC_IMPLEMENTATION_PLAN.md
│   └── API_integration_guide.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Instalación
```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/RAG-multimodal.git
cd RAG-multimodal

# Crear entorno virtual
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar API Qwen3-VL

Asegúrate de tener la API Qwen3-VL corriendo en Docker:
```bash
# Verificar API
curl http://localhost:8001/health

# Debería retornar:
# {"status": "healthy", "model_loaded": true, "device": "cuda"}
```

### 3. Ejecutar Pipeline POC
```bash
# Fase 1: Preparar dataset (si no está hecho)
python scripts/01_prepare_dataset.py

# Fase 2: Generar crops
python scripts/02_generate_clustered_crops.py

# Fase 3: Generar embeddings (con DINOv3 o con QWEN3VL-API)
python scripts/03_generate_embeddings_dinov3

# Fase 4: Construir índice FAISS
python scripts/04_build_faiss_index.py

# Fase 5: Probar retriever
python scripts/test_retriever.py
```

---

## 📚 Documentación Técnica

### Documentos Clave

- **[Diseño Técnico Completo](docs/RAG_MULTIMODAL_TECHNICAL_DESIGN.md)**: Decisiones arquitectónicas fundamentadas científicamente
- **[Plan de Implementación POC](docs/POC_IMPLEMENTATION_PLAN.md)**: Roadmap de 7 días con scripts detallados
- **[Guía de Integración API](docs/API_integration_guide.md)**: 3 opciones de estrategia de embeddings

### Decisiones Clave

1. **Embeddings Jerárquicos**: Global (imagen completa) + ROI (crops con padding)
2. **Padding Adaptativo**: 20-40% según tipo de daño y tamaño
3. **Índice FAISS**: IndexHNSWFlat para <10K vectores (POC)
4. **Estrategia Híbrida**: Qwen3-VL genera descripciones → Sentence-BERT embeddings

---

## 🎯 Próximos Pasos

### Inmediatos (Esta Semana)
- [ ] **Fase 5**: Implementar RAG Retriever completo
- [ ] **Fase 6**: Pipeline end-to-end con Qwen3-VL para análisis
- [ ] **Fase 7**: Evaluación con métricas (Recall@k, Precision@k)

### Futuro (Próximas Iteraciones)
- [ ] Escalar a dataset completo (1,700 imágenes → ~36K crops)
- [ ] Optimizar índice FAISS (migrar a IndexIVFPQ)
- [ ] Implementar embeddings nativos (Opción A - endpoint directo)
- [ ] API REST para inferencia en producción

---

## 📈 Métricas Esperadas (Targets POC)

| Métrica | Target | Baseline (sin RAG) |
|---------|--------|-------------------|
| Recall@5 | >90% | ~65% |
| Answer Relevance | >0.85 | ~0.60 |
| Hallucination Rate | <10% | ~35% |
| Query Latency p95 | <200ms | N/A |

---

## 🔧 Configuración

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

### Variables de Entorno (Opcional)
```bash
# .env
QWEN_API_ENDPOINT=http://localhost:8001
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_TYPE=IndexHNSWFlat
```

---

## 📊 Dataset

### POC (Actual)
- **Imágenes**: 20 de alta densidad, 20 de media densidad y 20 de baja densidad de defectos

### Completo (Futuro)
- **Imágenes**: 2.700

---

## 🤝 Contribución

Este es un proyecto de investigación académica/técnica. 

### Guidelines
- Código modular y testeable
- Documentación científica fundamentada
- Métricas cuantitativas para cada decisión

---

## 📄 Licencia

[Especificar licencia - MIT, Apache 2.0, etc.]

---

## 📧 Contacto

- **Autor**: [Tu nombre]
- **Email**: [tu email]
- **GitHub**: [tu usuario]

---

## 🙏 Agradecimientos

- **Qwen3-VL** (Alibaba Cloud): VLM de última generación
- **FAISS** (Meta): Librería de búsqueda de similitud eficiente
- **Sentence-Transformers** (UKPLab): Embeddings de texto de alta calidad

---

**Última actualización**: 2025-11-03  
**Versión**: 0.4.0-alpha (Fase 4/7 completada)