# 🚀 RAG Multimodal para Detección de Defectos en Vehículos

Sistema de Retrieval-Augmented Generation (RAG) multimodal basado en Qwen3-VL para detección y análisis de daños vehiculares.

## 📋 Descripción

Este proyecto implementa un sistema RAG que combina:
- **Vision-Language Model (VLM)**: Qwen3-VL-4B-Instruct
- **Vector Database**: FAISS para búsqueda de similitud
- **Embeddings híbridos**: Descripciones textuales + sentence-transformers

## 🏗️ Estructura del Proyecto
```
RAG-multimodal/
├── data/              # Datasets y datos procesados
├── src/               # Código fuente modular
├── scripts/           # Scripts ejecutables
├── config/            # Configuraciones
├── docs/              # Documentación
├── tests/             # Tests unitarios e integración
└── outputs/           # Resultados y modelos
```

## 🚀 Inicio Rápido

### 1. Instalación
```bash
# Clonar repositorio
git clone <repo-url>
cd RAG-multimodal

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Preparar Dataset POC (100 imágenes)
```bash
python scripts/01_prepare_dataset.py
```

### 3. Generar Crops con Padding Adaptativo
```bash
python scripts/02_generate_crops.py
```

### 4. Generar Embeddings
```bash
# Asegúrate de que tu API Qwen3-VL está corriendo
docker ps | grep qwen3vl

# Generar embeddings
python scripts/03_generate_embeddings.py
```

## 📚 Documentación

- [Plan de Implementación POC](docs/POC_IMPLEMENTATION_PLAN.md)
- [Diseño Técnico RAG](docs/RAG_MULTIMODAL_TECHNICAL_DESIGN.md)
- [Guía de Integración API](docs/API_integration_guide.md)

## 🧪 Testing
```bash
pytest tests/
```

## 📊 Estado del Proyecto

- [x] FASE 1: Preparación dataset (100 imágenes)
- [x] FASE 2: Generación de crops con padding
- [ ] FASE 3: Generación de embeddings
- [ ] FASE 4: Construcción índice FAISS
- [ ] FASE 5: Sistema RAG completo
- [ ] FASE 6: Evaluación y métricas

## 🤝 Contribución

Este es un proyecto de investigación. Para contribuir, por favor abre un issue primero.

## 📝 Licencia

[Especificar licencia]

## 📧 Contacto

[Tu información de contacto]
