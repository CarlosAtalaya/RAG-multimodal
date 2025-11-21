# main.py
import os
import json
import time
import cv2
from pathlib import Path

from utils import (
    cargar_configuracion,
    construir_arbol_taxonomia,
    image_to_base64,
    aplicar_y_codificar_distorsiones,
    obtener_info_sistema,
    imprimir_eta
)
from models import (
    consultar_modelo_vlm,
    consultar_modelo_text_only,
    normalizar_respuesta
)
from metrics import despachar_calculo_metricas
from reporting import imprimir_resumen_consola, generar_reporte_final

# ✨ RAG IMPORTS
try:
    from rag import DamageRAGRetriever, MultimodalEmbedder, RAGPromptBuilder
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  RAG module not available: {e}")
    print("⚠️  Running without RAG support.")
    RAG_AVAILABLE = False

CONFIG_FILE = "config.yaml"


def inicializar_rag(config):
    """
    Initializes RAG components with MultimodalEmbedder
    
    Returns:
        tuple: (retriever, embedder, prompt_builder) or (None, None, None)
    """
    if not RAG_AVAILABLE:
        return None, None, None
    
    rag_config = config.get("rag_config", {})
    
    if not rag_config.get("enabled", False):
        print("ℹ️  RAG disabled in configuration")
        return None, None, None
    
    print("\n" + "="*70)
    print("🔧 INITIALIZING RAG SYSTEM")
    print("="*70)
    
    try:
        # Index paths
        index_dir = Path(rag_config.get("index_path", "vector_indices/dinov3_hybrid_dano_nodano"))
        index_filename = rag_config.get("index_filename", "indexhnswflat_dano_nodano.index")
        metadata_filename = rag_config.get("metadata_filename", "metadata_dano_nodano.pkl")
        config_filename = rag_config.get("config_filename", "index_config_dano_nodano.json")
        
        index_path = index_dir / index_filename
        metadata_path = index_dir / metadata_filename
        config_path = index_dir / config_filename
        
        # Verify existence
        if not index_path.exists():
            print(f"❌ FAISS index not found: {index_path}")
            return None, None, None
        
        if not metadata_path.exists():
            print(f"❌ Metadata not found: {metadata_path}")
            return None, None, None
        
        # Normalización taxonómica
        enable_normalization = rag_config.get("enable_taxonomy_normalization", True)
        
        # Initialize retriever
        print(f"📦 Loading retriever...")
        print(f"   - Index: {index_path.name}")
        print(f"   - Taxonomy normalization: {'✅' if enable_normalization else '❌'}")
        
        retriever = DamageRAGRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            config_path=config_path if config_path.exists() else None,
            enable_taxonomy_normalization=enable_normalization
        )
        
        # ✅ Siempre usar MultimodalEmbedder
        visual_weight = rag_config.get("visual_weight", 0.6)
        text_weight = rag_config.get("text_weight", 0.4)
        
        print(f"\n🔧 Initializing MultimodalEmbedder (hybrid)...")
        print(f"   - Visual weight: {visual_weight}")
        print(f"   - Text weight: {text_weight}")
        
        embedder = MultimodalEmbedder(
            visual_weight=visual_weight,
            text_weight=text_weight,
            verbose=False
        )
        
        # Create prompt builder
        prompt_builder = RAGPromptBuilder()
        
        print("\n✅ RAG system initialized successfully")
        print(f"   • Index: {retriever.index.ntotal} vectors")
        print(f"   • Embedding dim: {retriever.embedding_dim}")
        print(f"   • Embedding type: hybrid (visual + text)")
        print(f"   • Data type: {retriever.data_type}")
        print(f"   • Taxonomy normalization: {'✅ Enabled' if enable_normalization else '❌ Disabled'}")
        print(f"   • Top-k: {rag_config.get('top_k', 5)}")
        
        # ✨ Mostrar composición del dataset
        if retriever.has_no_damage_samples:
            stats = retriever.get_stats()
            dataset_stats = stats['dataset_stats']
            print(f"   • Dataset composition:")
            print(f"     - Images with damage: {dataset_stats['damage_images']}")
            print(f"     - Images without damage: {dataset_stats['no_damage_images']}")
        
        print("="*70 + "\n")
        
        return retriever, embedder, prompt_builder
    
    except Exception as e:
        print(f"❌ Error initializing RAG: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def debe_usar_rag(task_name, modelo_a_evaluar, config, rag_components):
    """
    Determina si RAG debe usarse para esta tarea y modelo
    
    Returns:
        dict: {
            'use_rag': bool,
            'filters': dict | None,
            'balance': bool,
            'min_damage_examples': int
        }
    """
    # Si RAG no está inicializado, return False
    retriever, embedder, prompt_builder = rag_components
    if not all([retriever, embedder, prompt_builder]):
        return {
            'use_rag': False,
            'filters': None,
            'balance': False,
            'min_damage_examples': 2
        }
    
    rag_config = config.get("rag_config", {})
    
    # Check si RAG está globalmente habilitado
    if not rag_config.get("enabled", False):
        return {
            'use_rag': False,
            'filters': None,
            'balance': False,
            'min_damage_examples': 2
        }
    
    # Check si modelo actual debe usar RAG
    modelo_con_rag = rag_config.get("modelo_con_rag", "")
    if modelo_a_evaluar != modelo_con_rag:
        return {
            'use_rag': False,
            'filters': None,
            'balance': False,
            'min_damage_examples': 2
        }
    
    # ✨ NUEVA LÓGICA: Configurar filtros y balanceo por tarea
    filters = None
    balance = rag_config.get("balance_damage_types", False)
    min_damage_examples = rag_config.get("min_damage_examples", 2)
    
    if task_name == "Hallucination":
        # Para hallucination, ejemplos sin daño son MUY útiles
        # para calibrar y evitar falsos positivos
        filters = None  # Permitir ambos tipos
        balance = True  # Balancear con/sin daño
        min_damage_examples = 1  # Al menos 1 con daño
    
    elif task_name == "hF1":
        # Para clasificación, preferir ejemplos con daño
        # pero permitir algunos sin daño para calibración
        filters = None  # Permitir ambos
        balance = rag_config.get("balance_damage_types", False)
        min_damage_examples = 2
    
    elif task_name == "PDS":
        # Para robustez ante distorsiones, solo ejemplos con daño relevante
        if rag_config.get("exclude_no_damage_for_pds", True):
            filters = {'exclude_no_damage': True}
        balance = False
        min_damage_examples = 3
    
    else:
        # Otras tareas: permitir ambos tipos
        filters = None
        balance = False
        min_damage_examples = 2
    
    return {
        'use_rag': True,
        'filters': filters,
        'balance': balance,
        'min_damage_examples': min_damage_examples
    }


def construir_query_metadata_desde_gt(ground_truth, filename, verbose=False):
    """
    ✨ NUEVA FUNCIÓN: Construye metadata de query desde ground truth
    
    Detecta correctamente:
    - Imágenes CON daño
    - Imágenes SIN daño
    
    Args:
        ground_truth: Lista de dict con ground truth
        filename: Nombre de archivo (para extraer zona)
        verbose: Imprimir info de debug
    
    Returns:
        dict: Metadata con has_damage correcto
    """
    import re
    
    # ✨ 1. Detectar si hay daño en ground truth
    has_damage_in_gt = False
    defect_types_found = []
    
    if isinstance(ground_truth, list):
        for item in ground_truth:
            if isinstance(item, dict):
                damage = item.get('damage', 'unknown')
                
                # Detectar "No damage" explícitamente
                if damage and damage.lower() in ['no damage', 'no_damage', 'clean', 'none']:
                    has_damage_in_gt = False
                    defect_types_found = []
                    break
                
                # Detectar daño válido
                if damage and damage not in ['unknown', 'Unknown', '']:
                    has_damage_in_gt = True
                    defect_types_found.append(damage)
    
    # ✨ 2. Extraer zona del filename
    # Formato: zona1_ok_2_3_1556819769657_zona_7_imageDANO_original.jpg
    zone_num = 'unknown'
    zone_desc = 'unknown area'
    zone_area = 'unknown'
    
    zone_match = re.search(r'_zona_(\d+)_', filename)
    if zone_match:
        zone_num = zone_match.group(1)
        
        # Mapeo simple de zonas (AJUSTAR según tu config)
        zone_map = {
            '1': ('front left fender', 'frontal'),
            '2': ('hood center', 'frontal'),
            '3': ('front right fender', 'frontal'),
            '4': ('rear left quarter', 'posterior'),
            '5': ('rear bumper', 'posterior'),
            '6': ('rear right quarter', 'posterior'),
            '7': ('driver side door', 'lateral_left'),
            '8': ('driver side rocker', 'lateral_left'),
            '9': ('passenger side door', 'lateral_right'),
            '10': ('passenger side rocker', 'lateral_right')
        }
        
        if zone_num in zone_map:
            zone_desc, zone_area = zone_map[zone_num]
    
    # ✨ 3. Construir metadata según detección
    if has_damage_in_gt:
        metadata = {
            'has_damage': True,
            'defect_types': defect_types_found if defect_types_found else ['unknown'],
            'zone_description': zone_desc,
            'zone_area': zone_area,
            'vehicle_zone': zone_num
        }
    else:
        # ✨ Caso sin daño detectado
        metadata = {
            'has_damage': False,
            'defect_types': [],  # ✅ Lista vacía, NO ['unknown']
            'zone_description': zone_desc,
            'zone_area': zone_area,
            'vehicle_zone': zone_num
        }
    
    if verbose:
        print(f"     📋 Query metadata construida:")
        print(f"        - has_damage: {metadata['has_damage']}")
        print(f"        - defect_types: {metadata['defect_types']}")
        print(f"        - zone: {zone_desc} ({zone_area})")
    
    return metadata


def main():
    print("--- INICIANDO PROCESO DE EVALUACIÓN MULTI-TAREA ---")
    info_sistema = obtener_info_sistema()
    print("--- INFORMACIÓN DEL SISTEMA RECOLECTADA ---")
    for key, value in info_sistema.items():
        print(f"  {key}: {value}")
    
    config = cargar_configuracion(CONFIG_FILE)
    if not config:
        print("Falta el archivo de configuración. Finalizando.")
        return

    PROMPTS_FILE = config.get("prompts_file")
    if not PROMPTS_FILE or not os.path.exists(PROMPTS_FILE):
        print(f"Error: El archivo de prompts '{PROMPTS_FILE}' definido en el config no existe.")
        return
    
    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            prompts_globales = json.load(f)
        print(f"Archivo de prompts globales '{PROMPTS_FILE}' cargado correctamente.")
    except json.JSONDecodeError:
        print(f"Error: El archivo de prompts '{PROMPTS_FILE}' no es un JSON válido.")
        return

    gestor = config.get("gestor")
    api_key = None
    if gestor == "ollama":
        api_endpoint = config.get("ollama_config", {}).get("endpoint")
    elif gestor == "lm_studio":
        api_endpoint = config.get("lm_studio_config", {}).get("endpoint")
    elif gestor == "vllm":
        api_endpoint = config.get("vllm_config", {}).get("endpoint")
    elif gestor == 'qwen3':
        api_endpoint = config.get("qwen_config", {}).get("endpoint")
    elif gestor == "openai":
        api_endpoint = None
        api_key = config.get("openai_config", {}).get("api_key")
    elif gestor == "gemini":
        api_endpoint = None
        api_key = config.get("gemini_config", {}).get("api_key")
    else:
        print(f"Error: El gestor '{gestor}' no es válido. Opciones: 'ollama', 'lm_studio'.")
        return

    if not api_endpoint and gestor not in ["openai", "gemini"]:
        print(f"Error: No se encontró el 'endpoint' para el gestor '{gestor}' en el config.")
        return
        
    print(f"Usando el gestor: '{gestor}' con el endpoint: '{api_endpoint}'")
    
    OUTPUT_FILE = config.get("output_file")

    tareas_a_ejecutar = config.get("tareas_a_ejecutar", None)
    if tareas_a_ejecutar is not None:
        print(f"Se ejecutarán únicamente las siguientes tareas especificadas en el config: {tareas_a_ejecutar}")
    else:
        print("Advertencia: No se encontró la clave 'tareas_a_ejecutar'. Se ejecutarán todas las tareas encontradas en los archivos JSON.")

    arbol_taxonomia = construir_arbol_taxonomia()
    print(f"Ábol de taxonomía construido con {len(arbol_taxonomia.nodes())} nodos.")

    # ✨ INICIALIZAR RAG
    rag_components = inicializar_rag(config)
    retriever, embedder, prompt_builder = rag_components

    IMAGES_DIR = config.get("directorio_imagenes")
    if not IMAGES_DIR or not os.path.isdir(IMAGES_DIR):
        print(f"Error: El directorio de imágenes '{IMAGES_DIR}' no es válido.")
        return
    
    modelo_a_evaluar = config.get("modelo_a_evaluar")
    print(f"Evaluando el modelo '{modelo_a_evaluar}'...")
    
    pasos_completados = 0
    start_time_proceso = time.perf_counter()
    resultados_agregados_por_tarea = {}
    total_prompt_tokens = 0
    total_eval_tokens = 0
    
    print(f"\n=======================================================")
    print(f"======== ESCANEANDO IMÁGENES Y PROMPTS EN '{IMAGES_DIR}' ========")
    print(f"=======================================================")
    
    archivos_en_directorio = sorted(os.listdir(IMAGES_DIR))
    total_pasos = sum(1 for f in archivos_en_directorio if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    print(f"Se analizarán un total de {total_pasos} imágenes.")
    
    for filename in archivos_en_directorio:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(IMAGES_DIR, filename)
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(IMAGES_DIR, json_filename)
        
        if not os.path.exists(json_path):
            print(f"\nAdvertencia: Se encontró la imagen '{filename}' pero no su JSON ('{json_filename}'). Saltando.")
            continue
            
        print(f"\n--- Procesando imagen: {filename} con su JSON: {json_filename} ---")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                ground_truths_para_imagen = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: El archivo '{json_filename}' no es un JSON válido. Saltando.")
            continue

        imagen_b64_original = image_to_base64(image_path)
        if not imagen_b64_original:
            print(f"No se pudo procesar la imagen '{filename}'. Saltando.")
            continue

        for task_name, gt_info in ground_truths_para_imagen.items():
            if tareas_a_ejecutar is not None and task_name not in tareas_a_ejecutar:
                continue

            prompts_para_tarea = prompts_globales[task_name]
            
            if isinstance(gt_info, list) and gt_info:
                ground_truth_para_tarea = gt_info[0].get('ground_truth')
            else:
                ground_truth_para_tarea = None

            if ground_truth_para_tarea is None:
                print(f"Advertencia: No se encontró la clave 'ground_truth' para la tarea '{task_name}' en '{json_filename}'. Saltando tarea.")
                continue

            resultados_agregados_por_tarea.setdefault(task_name, [])
            
            # ===== TAREA PDS (CON DISTORSIONES) =====
            if task_name == "PDS":
                imagen_cv_original = cv2.imread(image_path)
                if imagen_cv_original is None:
                    print(f"Error al leer la imagen '{filename}' con OpenCV para la tarea PDS. Saltando.")
                    continue
                
                for i, item in enumerate(prompts_para_tarea):
                    max_level = item.get("nivel", 0)
                    if max_level == 0:
                        print("Advertencia: item de PDS no tiene 'nivel'. Saltando.")
                        continue
                        
                    for nivel_actual in range(1, max_level + 1):
                        print(f"  -> Tarea '{task_name}', Prompt {i+1}/{len(prompts_para_tarea)}, Nivel de distorsión {nivel_actual}/{max_level}")
                        
                        imagen_b64_distorsionada = aplicar_y_codificar_distorsiones(imagen_cv_original, nivel_actual)
                        
                        # ✨ RAG: Generar contexto si está habilitado
                        prompt_to_use = item['prompt']
                        rag_used = False
                        
                        # ✨ Obtener configuración RAG para tarea
                        rag_decision = debe_usar_rag(task_name, modelo_a_evaluar, config, rag_components)
                        
                        if rag_decision['use_rag']:
                            try:
                                print("     🔍 Generating RAG context...")
                                
                                # ✨ NUEVA FUNCIÓN: Construir metadata desde GT
                                query_metadata = construir_query_metadata_desde_gt(
                                    ground_truth=ground_truth_para_tarea,
                                    filename=filename,
                                    verbose=True
                                )
                                
                                # ✅ Generar embedding híbrido CON metadata correcta
                                query_embedding = embedder.generate_embedding(
                                    image_path=Path(image_path),
                                    normalize=True,
                                    metadata=query_metadata  # ✅ CRUCIAL: Tiene has_damage correcto
                                )
                                
                                print(f"     🔍 Debug:")
                                print(f"       - Query embedding dim: {query_embedding.shape[0]}")
                                print(f"       - Index embedding dim: {retriever.embedding_dim}")
                                print(f"       - Match: {query_embedding.shape[0] == retriever.embedding_dim}")
                                
                                # ✨ Búsqueda con filtros específicos de tarea
                                top_k = config.get("rag_config", {}).get("top_k", 5)
                                search_results = retriever.search(
                                    query_embedding=query_embedding,
                                    k=top_k,
                                    filters=rag_decision['filters']  # ✨ Filtros por tarea (PDS: solo con daño)
                                )
                                
                                # ✨ Construir prompt con balanceo opcional
                                max_examples = config.get("rag_config", {}).get("max_examples_in_prompt", 3)
                                prompt_to_use = prompt_builder.inject_rag_context(
                                    original_prompt=item['prompt'],
                                    search_results=search_results,
                                    max_examples=max_examples,
                                    balance=rag_decision['balance'],  # ✨ Balanceo por tarea
                                    min_damage_examples=rag_decision['min_damage_examples']
                                )
                                
                                rag_used = True
                                
                                # ✨ Estadísticas de resultados
                                n_damage = sum(1 for r in search_results if r.has_damage)
                                n_no_damage = len(search_results) - n_damage
                                print(f"     ✓ RAG context injected:")
                                print(f"       - {len(search_results)} examples retrieved")
                                print(f"       - {n_damage} with damage, {n_no_damage} without damage")
                            
                            except Exception as e:
                                print(f"     ⚠️  RAG failed: {e}")
                                import traceback
                                traceback.print_exc()
                                print(f"     → Falling back to original prompt")
                                prompt_to_use = item['prompt']
                                rag_used = False
                        
                        # Llamada al modelo
                        start_time = time.perf_counter()
                        respuesta_data = consultar_modelo_vlm(
                            prompt=prompt_to_use,
                            image_b64=imagen_b64_distorsionada,
                            model_name=modelo_a_evaluar,
                            temperature=config.get("hiperparametros", {}).get("temperature"),
                            top_k=config.get("hiperparametros", {}).get("top_k"),
                            endpoint=api_endpoint,
                            gestor=gestor,
                            api_key=api_key
                        )
                        end_time = time.perf_counter()
                        inference_time = end_time - start_time
                        
                        if respuesta_data:
                            respuesta_modelo_raw = respuesta_data["response"]
                            total_prompt_tokens += respuesta_data["prompt_tokens"]
                            total_eval_tokens += respuesta_data["eval_tokens"]
                        else:
                            respuesta_modelo_raw = None

                        # Normalización base
                        respuesta_modelo_norm = normalizar_respuesta(respuesta_modelo_raw, ground_truth_para_tarea)

                        print(f"     GT: {ground_truth_para_tarea}")
                        print(f"     Respuesta (Nivel {nivel_actual}): {respuesta_modelo_norm}")

                        # --- 🔒 Normalización adicional para compatibilidad con metrics.py ---
                        resp = respuesta_modelo_norm
                        if not isinstance(resp, (dict, list)):
                            # Si llega algo como int, float, None, etc.
                            respuesta_modelo_norm = [{"damage": "unknown", "part": "unknown"}]
                        elif isinstance(resp, dict):
                            # Garantiza que tenga ambas claves
                            respuesta_modelo_norm = [{
                                "damage": resp.get("damage", "unknown"),
                                "part": resp.get("part", "unknown")
                            }]
                        elif isinstance(resp, list):
                            # Asegura que cada elemento sea un dict válido
                            respuesta_modelo_norm = [
                                {"damage": r.get("damage", "unknown"), "part": r.get("part", "unknown")}
                                for r in resp if isinstance(r, dict)
                            ]
                        # --- fin normalización segura ---

                        resultado_individual = {
                            "imagen": filename,
                            "prompt": item['prompt'],
                            "ground_truth": ground_truth_para_tarea,
                            "respuesta_modelo": respuesta_modelo_raw,
                            "respuesta_normalizada": respuesta_modelo_norm,
                            "nivel": nivel_actual,
                            "tiempo_inferencia": inference_time,
                            "prompt_tokens": respuesta_data["prompt_tokens"] if respuesta_data else 0,
                            "eval_tokens": respuesta_data["eval_tokens"] if respuesta_data else 0,
                            "rag_used": rag_used
                        }

                        resultados_agregados_por_tarea[task_name].append(resultado_individual)
                        time.sleep(0.1)
            
            # ===== TAREAS NORMALES (SIN DISTORSIÓN) =====
            else:
                for i, item in enumerate(prompts_para_tarea):
                    print(f"  -> Tarea '{task_name}', Prompt {i+1}/{len(prompts_para_tarea)}") 
                    
                    # ✨ RAG: Generar contexto si está habilitado
                    prompt_to_use = item['prompt']
                    rag_used = False
                    
                    # ✨ Obtener configuración RAG para tarea
                    rag_decision = debe_usar_rag(task_name, modelo_a_evaluar, config, rag_components)
                    
                    if rag_decision['use_rag']:
                        try:
                            print("     🔍 Generating RAG context...")
                            
                            # ✨ NUEVA FUNCIÓN: Construir metadata desde GT
                            query_metadata = construir_query_metadata_desde_gt(
                                ground_truth=ground_truth_para_tarea,
                                filename=filename,
                                verbose=True
                            )
                            
                            # ✅ Generar embedding híbrido CON metadata correcta
                            query_embedding = embedder.generate_embedding(
                                image_path=Path(image_path),
                                normalize=True,
                                metadata=query_metadata  # ✅ CRUCIAL: Tiene has_damage correcto
                            )
                            
                            print(f"     🔍 Debug:")
                            print(f"       - Query embedding dim: {query_embedding.shape[0]}")
                            print(f"       - Index embedding dim: {retriever.embedding_dim}")
                            print(f"       - Match: {query_embedding.shape[0] == retriever.embedding_dim}")
                            
                            # ✨ Búsqueda con filtros específicos de tarea
                            top_k = config.get("rag_config", {}).get("top_k", 5)
                            search_results = retriever.search(
                                query_embedding=query_embedding,
                                k=top_k,
                                filters=rag_decision['filters']  # ✨ Filtros por tarea
                            )
                            
                            # ✨ Construir prompt con balanceo opcional
                            max_examples = config.get("rag_config", {}).get("max_examples_in_prompt", 3)
                            prompt_to_use = prompt_builder.inject_rag_context(
                                original_prompt=item['prompt'],
                                search_results=search_results,
                                max_examples=max_examples,
                                balance=rag_decision['balance'],  # ✨ Balanceo por tarea
                                min_damage_examples=rag_decision['min_damage_examples']
                            )
                            
                            rag_used = True
                            
                            # ✨ Estadísticas de resultados
                            n_damage = sum(1 for r in search_results if r.has_damage)
                            n_no_damage = len(search_results) - n_damage
                            print(f"     ✓ RAG context injected:")
                            print(f"       - {len(search_results)} examples retrieved")
                            print(f"       - {n_damage} with damage, {n_no_damage} without damage")
                        
                        except Exception as e:
                            print(f"     ⚠️  RAG failed: {e}")
                            import traceback
                            traceback.print_exc()
                            print(f"     → Falling back to original prompt")
                            prompt_to_use = item['prompt']
                            rag_used = False
                    
                    # Llamada al modelo
                    start_time = time.perf_counter()
                    respuesta_data = consultar_modelo_vlm(
                        prompt=prompt_to_use,
                        image_b64=imagen_b64_original,
                        model_name=modelo_a_evaluar,
                        temperature=config.get("hiperparametros", {}).get("temperature"),
                        top_k=config.get("hiperparametros", {}).get("top_k"),
                        endpoint=api_endpoint,
                        gestor=gestor,
                        api_key=api_key
                    )
                    
                    end_time = time.perf_counter()
                    inference_time = end_time - start_time

                    if respuesta_data:
                        respuesta_modelo_raw = respuesta_data["response"]
                        total_prompt_tokens += respuesta_data["prompt_tokens"]
                        total_eval_tokens += respuesta_data["eval_tokens"]
                    else:
                        respuesta_modelo_raw = None
                    
                    respuesta_modelo_norm = normalizar_respuesta(respuesta_modelo_raw, ground_truth_para_tarea)

                    print(f"     GT: {ground_truth_para_tarea}")
                    print(f"     Respuesta Multimodal: {respuesta_modelo_norm}")
                    print(f"     Tiempo de inferencia: {inference_time:.2f} segundos")
                    if respuesta_data:
                        print(f"     Prompt Tokens: {respuesta_data['prompt_tokens']}")
                        print(f"     Answer Tokens: {respuesta_data['eval_tokens']}")

                    resultado_individual = {
                        "imagen": filename,
                        "prompt": item['prompt'],
                        "ground_truth": ground_truth_para_tarea,
                        "respuesta_modelo": respuesta_modelo_raw,
                        "respuesta_normalizada": respuesta_modelo_norm,
                        "tiempo_inferencia": inference_time,
                        "prompt_tokens": respuesta_data["prompt_tokens"] if respuesta_data else 0,
                        "eval_tokens": respuesta_data["eval_tokens"] if respuesta_data else 0,
                        "rag_used": rag_used
                    }
                    resultados_agregados_por_tarea[task_name].append(resultado_individual)
                    time.sleep(0.1) 
        
        pasos_completados += 1
        imprimir_eta(pasos_completados, total_pasos, start_time_proceso)


    print(f"\n\n=======================================================")
    print(f"====== CÁLCULO FINAL DE MÉTRICAS PARA TODAS LAS TAREAS ======")
    print(f"=======================================================")
    
    evaluacion_completa = {}
    
    task_order = ["hF1", "Hallucination", "PDS"] 
    processed_tasks = set(resultados_agregados_por_tarea.keys())
    
    all_tasks_ordered = [t for t in task_order if t in processed_tasks] + \
                        [t for t in processed_tasks if t not in task_order]

    for task_name in all_tasks_ordered:
        resultados_detallados = resultados_agregados_por_tarea[task_name]
        metricas = despachar_calculo_metricas(task_name, resultados_detallados, arbol_taxonomia, metricas_base=evaluacion_completa)
        tiempos_inferencia = [r['tiempo_inferencia'] for r in resultados_detallados if 'tiempo_inferencia' in r]
        if tiempos_inferencia:
            tiempo_medio = sum(tiempos_inferencia) / len(tiempos_inferencia)
            metricas['tiempo_medio_inferencia_s'] = round(tiempo_medio, 2)
        
        imprimir_resumen_consola(task_name, modelo_a_evaluar, metricas)
        
        metricas_por_prompt = {}
        resultados_por_prompt = {}
        for resultado in resultados_detallados:
            prompt = resultado['prompt']
            if prompt not in resultados_por_prompt:
                resultados_por_prompt[prompt] = []
            resultados_por_prompt[prompt].append(resultado)
        
        for idx, (prompt, resultados_prompt) in enumerate(resultados_por_prompt.items(), 1):
            metricas_prompt = despachar_calculo_metricas(task_name, resultados_prompt, arbol_taxonomia, metricas_base=evaluacion_completa)
            prompt_key = f"Prompt_{idx}"
            metricas_por_prompt[prompt_key] = {
                "prompt_text": prompt,
                "metricas": metricas_prompt
            }
        
        evaluacion_completa[task_name] = {
            "metricas": metricas,
            "metricas_por_prompt": metricas_por_prompt,
            "resultados_detallados": resultados_detallados
        }

    if not evaluacion_completa:
        print("\nNo se procesó ninguna imagen con un JSON válido o ninguna tarea coincide con la configuración. No se generará ningún reporte.")
    else:
        REPORTS_DIR = "reportes"
        os.makedirs(REPORTS_DIR, exist_ok=True) 

        modelo_para_filename = modelo_a_evaluar.replace(":", "-")

        # ✨ Añadir sufijo RAG si está habilitado
        rag_suffix = ""
        rag_config = config.get("rag_config", {})
        if rag_config.get("enabled", False):
            # ✨ Añadir info de dataset usado
            index_path = rag_config.get("index_path", "")
            if "dano_nodano" in index_path:
                rag_suffix = "_rag_dano-nodano"
            else:
                rag_suffix = "_rag"

        nombre_base, extension = os.path.splitext(OUTPUT_FILE)
        nombre_archivo_dinamico = f"{nombre_base}_{modelo_para_filename}{rag_suffix}{extension}"

        ruta_salida_final = os.path.join(REPORTS_DIR, nombre_archivo_dinamico)

        reporte_final = generar_reporte_final(config, evaluacion_completa, total_prompt_tokens, total_eval_tokens, info_sistema)
        with open(ruta_salida_final, 'w', encoding='utf-8') as f:
            json.dump(reporte_final, f, ensure_ascii=False, indent=4)
        
        print(f"\n\nResultados de las tareas ejecutadas guardados en '{ruta_salida_final}'.")
        print(f"\n--- RESUMEN DE TOKENS ---")
        print(f"Total tokens de entrada (prompt): {total_prompt_tokens}")
        print(f"Total tokens de salida (generados): {total_eval_tokens}")
        print(f"Total tokens: {total_prompt_tokens + total_eval_tokens}")
    
    print("--- PROCESO DE EVALUACIÓN FINALIZADO ---")


if __name__ == "__main__":
    main()