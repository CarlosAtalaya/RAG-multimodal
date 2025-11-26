# main.py
import os
import json
import time
import cv2
from pathlib import Path

# Imports de utilidades originales (Intactos)
from utils import (
    cargar_configuracion,
    construir_arbol_taxonomia,
    image_to_base64,
    aplicar_y_codificar_distorsiones,
    obtener_info_sistema,
    imprimir_eta
)
# Imports de modelos (Restaurados completos)
from models import (
    consultar_modelo_vlm,
    consultar_modelo_text_only, # Recuperado por compatibilidad
    normalizar_respuesta
)
from metrics import despachar_calculo_metricas
from reporting import imprimir_resumen_consola, generar_reporte_final

# ✨ RAG IMPORTS (NUEVA ARQUITECTURA)
# Usamos un bloque try/except robusto para no detener la ejecución si falla la carga parcial
try:
    from rag import MultimodalRAGPipeline
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  RAG module not available: {e}")
    print("⚠️  Running without RAG support.")
    RAG_AVAILABLE = False

CONFIG_FILE = "config.yaml"

def inicializar_rag(config):
    """
    Inicializa el Pipeline RAG Multimodal completo.
    Devuelve None si está deshabilitado o falla.
    """
    if not RAG_AVAILABLE:
        return None
    
    rag_config = config.get("rag_config", {})
    
    if not rag_config.get("enabled", False):
        print("ℹ️  RAG disabled in configuration")
        return None
    
    print("\n" + "="*70)
    print("🔧 INITIALIZING MULTIMODAL RAG PIPELINE")
    print("="*70)
    
    try:
        # El pipeline inicializa internamente: SAM3, GridGenerator, MetaCLIP, FAISS
        rag_pipeline = MultimodalRAGPipeline(config)
        return rag_pipeline
    
    except Exception as e:
        print(f"❌ Error initializing RAG Pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def debe_usar_rag(task_name, modelo_a_evaluar, config, rag_pipeline):
    """
    Decide si activar RAG para una iteración específica.
    """
    if not rag_pipeline:
        return False
    
    rag_config = config.get("rag_config", {})
    
    # 1. Check global
    if not rag_config.get("enabled", False):
        return False
    
    # 2. Check modelo específico
    modelo_con_rag = rag_config.get("modelo_con_rag", "")
    if modelo_con_rag and modelo_a_evaluar != modelo_con_rag:
        return False
    
    return True

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
        print(f"Error: El archivo de prompts '{PROMPTS_FILE}' no existe.")
        return
    
    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            prompts_globales = json.load(f)
        print(f"Archivo de prompts '{PROMPTS_FILE}' cargado correctamente.")
    except json.JSONDecodeError:
        print(f"Error: El archivo de prompts '{PROMPTS_FILE}' no es un JSON válido.")
        return

    # Configuración del Gestor
    gestor = config.get("gestor")
    api_key = None
    api_endpoint = None
    
    if gestor == "ollama":
        api_endpoint = config.get("ollama_config", {}).get("endpoint")
    elif gestor == "lm_studio":
        api_endpoint = config.get("lm_studio_config", {}).get("endpoint")
    elif gestor == "vllm":
        api_endpoint = config.get("vllm_config", {}).get("endpoint")
    elif gestor == 'qwen3':
        api_endpoint = config.get("qwen_config", {}).get("endpoint")
    elif gestor == "openai":
        api_key = config.get("openai_config", {}).get("api_key")
    elif gestor == "gemini":
        api_key = config.get("gemini_config", {}).get("api_key")
    else:
        print(f"Error: El gestor '{gestor}' no es válido.")
        return

    if not api_endpoint and gestor not in ["openai", "gemini"]:
        print(f"Error: No se encontró el endpoint para '{gestor}'.")
        return
        
    print(f"Usando gestor: '{gestor}' con endpoint: '{api_endpoint}'")
    
    OUTPUT_FILE = config.get("output_file")
    tareas_a_ejecutar = config.get("tareas_a_ejecutar", None)
    
    if tareas_a_ejecutar:
        print(f"Tareas a ejecutar: {tareas_a_ejecutar}")
    
    arbol_taxonomia = construir_arbol_taxonomia()
    
    # ✨ INICIALIZAR RAG
    rag_pipeline = inicializar_rag(config)

    IMAGES_DIR = config.get("directorio_imagenes")
    if not IMAGES_DIR or not os.path.isdir(IMAGES_DIR):
        print(f"Error: El directorio '{IMAGES_DIR}' no es válido.")
        return
    
    modelo_a_evaluar = config.get("modelo_a_evaluar")
    print(f"Evaluando el modelo '{modelo_a_evaluar}'...")
    
    pasos_completados = 0
    start_time_proceso = time.perf_counter()
    resultados_agregados_por_tarea = {}
    total_prompt_tokens = 0
    total_eval_tokens = 0
    
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
            print(f"\nAdvertencia: Falta JSON para '{filename}'. Saltando.")
            continue
            
        print(f"\n--- Procesando imagen: {filename} ---")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                ground_truths_para_imagen = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: JSON inválido en '{json_filename}'.")
            continue

        imagen_b64_original = image_to_base64(image_path)
        if not imagen_b64_original:
            print(f"Error procesando imagen '{filename}'.")
            continue

        # ✨ OPTIMIZACIÓN RAG: Pre-cálculo por imagen
        # Calculamos el contexto RAG una sola vez para la imagen limpia
        # y lo reutilizamos en todas las tareas/prompts de esta imagen.
        rag_context_cached = None
        rag_calculated_flag = False # Para no reintentar si falla una vez

        for task_name, gt_info in ground_truths_para_imagen.items():
            if tareas_a_ejecutar is not None and task_name not in tareas_a_ejecutar:
                continue

            prompts_para_tarea = prompts_globales[task_name]
            ground_truth_para_tarea = gt_info[0].get('ground_truth') if gt_info else None

            if ground_truth_para_tarea is None:
                print(f"Advertencia: Sin GT para '{task_name}'. Saltando.")
                continue

            resultados_agregados_por_tarea.setdefault(task_name, [])
            
            # Decisión de RAG
            should_use_rag = debe_usar_rag(task_name, modelo_a_evaluar, config, rag_pipeline)

            # Ejecutar Pipeline RAG si es necesario y no se ha hecho aún
            if should_use_rag and not rag_calculated_flag:
                try:
                    print(f"     🔍 [RAG Pipeline] Analizando {filename}...")
                    rag_context_cached = rag_pipeline.run(image_path)
                    rag_calculated_flag = True
                except Exception as e:
                    print(f"     ⚠️ RAG Pipeline Error: {e}")
                    rag_context_cached = None
                    rag_calculated_flag = True # Marcar como intentado

            # ===== TAREA PDS (DISTORSIONES) =====
            if task_name == "PDS":
                imagen_cv_original = cv2.imread(image_path)
                if imagen_cv_original is None: continue
                
                for i, item in enumerate(prompts_para_tarea):
                    max_level = item.get("nivel", 0)
                    base_prompt = item['prompt']
                    
                    # Preparar Prompt con RAG
                    prompt_final = base_prompt
                    rag_used_in_prompt = False
                    if rag_context_cached:
                        prompt_final = f"{base_prompt}\n\n{rag_context_cached}"
                        rag_used_in_prompt = True

                    for nivel_actual in range(1, max_level + 1):
                        print(f"  -> {task_name} | Prompt {i+1} | Nivel {nivel_actual}")
                        
                        # Generar distorsión
                        img_dist = aplicar_y_codificar_distorsiones(imagen_cv_original, nivel_actual)
                        
                        # Inferencia
                        start_t = time.perf_counter()
                        resp_data = consultar_modelo_vlm(
                            prompt=prompt_final,
                            image_b64=img_dist,
                            model_name=modelo_a_evaluar,
                            temperature=config["hiperparametros"]["temperature"],
                            top_k=config["hiperparametros"]["top_k"],
                            endpoint=api_endpoint,
                            gestor=gestor,
                            api_key=api_key
                        )
                        end_t = time.perf_counter()
                        
                        # Procesar Respuesta
                        raw_response = resp_data["response"] if resp_data else None
                        
                        # Métricas de tokens
                        p_tok = resp_data["prompt_tokens"] if resp_data else 0
                        e_tok = resp_data["eval_tokens"] if resp_data else 0
                        total_prompt_tokens += p_tok
                        total_eval_tokens += e_tok

                        # Normalización (Con tu bloque de seguridad original)
                        norm_response = normalizar_respuesta(raw_response, ground_truth_para_tarea)
                        
                        # Safety check extra que tenías implícito en tu lógica
                        if not isinstance(norm_response, (dict, list)):
                            norm_response = [{"damage": "unknown", "part": "unknown"}]
                        elif isinstance(norm_response, dict):
                            norm_response = [norm_response]
                        
                        print(f"     GT: {ground_truth_para_tarea}")
                        print(f"     Resp: {norm_response}")

                        # Guardar Resultado (ESTRUCTURA EXACTA AL ORIGINAL)
                        resultados_agregados_por_tarea[task_name].append({
                            "imagen": filename,
                            "prompt": base_prompt,
                            "ground_truth": ground_truth_para_tarea,
                            "respuesta_modelo": raw_response,
                            "respuesta_normalizada": norm_response,
                            "nivel": nivel_actual,
                            "tiempo_inferencia": end_t - start_t,
                            "prompt_tokens": p_tok,
                            "eval_tokens": e_tok,
                            "rag_used": rag_used_in_prompt,
                            "rag_context": rag_context_cached # Info extra útil para debug
                        })
                        time.sleep(0.1)

            # ===== TAREAS NORMALES (hF1, Hallucination) =====
            else:
                for i, item in enumerate(prompts_para_tarea):
                    print(f"  -> {task_name} | Prompt {i+1}")
                    base_prompt = item['prompt']
                    
                    # Preparar Prompt con RAG
                    prompt_final = base_prompt
                    rag_used_in_prompt = False
                    if rag_context_cached:
                        prompt_final = f"{base_prompt}\n\n{rag_context_cached}"
                        rag_used_in_prompt = True
                    
                    # Inferencia
                    start_t = time.perf_counter()
                    resp_data = consultar_modelo_vlm(
                        prompt=prompt_final,
                        image_b64=imagen_b64_original,
                        model_name=modelo_a_evaluar,
                        temperature=config["hiperparametros"]["temperature"],
                        top_k=config["hiperparametros"]["top_k"],
                        endpoint=api_endpoint,
                        gestor=gestor,
                        api_key=api_key
                    )
                    end_t = time.perf_counter()
                    
                    raw_response = resp_data["response"] if resp_data else None
                    
                    p_tok = resp_data["prompt_tokens"] if resp_data else 0
                    e_tok = resp_data["eval_tokens"] if resp_data else 0
                    total_prompt_tokens += p_tok
                    total_eval_tokens += e_tok

                    norm_response = normalizar_respuesta(raw_response, ground_truth_para_tarea)
                    
                    # Safety check
                    if not isinstance(norm_response, (dict, list)):
                        norm_response = [{"damage": "unknown", "part": "unknown"}]
                    elif isinstance(norm_response, dict):
                        norm_response = [norm_response]

                    print(f"     Resp: {norm_response}")

                    resultados_agregados_por_tarea[task_name].append({
                        "imagen": filename,
                        "prompt": base_prompt,
                        "ground_truth": ground_truth_para_tarea,
                        "respuesta_modelo": raw_response,
                        "respuesta_normalizada": norm_response,
                        "tiempo_inferencia": end_t - start_t,
                        "prompt_tokens": p_tok,
                        "eval_tokens": e_tok,
                        "rag_used": rag_used_in_prompt,
                        "rag_context": rag_context_cached
                    })
                    time.sleep(0.1)
        
        pasos_completados += 1
        imprimir_eta(pasos_completados, total_pasos, start_time_proceso)

    # --- GENERACIÓN DE REPORTES ---
    print(f"\n\n=======================================================")
    print(f"====== CÁLCULO FINAL DE MÉTRICAS ======")
    print(f"=======================================================")
    
    evaluacion_completa = {}
    
    # Orden de tareas preferido
    task_order = ["hF1", "Hallucination", "PDS"]
    processed = set(resultados_agregados_por_tarea.keys())
    final_order = [t for t in task_order if t in processed] + [t for t in processed if t not in task_order]

    for task_name in final_order:
        resultados = resultados_agregados_por_tarea[task_name]
        metricas = despachar_calculo_metricas(task_name, resultados, arbol_taxonomia, metricas_base=evaluacion_completa)
        
        # Cálculo de tiempo medio
        times = [r['tiempo_inferencia'] for r in resultados if 'tiempo_inferencia' in r]
        if times: metricas['tiempo_medio_inferencia_s'] = round(sum(times)/len(times), 2)
        
        imprimir_resumen_consola(task_name, modelo_a_evaluar, metricas)
        
        # Agrupación por prompt para desglose
        res_by_prompt = {}
        met_by_prompt = {}
        for r in resultados:
            p = r['prompt']
            if p not in res_by_prompt: res_by_prompt[p] = []
            res_by_prompt[p].append(r)
            
        for idx, (p, sub_res) in enumerate(res_by_prompt.items(), 1):
            sub_met = despachar_calculo_metricas(task_name, sub_res, arbol_taxonomia, metricas_base=evaluacion_completa)
            met_by_prompt[f"Prompt_{idx}"] = {"prompt_text": p, "metricas": sub_met}

        evaluacion_completa[task_name] = {
            "metricas": metricas,
            "metricas_por_prompt": met_by_prompt,
            "resultados_detallados": resultados
        }

    if not evaluacion_completa:
        print("No se generaron resultados.")
    else:
        REPORTS_DIR = "reportes"
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Generar nombre de archivo con flag RAG
        rag_tag = "_RAG-Grid" if rag_pipeline else ""
        safe_model_name = modelo_a_evaluar.replace(":", "-")
        base_name = os.path.splitext(OUTPUT_FILE)[0]
        final_name = f"{base_name}_{safe_model_name}{rag_tag}.json"
        
        ruta_salida = os.path.join(REPORTS_DIR, final_name)
        
        reporte = generar_reporte_final(config, evaluacion_completa, total_prompt_tokens, total_eval_tokens, info_sistema)
        
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=4, ensure_ascii=False)
            
        print(f"\nResultados guardados en: {ruta_salida}")
        print(f"Tokens Prompt: {total_prompt_tokens}")
        print(f"Tokens Eval: {total_eval_tokens}")

    print("--- PROCESO DE EVALUACIÓN FINALIZADO ---")

if __name__ == "__main__":
    main()