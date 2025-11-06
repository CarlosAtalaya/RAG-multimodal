# src/core/api/ollama_client.py (VERSIÓN FINAL CORREGIDA)

import ollama
from pathlib import Path
from typing import Optional, Union
import time
import gc
import torch

class OllamaVLMClient:
    """Cliente para Qwen3-VL via Ollama"""
    
    def __init__(self, model: str = "qwen3-vl:4b", auto_cleanup: bool = True):
        self.model = model
        self.auto_cleanup = auto_cleanup
        
        try:
            ollama.show(self.model)
            print(f"✅ Modelo {self.model} encontrado")
        except:
            raise RuntimeError(f"Modelo {self.model} no encontrado")
    
    def generate(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """Genera respuesta"""
        
        try:
            # Construir mensaje
            if image_path:
                response = ollama.chat(
                    model=self.model,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [str(image_path)]
                    }],
                    options={
                        'num_predict': max_tokens,
                        'temperature': temperature,
                        'num_ctx': 8192,  # Contexto aumentado para visión
                    }
                )
            else:
                response = ollama.chat(
                    model=self.model,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }],
                    options={
                        'num_predict': max_tokens,
                        'temperature': temperature,
                    }
                )
            
            if self.auto_cleanup:
                self._clear_gpu_cache()
            
            # ✅ SOLUCIÓN: Manejar content y thinking
            if hasattr(response, 'message'):
                msg = response.message
                
                # Prioridad 1: content si existe y no está vacío
                if hasattr(msg, 'content') and msg.content:
                    return msg.content.strip()
                
                # Prioridad 2: thinking si content está vacío (caso VLM)
                if hasattr(msg, 'thinking') and msg.thinking:
                    print("ℹ️  Usando campo 'thinking' (comportamiento VLM)")
                    return msg.thinking.strip()
                
                # Prioridad 3: convertir mensaje a dict
                if isinstance(msg, dict):
                    if msg.get('content'):
                        return msg['content'].strip()
                    if msg.get('thinking'):
                        return msg['thinking'].strip()
            
            # Fallback para dict directo
            if isinstance(response, dict):
                if response.get('message', {}).get('content'):
                    return response['message']['content'].strip()
                if response.get('message', {}).get('thinking'):
                    return response['message']['thinking'].strip()
            
            # Último recurso
            return str(response)
        
        except Exception as e:
            self._clear_gpu_cache()
            print(f"❌ Error detallado: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error: {e}")
    
    def generate_with_retry(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        **kwargs
    ) -> str:
        """Genera con reintentos"""
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self._aggressive_cleanup()
                
                return self.generate(prompt=prompt, image_path=image_path, **kwargs)
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⚠️  Intento {attempt + 1}/{max_retries} falló")
                    print(f"   Error: {str(e)[:200]}")
                    print(f"   Esperando {wait_time:.1f}s...")
                    self._aggressive_cleanup()
                    time.sleep(wait_time)
                else:
                    print(f"❌ Error final después de {max_retries} intentos: {e}")
                    return f"Error: {str(e)}"
        
        return "Error: Máximo de reintentos"
    
    def _clear_gpu_cache(self):
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def _aggressive_cleanup(self):
        try:
            for _ in range(3):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(0.5)
        except:
            pass