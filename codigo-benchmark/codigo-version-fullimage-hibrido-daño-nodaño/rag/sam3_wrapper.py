# rag/sam3_wrapper.py

import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List, Any

# Gestión robusta de imports
try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
    from sam3.eval.postprocessors import PostProcessImage
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ [SAM3] Error crítico de dependencias: {e}")
    print("   Asegúrate de tener instalado: sam3, decord, torch, torchvision")
    SAM3_AVAILABLE = False
    
    # Definir Dummies para evitar NameError si fallan los imports
    class Datapoint: pass
    class SAMImage: pass
    
class SAM3Segmenter:
    """
    Wrapper profesional para SAM3.
    Mantiene el modelo cargado en memoria para inferencia continua imagen a imagen.
    """
    def __init__(self, config: dict = None):
        if not SAM3_AVAILABLE:
            raise ImportError("La librería SAM3 o sus dependencias (como 'decord') no están instaladas.")

        self.config = config or {}
        
        # IMPORTANTE: Forzado a CPU para dejar la GPU libre a Ollama/Qwen
        self.device = "cpu" 
        
        # Configuración de rutas
        self.bpe_path = self.config.get("bpe_path", "./assets/bpe_simple_vocab_16e6.txt.gz")
        self.text_prompt = self.config.get("text_prompt", "car")
        self.global_counter = 1

        print(f"🔧 [SAM3] Inicializando modelo en {self.device}...")
        
        # 1. Lógica Robusta de Carga de Assets (BPE)
        if not os.path.exists(self.bpe_path):
            print(f"   ⚠️ No se encontró BPE en ruta configurada: {self.bpe_path}")
            found_fallback = False
            
            # Intento 1: Buscar relativo a la librería instalada (si es posible)
            try:
                if hasattr(sam3, '__file__') and sam3.__file__:
                    sam3_root = os.path.dirname(os.path.dirname(sam3.__file__))
                    possible_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
                    if os.path.exists(possible_path):
                        self.bpe_path = possible_path
                        found_fallback = True
                        print(f"   ✅ BPE encontrado en librería: {self.bpe_path}")
            except Exception:
                pass

            # Intento 2: Buscar en carpeta local ./assets
            if not found_fallback:
                local_assets = os.path.join(os.getcwd(), "assets", "bpe_simple_vocab_16e6.txt.gz")
                if os.path.exists(local_assets):
                    self.bpe_path = local_assets
                    print(f"   ✅ BPE encontrado localmente: {self.bpe_path}")
            
        try:
            self.model = build_sam3_image_model(bpe_path=self.bpe_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"❌ Error fatal cargando pesos SAM3. Verifica que el archivo '{self.bpe_path}' existe. Error: {e}")

        # 2. Configurar Transformaciones
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # 3. Configurar Post-procesador
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=0.5,
            to_cpu=True,
        )
        print("✅ [SAM3] Modelo cargado y listo.")

    def _create_datapoint(self, pil_image) -> Datapoint:
        w, h = pil_image.size
        dp = Datapoint(find_queries=[], images=[])
        dp.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
        
        self.global_counter += 1
        query = FindQueryLoaded(
            query_text=self.text_prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=self.global_counter,
                original_image_id=self.global_counter,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
        dp.find_queries.append(query)
        return dp

    def process_image(self, image_path: str) -> Tuple[Optional[Image.Image], Optional[List[int]]]:
        if not os.path.exists(image_path):
            print(f"❌ [SAM3] Imagen no encontrada: {image_path}")
            return None, None

        try:
            pil_image = Image.open(image_path).convert("RGB")
            img_np = np.array(pil_image)
            
            dp = self._create_datapoint(pil_image)
            dp = self.transform(dp)
            
            batch = collate([dp], dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, torch.device(self.device), non_blocking=True)

            # Inferencia (Float32 para CPU, Bfloat16 si fuera CUDA)
            dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
            
            with torch.inference_mode(), torch.autocast(self.device, dtype=dtype):
                output = self.model(batch)
                results = self.postprocessor.process_results(output, batch.find_metadatas)

            if not results: return None, None
            result = list(results.values())[0]
            masks = result.get('masks')
            
            if masks is None: return None, None

            if isinstance(masks, torch.Tensor):
                masks_np = masks.float().cpu().numpy()
            elif isinstance(masks, list):
                masks_np = np.array(masks)
            else:
                masks_np = masks

            if masks_np.ndim > 2:
                h_m, w_m = masks_np.shape[-2:]
                masks_np = masks_np.reshape(-1, h_m, w_m)
            elif masks_np.ndim == 2:
                masks_np = masks_np[None, :, :]

            if masks_np.shape[0] == 0:
                return None, None

            areas = masks_np.sum(axis=(1, 2))
            largest_idx = int(np.argmax(areas))
            largest_mask = masks_np[largest_idx]
            largest_mask = (largest_mask > 0).astype(np.uint8)

            if largest_mask.shape != img_np.shape[:2]:
                temp_mask_pil = Image.fromarray(largest_mask * 255).resize(
                    (img_np.shape[1], img_np.shape[0]), resample=Image.NEAREST
                )
                largest_mask = np.array(temp_mask_pil) // 255

            contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bbox = [0, 0, img_np.shape[1], img_np.shape[0]]
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                bbox = list(cv2.boundingRect(largest_contour))
            else:
                return None, None

            mask_3d = np.stack([largest_mask] * 3, axis=-1)
            final_img_np = img_np * mask_3d
            
            return Image.fromarray(final_img_np.astype(np.uint8)), bbox

        except Exception as e:
            print(f"❌ [SAM3] Error procesando {os.path.basename(image_path)}: {e}")
            return None, None