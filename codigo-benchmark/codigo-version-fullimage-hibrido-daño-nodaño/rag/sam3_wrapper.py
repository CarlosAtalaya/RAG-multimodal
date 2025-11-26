import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List

# Imports de SAM3 (Tal cual los tienes en tu script funcional)
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
    print(f"⚠️ Error crítico: Librería SAM3 no encontrada o incompleta: {e}")
    SAM3_AVAILABLE = False

class SAM3Segmenter:
    """
    Wrapper profesional para SAM3.
    Mantiene el modelo cargado en memoria para inferencia continua imagen a imagen.
    """
    def __init__(self, config: dict = None):
        if not SAM3_AVAILABLE:
            raise ImportError("SAM3 no está disponible. Revisa los requirements.")

        self.config = config or {}
        
        # Configuración hardcodeada basada en tu script exitoso
        # Ajusta las rutas si cambian en tu entorno final
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bpe_path = self.config.get("bpe_path", "./assets/bpe_simple_vocab_16e6.txt.gz")
        self.text_prompt = self.config.get("text_prompt", "car")
        
        # Gestión de IDs internos de SAM
        self.global_counter = 1

        print(f"🔧 [SAM3] Inicializando modelo en {self.device}...")
        
        # 1. Cargar Modelo
        # Fallback de ruta si no se encuentra assets relativo
        if not os.path.exists(self.bpe_path):
            sam3_root = os.path.dirname(os.path.dirname(sam3.__file__))
            self.bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
            
        self.model = build_sam3_image_model(bpe_path=self.bpe_path)
        self.model.to(self.device)
        self.model.eval()

        # 2. Configurar Transformaciones (Exactas a tu script)
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
        """Helper interno para crear la estructura de datos que exige SAM3"""
        w, h = pil_image.size
        
        # Crear Datapoint vacío
        dp = Datapoint(find_queries=[], images=[])
        
        # Set Image
        dp.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
        
        # Add Text Prompt ("car")
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
        """
        Procesa una imagen y devuelve:
        1. PIL Image con el coche segmentado (fondo negro).
        2. Bounding Box [x, y, w, h] del contorno del coche.
        """
        if not os.path.exists(image_path):
            print(f"❌ [SAM3] Imagen no encontrada: {image_path}")
            return None, None

        try:
            # Cargar imagen
            pil_image = Image.open(image_path).convert("RGB")
            img_np = np.array(pil_image)
            
            # Preparar inputs
            dp = self._create_datapoint(pil_image)
            dp = self.transform(dp)
            
            # Collate (Simular batch de 1)
            batch = collate([dp], dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, torch.device(self.device), non_blocking=True)

            # Inferencia (Usando bfloat16 como en tu script)
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                output = self.model(batch)
                results = self.postprocessor.process_results(output, batch.find_metadatas)

            # Extraer resultado (primero del batch)
            if not results: return None, None
            result = list(results.values())[0]
            
            masks = result.get('masks')
            if masks is None: return None, None

            # Convertir máscaras a Numpy
            if isinstance(masks, torch.Tensor):
                masks_np = masks.float().cpu().numpy()
            elif isinstance(masks, list):
                masks_np = np.array(masks)
            else:
                masks_np = masks

            # Reshape logic (Vital para consistencia)
            if masks_np.ndim > 2:
                h_m, w_m = masks_np.shape[-2:]
                masks_np = masks_np.reshape(-1, h_m, w_m)
            elif masks_np.ndim == 2:
                masks_np = masks_np[None, :, :]

            if masks_np.shape[0] == 0:
                print(f"⚠️ [SAM3] No se detectaron máscaras para {os.path.basename(image_path)}")
                return None, None

            # --- LÓGICA DE SELECCIÓN (Tu script original) ---
            # Quedarnos con la máscara más grande
            areas = masks_np.sum(axis=(1, 2))
            largest_idx = int(np.argmax(areas))
            
            largest_mask = masks_np[largest_idx]
            largest_mask = (largest_mask > 0).astype(np.uint8)

            # Resize de seguridad (por si SAM devuelve mask en otra resolución)
            if largest_mask.shape != img_np.shape[:2]:
                temp_mask_pil = Image.fromarray(largest_mask * 255).resize(
                    (img_np.shape[1], img_np.shape[0]), resample=Image.NEAREST
                )
                largest_mask = np.array(temp_mask_pil) // 255

            # Calcular Contorno y BBox
            contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bbox = [0, 0, img_np.shape[1], img_np.shape[0]] # Default full image
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                # BBox: x, y, w, h
                bbox = list(cv2.boundingRect(largest_contour))
            else:
                # Si hay máscara pero no contorno claro, retornamos None por seguridad
                return None, None

            # Aplicar máscara negra al fondo
            mask_3d = np.stack([largest_mask] * 3, axis=-1)
            final_img_np = img_np * mask_3d
            
            # Retornar imagen limpia y bbox
            final_image = Image.fromarray(final_img_np.astype(np.uint8))
            return final_image, bbox

        except Exception as e:
            print(f"❌ [SAM3] Error procesando {os.path.basename(image_path)}: {e}")
            import traceback
            traceback.print_exc()
            return None, None