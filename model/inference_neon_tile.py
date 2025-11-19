# model/inference_neon_tile.py
"""
Inferencia sobre un tile del dataset NEON para demostración en Streamlit.

Este módulo envuelve:
- Carga de modelos (CHM + RNet) desde ssl_model.
- Construcción del NeonDataset con la misma lógica que en el paper.
- Cálculo de métricas equivalentes al evaluate() original, pero sobre un solo tile.

Se usa en el "Modo NEON" de la página de demostración.
"""

from typing import Dict, Any

import numpy as np
import torch
import torchvision.transforms as T

from .ssl_model import load_chm_model, load_rnet_normalizer
from .neon_data import build_neon_dataset, get_neon_sample
from .metrics import compute_all_metrics


def setup_neon_inference(
    checkpoint_name: str = "compressed_SSLhuge_aerial.pth",
    normtype: int = 2,
    trained_rgb: bool = False,
    src_img: str = "neon",
) -> Dict[str, Any]:
    """
    Configura todos los componentes necesarios para hacer inferencia sobre NEON:

    - Carga el modelo de normalización RNet (si normtype requiere RNet).
    - Construye el NeonDataset con la configuración deseada.
    - Carga el modelo de altura de dosel (SSLModule) con el checkpoint dado.
    - Define la normalización global de imagen usada en el paper.

    Esta función está pensada para llamarse una sola vez
    (por ejemplo, decorada con st.cache_resource en Streamlit).

    Parameters
    ----------
    checkpoint_name:
        Nombre del checkpoint de CHM dentro de saved_checkpoints/.
        Por defecto 'compressed_SSLhuge_aerial.pth'.
    normtype:
        Tipo de normalización de imágenes aéreas, igual que en el paper:
        - 0: no_norm=True (sin normalización de dominio con RNet/Maxar).
        - 1: normalización "vieja" usando imagen Maxar.
        - 2: normalización automática usando RNet (default del paper).
    trained_rgb:
        True si se usara un modelo finetuneado en RGB aéreo (no es el caso aquí).
    src_img:
        Tipo de imagen fuente en el CSV. En este proyecto usamos 'neon'.

    Returns
    -------
    components: dict con claves:
        - "model": modelo de CHM (SSLModule) en eval().
        - "device": dispositivo donde está el modelo ('cpu' o 'cuda:0').
        - "dataset": instancia de NeonDataset.
        - "norm": transform Normalize para la entrada del modelo.
    """
    # 1. Cargar RNet solo si normtype lo requiere
    if normtype == 2:
        model_norm = load_rnet_normalizer()
    else:
        model_norm = None

    # 2. Construir el dataset NEON con esa configuración
    dataset = build_neon_dataset(
        model_norm=model_norm,
        normtype=normtype,
        trained_rgb=trained_rgb,
        src_img=src_img,
    )

    # 3. Cargar el modelo de altura de dosel (CHM) desde el checkpoint indicado
    model, device = load_chm_model(checkpoint_name=checkpoint_name)

    # 4. Normalización global por canal, igual que en inference.py
    norm = T.Normalize(
        mean=(0.420, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )

    components = {
        "model": model,
        "device": device,
        "dataset": dataset,
        "norm": norm,
    }
    return components


def run_neon_tile_inference(
    components: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """
    Ejecuta inferencia sobre un tile del NeonDataset dado por 'index'.

    Usa los componentes devueltos por setup_neon_inference:
    - modelo de CHM
    - dispositivo
    - dataset
    - normalización global

    Calcula las métricas equivalentes al evaluate() original:
    - MAE
    - RMSE
    - R2 pixel a pixel
    - R2_block (sobre bloques downsampleados)
    - Bias

    Parameters
    ----------
    components:
        Diccionario devuelto por setup_neon_inference().
    index:
        Índice entero del tile a evaluar dentro del NeonDataset.

    Returns
    -------
    result: dict con:
        - "img_rgb": np.ndarray [H,W,3] con la imagen aérea original.
        - "chm_gt": np.ndarray [H,W] con el CHM real.
        - "chm_pred": np.ndarray [H,W] con el CHM predicho.
        - "metrics": dict con claves "mae", "rmse", "r2", "r2_block", "bias".
    """
    model = components["model"]
    device = components["device"]
    dataset = components["dataset"]
    norm = components["norm"]

    # 1. Obtener el sample del dataset (imagen original, normalizada y CHM real)
    img_no_norm, img_norm, chm = get_neon_sample(dataset, index)

    # 2. Preparar batch de tamaño 1 y aplicar normalización global
    x = img_norm.unsqueeze(0)  # [1,3,H,W]
    x = norm(x)
    x = x.to(device)

    # 3. Ejecutar inferencia
    model.eval()
    with torch.no_grad():
        pred = model(x)          # [1,1,H,W]
        pred = pred.cpu().relu() # asegurar no-negatividad
        pred_map = pred[0, 0].numpy()  # [H,W]

    chm_map = chm[0].numpy()  # [H,W]

    # 4. Calcular métricas usando el helper del módulo metrics
    metrics = compute_all_metrics(pred_map, chm_map)

    # 5. Convertir la imagen RGB a formato [H,W,3] para visualización en Streamlit
    img_rgb = np.moveaxis(img_no_norm.numpy(), 0, 2)  # [H,W,3]

    result = {
        "img_rgb": img_rgb,
        "chm_gt": chm_map,
        "chm_pred": pred_map,
        "metrics": metrics,
    }
    return result
