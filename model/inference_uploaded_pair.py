# model/inference_uploaded_pair.py
"""
Inferencia sobre un par (imagen RGB subida + CHM real opcional).

Este módulo está pensado para el "Modo de imagen subida" de la demo en Streamlit.
Hace SOLO la parte de modelo:

- Carga el modelo de altura de dosel (CHM) desde ssl_model.load_chm_model.
- Define la normalización global de imagen (igual que en el paper).
- Recibe una imagen RGB en memoria (np.ndarray) y, opcionalmente, un CHM real.
- Devuelve el CHM predicho y, si hay CHM real, las métricas:
  MAE, RMSE, R² pixel, R² en bloques y Bias.

Las validaciones fuertes del input (tamaño 256×256, 3 canales, contraste, etc.)
se recomienda hacerlas en la capa de Streamlit antes de llamar a estas funciones.
"""

from typing import Dict, Any, Optional

import numpy as np
import torch
import torchvision.transforms as T

from .ssl_model import load_chm_model
from .metrics import compute_all_metrics


def setup_uploaded_inference(
    checkpoint_name: str = "compressed_SSLhuge_aerial.pth",
) -> Dict[str, Any]:
    """
    Carga el modelo de CHM y la normalización global usada en el paper.

    Parameters
    ----------
    checkpoint_name:
        Nombre del checkpoint de CHM dentro de saved_checkpoints/.
        Por defecto 'compressed_SSLhuge_aerial.pth'.

    Returns
    -------
    components: dict con:
        - "model": modelo de CHM (SSLModule) en eval().
        - "device": dispositivo ('cpu' o 'cuda:0').
        - "norm": transform torchvision.transforms.Normalize para la entrada.
    """
    model, device = load_chm_model(checkpoint_name=checkpoint_name)

    # Normalización global por canal (igual que inference.py del repo)
    norm = T.Normalize(
        mean=(0.420, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )

    components = {
        "model": model,
        "device": device,
        "norm": norm,
    }
    return components


def run_uploaded_inference(
    components: Dict[str, Any],
    rgb: np.ndarray,
    chm_gt: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Ejecuta inferencia sobre una imagen RGB subida por el usuario.

    Parameters
    ----------
    components:
        Diccionario devuelto por setup_uploaded_inference().
    rgb:
        Imagen RGB como np.ndarray de forma [H, W, 3].
        Puede venir en uint8 (0–255) o float32 (0–1).
    chm_gt:
        CHM real opcional como np.ndarray de forma [H, W] o [H, W, 1].
        Si se proporciona y las dimensiones coinciden con la predicción,
        se calculan las métricas de error.

    Returns
    -------
    result: dict con claves:
        - "img_rgb": np.ndarray [H, W, 3] float32 en [0,1].
        - "chm_pred": np.ndarray [H, W] con el CHM predicho.
        - "chm_gt": np.ndarray [H, W] con el CHM real (o None).
        - "metrics": dict con "mae", "rmse", "r2", "r2_block", "bias" (o None).
    """
    model = components["model"]
    device = components["device"]
    norm = components["norm"]

    # -------------------------
    # Validaciones básicas RGB
    # -------------------------
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"La imagen RGB debe tener forma [H, W, 3]. "
            f"Forma recibida: {rgb.shape}"
        )

    # Convertir a float32 en [0,1]
    if rgb.dtype != np.float32:
        rgb_f = rgb.astype("float32")
        if rgb_f.max() > 1.0 + 1e-3:
            rgb_f /= 255.0
    else:
        rgb_f = rgb.copy()

    # -------------------------
    # Preparar tensor para el modelo
    # -------------------------
    # [H,W,3] -> [1,3,H,W]
    x = torch.from_numpy(rgb_f).permute(2, 0, 1).unsqueeze(0)
    x = norm(x).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x)              # [1,1,H,W]
        pred = pred.cpu().relu()     # asegurar no-negatividad
        chm_pred = pred[0, 0].numpy()  # [H,W]

    chm_gt_out: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, float]] = None

    # -------------------------
    # Si hay CHM real, alinear y calcular métricas
    # -------------------------
    if chm_gt is not None:
        chm_arr = np.asarray(chm_gt, dtype="float32")

        # Si viene con canal extra [H,W,1], lo exprimimos
        if chm_arr.ndim == 3 and chm_arr.shape[2] == 1:
            chm_arr = chm_arr[..., 0]

        if chm_arr.shape != chm_pred.shape:
            raise ValueError(
                f"El CHM real tiene forma {chm_arr.shape}, "
                f"pero la predicción tiene forma {chm_pred.shape}. "
                "Deben coincidir exactamente."
            )

        chm_gt_out = chm_arr
        metrics = compute_all_metrics(chm_pred, chm_gt_out)

    result: Dict[str, Any] = {
        "img_rgb": rgb_f,
        "chm_pred": chm_pred,
        "chm_gt": chm_gt_out,
        "metrics": metrics,
    }
    return result
