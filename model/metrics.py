# model/metrics.py
"""
Funciones de métricas para evaluar mapas de altura de dosel.

Reproduce las métricas del evaluate() original:
- MAE
- RMSE
- R2 pixel a pixel
- R2 sobre bloques (downsample)
- Bias

Se usan tanto en:
- Modo NEON (tile desde el dataset).
- Modo archivos subidos (RGB + CHM).
"""

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torchmetrics

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convierte un tensor de PyTorch o un array de NumPy en un np.ndarray 2D.
    Si viene con shape [1,H,W] o [H,W], lo deja en [H,W].
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    # Quitar dimensión de canal si es [1,H,W]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    return x


def compute_all_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    block_size: int = 50,
) -> Dict[str, float]:
    """
    Calcula MAE, RMSE, R2, R2_block y Bias para un par de mapas 2D.

    Parameters
    ----------
    pred:
        Mapa predicho (torch.Tensor o np.ndarray), shape [H,W] o [1,H,W].
    target:
        Mapa real (torch.Tensor o np.ndarray), misma shape que pred.
    block_size:
        Tamaño del bloque para el cálculo de R2_block. Por defecto 50,
        igual que en el código original (AvgPool2d(50)).

    Returns
    -------
    metrics: dict con claves
        - "mae"
        - "rmse"
        - "r2"
        - "r2_block"
        - "bias"
    """
    # Convertir a tensores 1x1xH xW para usar torchmetrics
    if isinstance(pred, np.ndarray):
        pred_t = torch.from_numpy(pred).float()
    else:
        pred_t = pred.detach().float().cpu()

    if isinstance(target, np.ndarray):
        target_t = torch.from_numpy(target).float()
    else:
        target_t = target.detach().float().cpu()

    # Asegurar shape [1,1,H,W]
    if pred_t.ndim == 2:
        pred_t = pred_t.unsqueeze(0).unsqueeze(0)
    elif pred_t.ndim == 3:  # [1,H,W]
        pred_t = pred_t.unsqueeze(0)

    if target_t.ndim == 2:
        target_t = target_t.unsqueeze(0).unsqueeze(0)
    elif target_t.ndim == 3:
        target_t = target_t.unsqueeze(0)

    # Métricas tipo canopy height (igual que en evaluate())
    mae_metric = torchmetrics.MeanAbsoluteError()
    rmse_metric = torchmetrics.MeanSquaredError(squared=False)
    r2_metric = torchmetrics.R2Score()

    mae = mae_metric(pred_t, target_t).item()
    rmse = rmse_metric(pred_t, target_t).item()

    # R2 pixel a pixel: flatten
    r2 = r2_metric(
        pred_t.flatten(),
        target_t.flatten(),
    ).item()

    # R2 por bloques: downsample de ambos mapas
    # replicando AvgPool2d(50) de evaluate()
    downsampler = nn.AvgPool2d(kernel_size=block_size)
    bd = 3  # borde que quitaban en el código original

    # recortar el borde como en evaluate(): chm[..., bd:, bd:]
    tgt_block = downsampler(target_t[..., bd:, bd:])
    pred_block = downsampler(pred_t[..., bd:, bd:])

    r2_block_metric = torchmetrics.R2Score()
    r2_block = r2_block_metric(
        pred_block.flatten(),
        tgt_block.flatten(),
    ).item()

    # Bias: media de (pred - target) pixel a pixel
    pred_np = _to_numpy(pred_t)
    target_np = _to_numpy(target_t)
    bias = float((pred_np - target_np).mean())

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "r2_block": r2_block,
        "bias": bias,
    }
