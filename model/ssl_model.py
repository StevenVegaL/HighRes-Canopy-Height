# model/ssl_model.py
"""
Módulo de carga de modelos preentrenados para altura de dosel.

Aquí definimos:
- SSLAE: backbone Vision Transformer + head DPT.
- SSLModule: wrapper Lightning que carga los checkpoints de CHM.
- Funciones helper para cargar el modelo de CHM y el modelo de normalización RNet.

Este módulo reemplaza la parte de modelo de inference.py, pero sin CLI ni main().
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl

# Estos imports asumen que copiaste backbone.py, dpt_head.py y regressor.py
# desde el repositorio original a la carpeta `model/`.
from models.backbone import SSLVisionTransformer
from models.dpt_head import DPTHead
from models.regressor import RNet



# --- Rutas base del proyecto -------------------------------------------------

# ssl_model.py vive en STREAMLITANALITICA/model/ssl_model.py
# parents[0] -> .../model
# parents[1] -> raíz del proyecto STREAMLITANALITICA
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "saved_checkpoints"


# --- Definición del modelo de CHM (igual que en inference.py) ----------------

class SSLAE(nn.Module):
    """
    Módulo principal de altura de dosel:
    - Backbone: Vision Transformer preentrenado (SSLVisionTransformer).
    - Decode head: DPTHead para predicción densa (mapa de CHM).

    Si huge=True, usa la configuración ViT-Huge (dim=1280, 32 capas, etc.),
    como en el paper. Si no, usa los defaults de SSLVisionTransformer.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        classify: bool = True,
        n_bins: int = 256,
        huge: bool = False,
    ) -> None:
        super().__init__()

        if huge:
            # Configuración ViT-Huge (modelo grande del paper)
            self.backbone = SSLVisionTransformer(
                embed_dim=1280,
                num_heads=20,
                out_indices=(9, 16, 22, 29),
                depth=32,
                pretrained=pretrained,
            )
            self.decode_head = DPTHead(
                classify=classify,
                in_channels=(1280, 1280, 1280, 1280),
                embed_dims=1280,
                post_process_channels=[160, 320, 640, 1280],
            )
        else:
            # Configuración por defecto (base/large según SSLVisionTransformer)
            self.backbone = SSLVisionTransformer(pretrained=pretrained)
            self.decode_head = DPTHead(classify=classify, n_bins=n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor [B, 3, H, W] con imagen RGB normalizada.
        Devuelve: tensor [B, 1, H, W] con alturas de dosel (antes del reescalado final).
        """
        x = self.backbone(x)
        x = self.decode_head(x)
        return x


class SSLModule(pl.LightningModule):
    """
    Wrapper Lightning que carga el modelo de CHM desde un checkpoint.

    - Si el nombre del checkpoint contiene 'huge', se instancia SSLAE en modo huge.
    - Si el nombre del checkpoint contiene 'compressed', se aplica quantization
      dinámica (int8) para ejecutar eficientemente en CPU y se carga el state_dict
      directamente.
    - Si no es 'compressed', asume un checkpoint de Lightning con clave 'state_dict'.

    En todos los casos, la salida del modelo se multiplica por 10 para llevarla
    a unidades físicas (aprox. metros), como en el código original.
    """

    def __init__(self, ssl_path: str = "compressed_SSLhuge_aerial.pth") -> None:
        super().__init__()

        # Elegimos arquitectura huge o normal según el nombre del checkpoint
        if "huge" in ssl_path:
            self.chm_module_ = SSLAE(classify=True, huge=True).eval()
        else:
            self.chm_module_ = SSLAE(classify=True, huge=False).eval()

        # Carga de pesos
        if "compressed" in ssl_path:
            # Checkpoint cuantizado pensado para CPU
            ckpt = torch.load(ssl_path, map_location="cpu", weights_only=False)
            # Quantization dinámica: Linear, Conv2d, ConvTranspose2d a int8
            self.chm_module_ = torch.quantization.quantize_dynamic(
                self.chm_module_,
                {nn.Linear, nn.Conv2d, nn.ConvTranspose2d},
                dtype=torch.qint8,
            )
            self.chm_module_.load_state_dict(ckpt, strict=False)
        else:
            # Checkpoint "normal" con formato Lightning: ckpt['state_dict']
            ckpt = torch.load(ssl_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["state_dict"]
            self.chm_module_.load_state_dict(state_dict)

        # Escalado final a unidades físicas (10x)
        self.chm_module = lambda x: 10 * self.chm_module_(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor [B, 3, H, W] ya normalizado.
        Devuelve: mapa de altura de dosel en metros aprox. [B, 1, H, W].
        """
        return self.chm_module(x)


# --- Funciones helper para cargar modelos ------------------------------------


def _resolve_checkpoint(name: str) -> Path:
    """
    Resuelve la ruta completa de un checkpoint dentro de saved_checkpoints/.
    """
    ckpt_path = CHECKPOINT_DIR / name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {ckpt_path}")
    return ckpt_path


def load_chm_model(
    checkpoint_name: str = "compressed_SSLhuge_aerial.pth",
    device: Optional[str] = None,
) -> Tuple[SSLModule, str]:
    """
    Carga el modelo de altura de dosel (SSLModule) con el checkpoint indicado.

    Parameters
    ----------
    checkpoint_name:
        Nombre del archivo dentro de saved_checkpoints/.
        Por defecto usamos 'compressed_SSLhuge_aerial.pth', que es el modelo
        ViT-Huge preentrenado y ajustado para imágenes aéreas NEON.
    device:
        Dispositivo donde colocar el modelo. Si es None:
        - Si el nombre contiene 'compressed' -> 'cpu'.
        - En otro caso, 'cuda:0' si hay GPU, si no 'cpu'.

    Returns
    -------
    model: SSLModule ya en eval() y en el dispositivo elegido.
    device: str con el nombre del dispositivo usado.
    """
    ckpt_path = _resolve_checkpoint(checkpoint_name)

    if device is None:
        if "compressed" in checkpoint_name:
            device = "cpu"
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = SSLModule(ssl_path=str(ckpt_path))
    model.to(device)
    model.eval()
    return model, device


def load_rnet_normalizer(
    checkpoint_name: str = "aerial_normalization_quantiles_predictor.ckpt",
    device: str = "cpu",
) -> nn.Module:
    """
    Carga el modelo RNet que predice los cuantiles (p5, p95) por canal para
    normalizar imágenes aéreas al dominio de las imágenes Maxar.

    Es exactamente la misma lógica que en main() de inference.py:
    - Carga el checkpoint de Lightning.
    - Ajusta las claves del state_dict (elimina el prefijo 'backbone.').
    - Instancia RNet(n_classes=6).
    - Carga pesos y pone el modelo en eval().

    Parameters
    ----------
    checkpoint_name:
        Nombre del archivo .ckpt dentro de saved_checkpoints/.
    device:
        Dispositivo donde se colocará RNet. Normalmente 'cpu'.

    Returns
    -------
    model_norm: instancia de RNet en eval() y en el dispositivo indicado.
    """
    ckpt_path = _resolve_checkpoint(checkpoint_name)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]

    # El checkpoint fue entrenado con un módulo 'backbone.' anidado;
    # eliminamos ese prefijo para que coincida con RNet(n_classes=6).
    for k in list(state_dict.keys()):
        if "backbone." in k:
            new_k = k.replace("backbone.", "")
            state_dict[new_k] = state_dict.pop(k)

    model_norm = RNet(n_classes=6)
    model_norm.load_state_dict(state_dict)
    model_norm.to(device)
    model_norm.eval()

    return model_norm
