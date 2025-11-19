# model/neon_data.py
"""
Manejo del dataset NEON aéreo para inferencia y demostración en Streamlit.

Aquí definimos:
- NeonDataset: adaptación de la clase del repositorio original.
- build_neon_dataset: helper para crear el dataset con un normtype dado.
- get_neon_sample: helper para obtener un ejemplo por índice (para el slider).

La normalización por cuantiles puede usar un modelo RNet (model_norm),
que se carga desde ssl_model.load_rnet_normalizer().
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

# Ruta base del proyecto: STREAMLITANALITICA/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "neon_test_data.csv"


class NeonDataset(Dataset):
    """
    Dataset NEON aéreo adaptado del código original.

    Para cada índice entero i, genera un recorte 256x256:
    - 'img': imagen normalizada (lista para entrar al modelo de CHM).
    - 'img_no_norm': imagen original NEON (tensor 3xH xW).
    - 'chm': CHM real correspondiente (tensor 1xH xW).
    - 'lat', 'lon': coordenadas del tile (si están disponibles en el CSV).

    La normalización puede ser:
    - sin normalización (no_norm=True),
    - normalización "vieja" usando imagen Maxar,
    - normalización "nueva" usando un modelo RNet (model_norm).
    """

    def __init__(
        self,
        model_norm: Optional[torch.nn.Module],
        new_norm: bool,
        src_img: str = "neon",
        trained_rgb: bool = False,
        no_norm: bool = False,
    ) -> None:
        super().__init__()

        self.no_norm = no_norm
        self.model_norm = model_norm
        self.new_norm = new_norm
        self.trained_rgb = trained_rgb
        self.size = 256
        self.src_img = src_img

        if not CSV_PATH.exists():
            raise FileNotFoundError(f"No se encontró el archivo CSV: {CSV_PATH}")

        self.df = pd.read_csv(CSV_PATH, index_col=0)

        # Número de recortes horizontales/verticales por imagen grande
        self.size_multiplier = 6

        # Directorio raíz de imágenes NEON / Maxar
        if not IMAGES_DIR.exists():
            raise FileNotFoundError(f"No se encontró el directorio de imágenes: {IMAGES_DIR}")
        self.root_dir = IMAGES_DIR

    def __len__(self) -> int:
        # En el código original, si src_img == 'neon' se amplía 30x el número de filas
        # para cubrir todos los recortes posibles por imagen.
        if self.src_img == "neon":
            return 30 * len(self.df)
        return len(self.df)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        Devuelve un diccionario con:
        - 'img': tensor 3x256x256 normalizado al dominio Maxar (si aplica).
        - 'img_no_norm': tensor 3x256x256 original NEON.
        - 'chm': tensor 1x256x256 con alturas, clippeadas a >= 0.
        - 'lat', 'lon': tensores escalares con coordenadas (si existen).
        """
        n = self.size_multiplier
        ix = i // (n ** 2)
        jx = (i % (n ** 2)) // n
        jy = (i % (n ** 2)) % n

        if self.src_img == "neon":
            l = self.df.iloc[ix]
        else:
            # En este proyecto solo usaremos src_img='neon'
            l = self.df.iloc[ix]

        # Coordenadas del recorte dentro de la imagen grande
        x_list = list(range(l.bord_x, l.imsize - l.bord_x - self.size, self.size))
        y_list = list(range(l.bord_y, l.imsize - l.bord_y - self.size, self.size))
        x = x_list[jx]
        y = y_list[jy]

        # Carga de imagen NEON (o Maxar, según src_img) y CHM
        img_path = self.root_dir / l[self.src_img]
        chm_path = self.root_dir / l.chm

        img = TF.to_tensor(
            Image.open(img_path).crop((x, y, x + self.size, y + self.size))
        )
        chm = TF.to_tensor(
            Image.open(chm_path).crop((x, y, x + self.size, y + self.size))
        )

        # Alturas negativas se llevan a cero
        chm[chm < 0] = 0

        # Normalización de dominio (NEON -> Maxar-like) si aplica
        if not self.trained_rgb and self.src_img == "neon":
            if self.no_norm:
                norm_in = img
            else:
                if self.new_norm:
                    # Normalización automática usando RNet (model_norm)
                    if self.model_norm is None:
                        raise ValueError(
                            "model_norm es None pero new_norm=True. "
                            "Debe proporcionarse un modelo RNet."
                        )
                    x_batch = img.unsqueeze(0)  # [1,3,H,W]
                    norm_img = self.model_norm(x_batch).detach()
                    p5I = [
                        norm_img[0][0].item(),
                        norm_img[0][1].item(),
                        norm_img[0][2].item(),
                    ]
                    p95I = [
                        norm_img[0][3].item(),
                        norm_img[0][4].item(),
                        norm_img[0][5].item(),
                    ]
                else:
                    # Normalización "vieja" usando imagen Maxar co-registrada
                    maxar_path = self.root_dir / l["maxar"]
                    I = TF.to_tensor(
                        Image.open(maxar_path).crop(
                            (x, y, x + self.size, y + self.size)
                        )
                    )
                    p5I = [np.percentile(I[c, :, :].flatten(), 5) for c in range(3)]
                    p95I = [np.percentile(I[c, :, :].flatten(), 95) for c in range(3)]

                # Percentiles de la imagen NEON de entrada
                p5In = [np.percentile(img[c, :, :].flatten(), 5) for c in range(3)]
                p95In = [np.percentile(img[c, :, :].flatten(), 95) for c in range(3)]

                norm_in = img.clone()
                for c in range(3):
                    num = (img[c, :, :] - p5In[c]) * (p95I[c] - p5I[c])
                    den = (p95In[c] - p5In[c]) if (p95In[c] - p5In[c]) != 0 else 1.0
                    norm_in[c, :, :] = num / den + p5I[c]
        else:
            # Caso en que el modelo ya fue entrenado en RGB aéreo
            norm_in = img

        # Coordenadas (si no existen, se reemplazan por 0)
        lat = torch.tensor([getattr(l, "lat", 0.0)]).nan_to_num(0)
        lon = torch.tensor([getattr(l, "lon", 0.0)]).nan_to_num(0)

        return {
            "img": norm_in,
            "img_no_norm": img,
            "chm": chm,
            "lat": lat,
            "lon": lon,
        }


# --- Helpers para usar el dataset en la app ----------------------------------


def build_neon_dataset(
    model_norm: Optional[torch.nn.Module],
    normtype: int = 2,
    trained_rgb: bool = False,
    src_img: str = "neon",
) -> NeonDataset:
    """
    Construye un NeonDataset configurado igual que en evaluate(), según normtype.

    normtype:
        0 -> no_norm=True   (sin normalización de dominio)
        1 -> new_norm=False (normalización "vieja" con Maxar)
        2 -> new_norm=True  (normalización automática con RNet, default del paper)
    """
    new_norm = True
    no_norm = False

    if normtype == 0:
        no_norm = True
    elif normtype == 1:
        new_norm = False
    elif normtype == 2:
        new_norm = True
    else:
        raise ValueError(f"normtype desconocido: {normtype}")

    ds = NeonDataset(
        model_norm=model_norm,
        new_norm=new_norm,
        src_img=src_img,
        trained_rgb=trained_rgb,
        no_norm=no_norm,
    )
    return ds


def get_neon_sample(
    dataset: NeonDataset, index: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Devuelve (img_no_norm, img_norm, chm) para un índice dado del NeonDataset.

    - img_no_norm: tensor 3x256x256 con la imagen NEON original.
    - img_norm: tensor 3x256x256 normalizado para entrar al modelo.
    - chm: tensor 1x256x256 con el CHM real.
    """
    sample = dataset[index]
    img_no_norm = sample["img_no_norm"]
    img_norm = sample["img"]
    chm = sample["chm"]
    return img_no_norm, img_norm, chm
