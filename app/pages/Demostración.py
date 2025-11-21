# -*- coding: utf-8 -*-
import streamlit as st
import random
import base64
from pathlib import Path
import mimetypes
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import io  # <--- ya que estás, pon este import arriba con los demás


# --- AÑADIR RAÍZ DEL PROYECTO AL sys.path ---
ROOT = Path(__file__).resolve().parents[2]  # .../streamlitanalitica
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.inference_neon_tile import setup_neon_inference, run_neon_tile_inference
from model.neon_data import get_neon_sample
from model.ssl_model import load_chm_model
from model.metrics import compute_all_metrics

# ======================
# CONFIG BÁSICA
# ======================
st.set_page_config(
    page_title="CHM • Demostración",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Paths de imágenes de diseño (thumbnails + hero)
BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "imagenes"

def img64(filename: str) -> str:
    """Devuelve una data URI base64 para usar en <img src="...">."""
    file_path = IMG_DIR / filename
    if not file_path.exists():
        return ""
    if filename.lower().endswith(".png"):
        mime = "image/png"
    else:
        mime = "image/jpeg"
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

# Solo para hero (diseño)
THUMB_FILES = [
    "tile1.jpg",
    "tile2.jpg",
    "tile3.jpg",
    "tile4.jpg",
    "tile5.jpg",
    "tile6.png",  # esta es .png
]

# ======================
# ESTADO
# ======================
if "neon_idx" not in st.session_state:
    st.session_state.neon_idx = 0          # índice global NEON (0..N-1 real)
if "mode" not in st.session_state:
    st.session_state.mode = "Modo NEON (dataset)"

# ======================
# FUNCIONES AUXILIARES MODELOS
# ======================

@st.cache_resource
def get_neon_components():
    # Modo NEON: usamos normtype=2 (RNet) y el checkpoint aerial huge comprimido
    return setup_neon_inference(
        checkpoint_name="compressed_SSLhuge_aerial.pth",
        normtype=2,
        trained_rgb=False,
        src_img="neon",
    )

@st.cache_resource
def get_uploaded_components():
    # Cargamos el modelo CHM para el modo de archivos subidos
    model, device = load_chm_model(checkpoint_name="compressed_SSLhuge_aerial.pth")
    norm = T.Normalize(
        mean=(0.420, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return {"model": model, "device": device, "norm": norm}

@st.cache_data(show_spinner=False)
def neon_tile_rgb(idx: int) -> np.ndarray:
    """
    Devuelve la imagen RGB [H,W,3] de un tile NEON (sin normalizar) para vistas previas y thumbnails.
    """
    components = get_neon_components()
    dataset = components["dataset"]
    img_no_norm, _, _ = get_neon_sample(dataset, idx)
    img_rgb = np.moveaxis(img_no_norm.numpy(), 0, 2)  # [H,W,3]
    return img_rgb

def chm_to_rgb(chm_01: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """
    Recibe un CHM normalizado en [0,1] y devuelve una imagen RGB aplicando un colormap.
    """
    cmap = cm.get_cmap(cmap_name)
    chm_rgba = cmap(chm_01)        # [H,W,4] en [0,1]
    chm_rgb = chm_rgba[..., :3]    # nos quedamos con RGB
    return chm_rgb

def np_to_base64(img: np.ndarray) -> str:
    """
    Convierte una imagen numpy [H,W,3] (float 0-1 o uint8) a base64 PNG.
    Solo para mostrar dentro de los cards.
    """
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype("uint8")
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")






# ======================
# CSS GLOBAL + NAVBAR
# ======================
st.markdown(
    """
<style>
/* Fondo gradiente general NUEVO */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0e1624, #0b1220) !important;
}
.block-container {
    padding-top: 0;
    padding-bottom: 3rem;
}

/* Ocultar header y sidebar de Streamlit */
[data-testid="stHeader"]{height:0;visibility:hidden}
[data-testid="stSidebar"] { display:none !important; }
[data-testid="stSidebarNav"] { display:none !important; }

/* FRAME CENTRADO */
.chm-container {
    max-width: 1100px;
    margin: 64px auto 40px;
    padding: 0 16px 40px;
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* TITULOS */
.chm-main-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
}
.chm-main-subtitle {
    font-size: 15px;
    opacity: 0.82;
    margin-bottom: 18px;
}

/* BARRA DE MODOS */
.mode-bar {
    max-width: 720px;
    margin: 18px auto 10px;
    padding: 4px;
    border-radius: 999px;
    background: #020617;
    box-shadow: 0 12px 30px rgba(0,0,0,0.6);
    border: 1px solid #111827;
}
.mode-bar div[data-testid="stRadio"] > div {
    display: flex;
    justify-content: space-between;
    width: 100%;
}
.mode-bar div[data-testid="stRadio"] label {
    cursor: pointer;
    padding: 10px 22px;
    border-radius: 999px;
    border: 1px solid transparent;
    background: #0b1120;
    color: #9ca3af;
    font-size: 14px;
    margin: 0 4px;
    box-shadow: none;
    transition: all 0.18s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    flex: 1;
}
.mode-bar div[data-testid="stRadio"] input[type="radio"] {
    display: none;
}
.mode-bar div[data-testid="stRadio"] label:hover {
    background: #111827;
    color: #e5e7eb;
}
.mode-bar div[data-testid="stRadio"] label:has(input:checked) {
    background: #1d4ed8;
    border-color: #3b82f6;
    box-shadow: 0 0 20px rgba(37,99,235,0.7);
    color: #f9fafb;
}

/* HERO CARD (azul sólido, sin degradado raro) */
.hero-card {
    background: #0b1f3b;
    border-radius: 18px;
    padding: 24px 28px;
    margin-top: 18px;
    margin-bottom: 32px;
    box-shadow: 0 20px 45px rgba(0,0,0,0.55);
    border: 1px solid #2563eb;
}
.hero-inner {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
    gap: 24px;
}
.hero-left {
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.hero-icon {
    width: 36px;
    height: 36px;
    border-radius: 999px;
    background: #1d4ed8;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 0 12px #1d4ed877;
}
.hero-title {
    font-size: 24px;
    font-weight: 700;
}
.hero-text {
    font-size: 14px;
    opacity: 0.9;
    line-height: 1.5;
}
.hero-buttons {
    display: flex;
    gap: 12px;
    margin-top: 8px;
}
.hero-btn-primary,
.hero-btn-secondary {
    padding: 10px 18px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
}
.hero-btn-primary {
    border: none;
    background: #2563eb;
    color: white;
    box-shadow: 0 10px 25px #1d4ed855;
}
.hero-btn-secondary {
    border: 1px solid #4b5563;
    background: transparent;
    color: #e5e7eb;
}
.hero-right {
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    background: #020617;
    border: 1px solid #111827;
}
.hero-right img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.hero-caption {
    position: absolute;
    right: 12px;
    bottom: 10px;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(15,23,42,0.85);
    border: 1px solid #1f2937;
}

/* SECCIONES */
.section-title {
    font-size: 20px;
    margin: 18px 0 12px;
}

/* TEXTO DEL ÍNDICE NEON - AZULITO */
.neon-info-main {
    font-size: 24px;
    color: #38bdf8;   /* azul */
    font-weight: 600;
    margin: 16px 0 4px;
}
.neon-info-sub {
    font-size: 18px;
    color: #38bdf8;   /* azul */
    margin-bottom: 4px;
}

/* Number input dark */
div[data-testid="stNumberInput"] {
    margin-top: 4px;
}
div[data-testid="stNumberInput"] > label {
    font-size: 12px;
    margin-bottom: 6px;
    color: #e5e7eb;
}
div[data-testid="stNumberInput"] input {
    background: #020617;
    border-radius: 12px;
    border: 1px solid #111827;
    color: #e5e7eb;
}

/* THUMBNAILS (NEON reales) */
.thumb-card {
    width: 100%;
    background: #020617;
    padding: 6px;
    border-radius: 12px;
    text-align: center;
    border: 2px solid transparent;
    box-shadow: 0 6px 15px rgba(0,0,0,0.4);
    transition: 0.18s transform, 0.18s box-shadow, 0.18s border-color;
}
.thumb-card img {
    width: 100%;
    height: 80px;
    border-radius: 8px;
    object-fit: cover;
}
.thumb-label {
    font-size: 12px;
    margin-top: 6px;
    opacity: 0.8;
}
.thumb-active {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(37,99,235,0.55);
    border-color: #3b82f6;
}

/* Botones prev/next/random (contenedor para personalizar solo estos) */
.tile-nav-buttons button {
    width: 100%;
    background: #0f172a;
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid #1f2937;
    cursor: pointer;
    color: white;
    font-size: 13px;
}
.tile-nav-buttons button:hover {
    background: #1f2937;
}

/* PREVIEW (un poco más grande y SIEMPRE dentro del card) */
.preview-card {
    background: linear-gradient(135deg, #0b1f3b, #020617);
    padding: 16px;
    border-radius: 16px;
    border: 1px solid #2563eb;
}
.preview-title {
    margin-bottom: 10px;
    font-size: 15px;
    font-weight: 600;
}
.preview-card [data-testid="stImage"] img,
.preview-card img {
    display: block;
    margin: 0 auto;
    max-height: 360px;
    width: auto !important;
    max-width: 100%;
    object-fit: contain;
}

/* BOTÓN INFERENCIA */
.infer-wrapper {
    text-align: center;
    margin-top: 8px;
    margin-bottom: 26px;
}
.infer-wrapper button {
    background: #0284c7;
    color: white;
    padding: 12px 26px;
    border-radius: 999px;
    border: none;
    cursor: pointer;
    font-weight: 600;
    font-size: 15px;
    box-shadow: 0 0 18px #0ea5e955;
}
.infer-wrapper button:hover {
    background:#0ea5e9;
}

/* RESULTADOS (cards azules lindos) */
.result-card {
    background: linear-gradient(135deg, #0b1f3b, #020617);
    padding: 14px;
    border-radius: 16px;
    border: 1px solid #2563eb;
    box-shadow: 0 15px 35px rgba(0,0,0,0.55);
}
.result-title {
    margin-bottom: 8px;
    font-size: 14px;
    font-weight: 600;
}
.result-card img {
    width: 100%;
    border-radius: 10px;
    object-fit: contain;
}

/* METRICAS (dos filas de cards) */
.metric-card {
    background: linear-gradient(135deg, #0b1f3b, #020617);
    padding: 16px 18px;
    border-radius: 16px;
    border: 1px solid #2563eb;
    box-shadow: 0 12px 30px rgba(0,0,0,0.4);
}
.metric-title {
    font-size: 13px;
    opacity: 0.9;
    margin-bottom: 6px;
}
.metric-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
}
.metric-change-pos {
    font-size: 12px;
    color: #4ade80;
}
.metric-change-neg {
    font-size: 12px;
    color: #f97373;
}
.metric-sub {
    font-size: 12px;
    opacity: 0.7;
}

.chart-card {
    background: #020617;              /* más oscuro para que se note el card */
    padding: 14px 16px;
    border-radius: 16px;
    border: 1px solid #2563eb;
    box-shadow: 0 12px 28px rgba(0,0,0,0.6);
}

.chart-title {
    font-size: 14px;
    margin-bottom: 8px;
    font-weight: 600;
}

/* wrapper interno para que la figura no pegue contra el borde */
.chart-inner {
    padding: 6px 4px 4px;
    border-radius: 12px;
    background: #020617;
}


/* NAV SUPERIOR */
.topbar{position:fixed; inset:0 0 auto 0; z-index:9999; width:100vw; margin-left:calc(50% - 50vw);
  display:flex; align-items:center; justify-content:space-between; gap:12px; padding:10px 18px;
  background:linear-gradient(180deg, rgba(14,22,36,.95), rgba(11,18,32,.5)); backdrop-filter: blur(6px);
  border-bottom: 1px solid rgba(255,255,255,.06)}
.scrolled .topbar{ background:rgba(15,23,42,.98); box-shadow:0 4px 18px rgba(0,0,0,.35); }
.brand{display:flex; align-items:center; gap:10px; font-weight:800}
.brand .logo{width:28px;height:28px;border-radius:8px;display:grid;place-items:center;background:#0b2b3d;color:#8bd3ff;border:1px solid rgba(140,210,255,.35)}
.nav{display:flex; gap:10px}
.nav .btn{padding:10px 14px;border-radius:999px;font-weight:700;border:1px solid rgba(255,255,255,.10); text-decoration:none;color:#e5e7eb}
.nav .btn:hover{border-color:rgba(255,255,255,.22)}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- NAV HTML ----------
st.markdown(
    """
<div class="topbar" id="nav">
  <div class="brand"><div class="logo">🌳</div><span>CHM</span></div>
  <div class="nav">
    <a href="/streamlit_landing_CHM_app" class="btn">Inicio</a>
    <a href="/pages/1_Metodología" class="btn">Metodología</a>
    <a href="#" class="btn">Demostración</a>
  </div>
</div>
<script>
(function(){
  const nav = document.getElementById('nav');
  function onScroll(){
    const sc = window.scrollY || document.documentElement.scrollTop || 0;
    document.documentElement.classList.toggle('scrolled', sc > 24);
  }
  window.addEventListener('scroll', onScroll, {passive:true});
  onScroll();
})();
</script>
""",
    unsafe_allow_html=True,
)

# ======================
# CONFIG POR MODO (para hero)
# ======================
mode_configs = {
    "Modo NEON (dataset)": {
        "icon": "🛰️",
        "title": "Modo NEON (dataset)",
        "desc": (
            "Explora tiles reales del dataset NEON usado en el paper. Cada índice "
            "corresponde a una imagen aérea de alta resolución con su CHM de referencia "
            "derivado de LiDAR."
        ),
        "primary": "Ver mosaico de tiles",
        "secondary": "Más información del dataset",
        "hero_img": "tile1.jpg",
        "caption": "Muestra aérea de un tile NEON.",
    },
    "Modo de imagen subida": {
        "icon": "🖼️",
        "title": "Modo de imagen subida",
        "desc": (
            "Sube tu propia imagen RGB (y opcionalmente un CHM real) para probar cómo se "
            "comporta el modelo fuera del dataset original y explorar posibles dominios nuevos."
        ),
        "primary": "Subir imagen RGB",
        "secondary": "Ver requisitos de la imagen",
        "hero_img": "tile2.jpg",
        "caption": "Ejemplo ilustrativo de imagen de entrada.",
    },
}

# ======================
# CONTENIDO
# ======================
st.markdown('<div class="chm-container">', unsafe_allow_html=True)

# HEADER
st.markdown(
    """
<div class="chm-main-title">Demostración interactiva del modelo CHM</div>
<div class="chm-main-subtitle">
Exploración visual del modelo basado en Transformers (DINOv2 + DPT) para estimar la altura del dosel a partir de imágenes aéreas de alta resolución.
</div>
""",
    unsafe_allow_html=True,
)

# BARRA DE MODOS
st.markdown('<div class="mode-bar">', unsafe_allow_html=True)
mode = st.radio(
    "",
    list(mode_configs.keys()),
    index=list(mode_configs.keys()).index(st.session_state.mode),
    horizontal=True,
)
st.markdown("</div>", unsafe_allow_html=True)
st.session_state.mode = mode
cfg = mode_configs[mode]

# HERO CARD
hero_src = img64(cfg["hero_img"])
hero_html = f"""
<div class="hero-card">
  <div class="hero-inner">
    <div class="hero-left">
      <div class="hero-icon">{cfg['icon']}</div>
      <div class="hero-title">{cfg['title']}</div>
      <div class="hero-text">{cfg['desc']}</div>
      <div class="hero-buttons">
        <button class="hero-btn-primary">{cfg['primary']}</button>
        <button class="hero-btn-secondary">{cfg['secondary']}</button>
      </div>
    </div>
    <div class="hero-right">
      <img src="{hero_src}" alt="{cfg['title']}">
      <div class="hero-caption">{cfg['caption']}</div>
    </div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# =====================================================
# MODO 1: NEON (DATASET)
# =====================================================
if mode == "Modo NEON (dataset)":
    components = get_neon_components()
    dataset = components["dataset"]
    dataset_len = len(dataset)
    total_tiles = dataset_len

    # --- NAVEGADOR DE TILES ---
    st.markdown('<div class="section-title">Navegador de tiles</div>', unsafe_allow_html=True)
    cols_nav = st.columns([2, 3])

    # Thumbnails NEON reales a la izquierda
    with cols_nav[0]:
        THUMB_COUNT = 6
        if total_tiles >= THUMB_COUNT:
            step = max(1, total_tiles // THUMB_COUNT)
            thumb_indices = [i * step for i in range(THUMB_COUNT)]
        else:
            thumb_indices = list(range(total_tiles))

        thumb_cols = st.columns(len(thumb_indices))
        for idx_thumb, col in zip(thumb_indices, thumb_cols):
            src_img = neon_tile_rgb(int(idx_thumb))
            active = (
                abs(int(st.session_state.neon_idx) - idx_thumb) < step // 2
                if total_tiles >= THUMB_COUNT
                else False
            )
            active_cls = "thumb-card thumb-active" if active else "thumb-card"
            with col:
                st.markdown(f'<div class="{active_cls}">', unsafe_allow_html=True)
                st.image(src_img)
                st.markdown(f'<div class="thumb-label">Tile #{idx_thumb}</div></div>', unsafe_allow_html=True)

        # Botones de navegación (mueven el índice NEON)
        st.markdown('<div class="tile-nav-buttons">', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("⬅️ Anterior", key="prev"):
                st.session_state.neon_idx = max(0, int(st.session_state.neon_idx) - 1)
        with b2:
            if st.button("Siguiente ➡️", key="next"):
                st.session_state.neon_idx = min(total_tiles - 1, int(st.session_state.neon_idx) + 1)
        with b3:
            if st.button("🔄 Aleatorio", key="rand"):
                st.session_state.neon_idx = random.randint(0, total_tiles - 1)
        st.markdown("</div>", unsafe_allow_html=True)

        # Texto + number_input para el índice NEON
        st.markdown(
            f'<div class="neon-info-main">Total de tiles virtuales: {total_tiles}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="neon-info-sub">Escribe un índice entre 0 y {total_tiles-1}.</div>',
            unsafe_allow_html=True,
        )

        neon_idx = st.number_input(
            "Índice del tile NEON",
            min_value=0,
            max_value=total_tiles - 1,
            value=int(st.session_state.neon_idx),
            step=1,
        )
        st.session_state.neon_idx = int(neon_idx)

    # Preview a la derecha (usa el tile real del dataset, dentro del card)
    with cols_nav[1]:
        try:
            img_prev_rgb = neon_tile_rgb(int(st.session_state.neon_idx))
            b64_prev = np_to_base64(img_prev_rgb)
            idx_txt = int(st.session_state.neon_idx)
            preview_html = f"""
            <div class="preview-card">
              <div class="preview-title">Vista previa del tile seleccionado</div>
              <img src="data:image/png;base64,{b64_prev}" alt="Tile NEON índice {idx_txt}">
              <div class="thumb-label" style="margin-top:6px;">Tile NEON índice {idx_txt}</div>
            </div>
             """
            st.markdown(preview_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"No se pudo mostrar la vista previa del tile: {e}")
            



    # Botón de inferencia
    st.markdown('<div class="infer-wrapper">', unsafe_allow_html=True)
    run_neon_btn = st.button("⚡ Calcular CHM para este tile", key="run_neon")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- RESULTADOS DE INFERENCIA ---
    if run_neon_btn:
        idx = int(st.session_state.neon_idx)
        result = run_neon_tile_inference(components, idx)

        img_rgb = result["img_rgb"]
        chm_gt = result["chm_gt"]
        chm_pred = result["chm_pred"]
        metrics = result["metrics"]

        # Normalización para visualización en color
        vmin = float(min(chm_gt.min(), chm_pred.min()))
        vmax = float(max(chm_gt.max(), chm_pred.max()))
        eps = 1e-6

        chm_pred_vis = (chm_pred - vmin) / (vmax - vmin + eps)
        chm_pred_vis = np.clip(chm_pred_vis, 0.0, 1.0)
        chm_gt_vis = (chm_gt - vmin) / (vmax - vmin + eps)
        chm_gt_vis = np.clip(chm_gt_vis, 0.0, 1.0)

        chm_pred_rgb = chm_to_rgb(chm_pred_vis, cmap_name="viridis")
        chm_gt_rgb = chm_to_rgb(chm_gt_vis, cmap_name="viridis")

        # ---- IMÁGENES ----
        st.markdown('<div class="section-title">Resultados de la inferencia</div>', unsafe_allow_html=True)
        cols_res = st.columns(3)
        titles_res = ["Imagen aérea (RGB)", "CHM predicho (m)", "CHM real (LiDAR, m)"]
        imgs_res = [img_rgb, chm_pred_rgb, chm_gt_rgb]

        for col, title, img in zip(cols_res, titles_res, imgs_res):
            b64_img = np_to_base64(img)
            card_html = f"""
            <div class="result-card">
              <div class="result-title">{title}</div>
              <img src="data:image/png;base64,{b64_img}" alt="{title}">
            </div>
            """
            with col:
              st.markdown(card_html, unsafe_allow_html=True)

        # ---- MÉTRICAS: 2 filas (paper + censo) ----
        st.markdown('<div class="section-title">Métricas científicas y de censo estructural</div>', unsafe_allow_html=True)

        # Censo estructural simple sobre CHM predicho
        mask_trees = chm_pred > 1.0
        if mask_trees.any():
            mean_h_pred = float(chm_pred[mask_trees].mean())
            p95_pred = float(np.percentile(chm_pred[mask_trees], 95))
        else:
            mean_h_pred = 0.0
            p95_pred = 0.0
        forest_pct = float(mask_trees.mean() * 100.0)

        metrics_paper = [
            {
                "title": "MAE",
                "value": f"{metrics['mae']:.2f} m",
                "subtitle": "Error medio absoluto.",
            },
            {
                "title": "RMSE",
                "value": f"{metrics['rmse']:.2f} m",
                "subtitle": "Error cuadrático medio.",
            },
            {
                "title": "R² (pixel)",
                "value": f"{metrics['r2']:.3f}",
                "subtitle": "Ajuste pixel a pixel.",
            },
            {
                "title": "R² (bloques)",
                "value": f"{metrics['r2_block']:.3f}",
                "subtitle": "Ajuste en bloques 50×50.",
            },
        ]

        metrics_censo = [
            {
                "title": "Bias",
                "value": f"{metrics['bias']:.2f} m",
                "subtitle": "Sesgo medio de la predicción.",
            },
            {
                "title": "Altura promedio del dosel",
                "value": f"{mean_h_pred:.1f} m",
                "subtitle": "Promedio de píxeles con h > 1 m.",
            },
            {
                "title": "Altura p95",
                "value": f"{p95_pred:.1f} m",
                "subtitle": "Altura típica de las copas más altas.",
            },
            {
                "title": "% área con árboles",
                "value": f"{forest_pct:.0f} %",
                "subtitle": "Proporción de píxeles con h > 1 m.",
            },
        ]

        # Fila 1: métricas del paper
        cols_met1 = st.columns(4)
        for col, m in zip(cols_met1, metrics_paper):
            card_html = f"""
            <div class="metric-card">
              <div class="metric-title">{m['title']}</div>
              <div class="metric-row">
                <span class="metric-value">{m['value']}</span>
              </div>
              <div class="metric-sub">{m['subtitle']}</div>
            </div>
            """
            with col:
                st.markdown(card_html, unsafe_allow_html=True)

        # Fila 2: métricas nuevas tipo censo
        cols_met2 = st.columns(4)
        for col, m in zip(cols_met2, metrics_censo):
            card_html = f"""
            <div class="metric-card">
              <div class="metric-title">{m['title']}</div>
              <div class="metric-row">
                <span class="metric-value">{m['value']}</span>
              </div>
              <div class="metric-sub">{m['subtitle']}</div>
            </div>
            """
            with col:
                st.markdown(card_html, unsafe_allow_html=True)

        # ---- RESUMEN ESTRUCTURAL (GRÁFICOS) ----
        st.markdown('<div class="section-title">Resumen estructural del bosque en este tile</div>', unsafe_allow_html=True)
        cols_ch = st.columns([3, 2])

        # Histograma de alturas predichas
        with cols_ch[0]:
            st.markdown(
            '<div class="chart-card">'
            '<div class="chart-title">Distribución de alturas (CHM predicho)</div>'
            '<div class="chart-inner">',
            unsafe_allow_html=True,
            )

            fig, ax = plt.subplots(figsize=(3.2, 2.1))  # más pequeño
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")

            ax.hist(chm_pred.flatten(), bins=20, color="#3b82f6", edgecolor="#020617")
            ax.set_xlabel("Altura (m)", color="#e5e7eb", fontsize=9)
            ax.set_ylabel("Número de píxeles", color="#e5e7eb", fontsize=9)

            ax.tick_params(colors="#e5e7eb", labelsize=8)
            for spine in ax.spines.values():
             spine.set_color("#4b5563")

            ax.grid(alpha=0.25, color="#4b5563")
            plt.tight_layout()

            # IMPORTANTE: sin use_container_width para que quede centrado dentro del card
            st.pyplot(fig, use_container_width=False, transparent=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        # Barras simple: bosque vs no bosque
        with cols_ch[1]:
            st.markdown(
            '<div class="chart-card">'
            '<div class="chart-title">Área con y sin árboles</div>'
            '<div class="chart-inner">',
            unsafe_allow_html=True,
            )

            fig2, ax2 = plt.subplots(figsize=(2.6, 2.1))
            fig2.patch.set_alpha(0.0)
            ax2.set_facecolor("none")

            bosque = forest_pct
            no_bosque = 100.0 - forest_pct
            ax2.bar(["Bosque", "Sin bosque"], [bosque, no_bosque], color="#3b82f6")

            ax2.set_ylim(0, 100)
            ax2.set_ylabel("% del tile", color="#e5e7eb", fontsize=9)

            ax2.tick_params(colors="#e5e7eb", labelsize=8)
            for spine in ax2.spines.values():
             spine.set_color("#4b5563")

            ax2.grid(alpha=0.25, color="#4b5563", axis="y")
            plt.tight_layout()

            st.pyplot(fig2, use_container_width=False, transparent=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    else:
        st.info(
            "Elige un índice y pulsa **«⚡ Calcular CHM para este tile»** para ver las predicciones.",
            icon="👆",
        )

# =====================================================
# MODO 2: IMAGEN SUBIDA
# =====================================================
else:
    components_up = get_uploaded_components()

    st.markdown(
        """
<div class="section-title">Modo de imagen subida</div>
<p style="font-size:14px;opacity:0.85;">
En este modo puedes subir una <strong>imagen RGB</strong> y, opcionalmente,
un <strong>CHM real</strong> co-registrado. El modelo generará un mapa de altura
del dosel para la imagen y, si proporcionas CHM real, se calcularán las métricas
de error igual que en el paper.
</p>
<p style="font-size:13px;opacity:0.8;">
Recomendación: usar recortes 256×256 píxeles de imágenes similares a NEON
(aéreas, alta resolución, con tres bandas RGB). Si subes también el CHM real,
debe tener el <strong>mismo tamaño</strong> que la imagen RGB.
</p>
""",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        rgb_file = st.file_uploader(
            "📁 Subir imagen RGB",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            help="Imagen aérea RGB. Idealmente un recorte 256×256 similar a los tiles NEON.",
            key="rgb_upload",
        )
        chm_file = st.file_uploader(
            "📁 (Opcional) Subir CHM real",
            type=["tif", "tiff", "png"],
            help="Mapa de altura de dosel real, co-registrado con la imagen RGB.",
            key="chm_upload",
        )

        run_btn = st.button("Ejecutar inferencia", type="primary")

    with col_right:
        st.markdown(
            """
<div class="chart-card">
  <div class="chart-title">Notas de validación</div>
  <p style="font-size:13px;opacity:0.85;">
  • La imagen debe ser <strong>RGB (3 canales)</strong>.<br>
  • El CHM (si se sube) debe tener exactamente el mismo tamaño que la imagen.<br>
  • Imágenes muy grandes se redimensionan automáticamente para evitar errores de memoria.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

    # --- LÓGICA DEL BOTÓN ---
    if run_btn:
        if rgb_file is None:
            st.warning(
                "Por favor sube al menos una imagen RGB antes de ejecutar la inferencia.",
                icon="⚠️",
            )
        else:
            import traceback
            with st.spinner("Ejecutando inferencia sobre la imagen subida..."):
                try:
                    model = components_up["model"]
                    device = components_up["device"]
                    norm = components_up["norm"]

                    # --- Preparar imagen RGB ---
                    rgb_img = Image.open(rgb_file).convert("RGB")
                    img_np = np.array(rgb_img).astype("float32") / 255.0  # [H,W,3] en [0,1]

                    # === Restricciones de calidad y tamaño ===
                    if img_np.ndim != 3:
                        raise ValueError("La imagen debe tener 3 canales (formato RGB).")

                    H0, W0, C = img_np.shape
                    if C != 3:
                        raise ValueError(f"La imagen debe tener 3 canales RGB, se encontró {C}.")

                    # Muy pequeña -> rechazamos
                    if H0 < 128 or W0 < 128:
                        raise ValueError(
                            f"La imagen es muy pequeña ({H0}×{W0} píxeles). "
                            "Usa recortes de al menos 256×256 píxeles similares a los tiles NEON."
                        )

                    # Muy grande -> redimensionamos para evitar OOM
                    MAX_SIDE = 512  # máximo 512 px por lado
                    resized_msg = None
                    if max(H0, W0) > MAX_SIDE:
                        scale = MAX_SIDE / max(H0, W0)
                        newW = int(W0 * scale)
                        newH = int(H0 * scale)

                        rgb_img_resized = rgb_img.resize((newW, newH), Image.BILINEAR)
                        img_np = np.array(rgb_img_resized).astype("float32") / 255.0
                        resized_msg = (
                            f"La imagen original era de {H0}×{W0} píxeles "
                            f"y se redimensionó automáticamente a {newH}×{newW} "
                            "para poder ejecutar el modelo sin quedarse sin memoria."
                        )

                    # Guardamos esta versión (posible resize) para mostrar
                    img_rgb_up = img_np
                    H, W, _ = img_rgb_up.shape  # tamaño final usado por el modelo

                    # --- Tensor para el modelo ---
                    img_t = torch.from_numpy(img_rgb_up).permute(2, 0, 1)  # [3,H,W]
                    x = img_t.unsqueeze(0)  # [1,3,H,W]
                    x = norm(x).to(device)

                    model.eval()
                    with torch.no_grad():
                        pred = model(x)
                        pred = pred.cpu().relu()[0, 0].numpy()  # [H,W]
                    chm_pred_up = pred

                    chm_gt_up = None
                    metrics_up = None

                    # --- Si hay CHM real, cargar y calcular métricas ---
                    # --- Si hay CHM real, cargar y (si hace falta) redimensionar y calcular métricas ---
                    if chm_file is not None:
                        # Abrimos el CHM original
                        chm_img = Image.open(chm_file)
                        chm_arr = np.array(chm_img).astype("float32")

                        # Si viene con varios canales, nos quedamos con uno
                        if chm_arr.ndim == 3:
                            chm_arr = chm_arr[..., 0]

                        # Tamaño de la predicción del modelo
                        H_pred, W_pred = chm_pred_up.shape

                        # Si el tamaño del CHM NO coincide, lo redimensionamos al tamaño de la predicción
                        if chm_arr.shape != (H_pred, W_pred):
                            # Redimensionamos usando PIL para que coincida con la grilla del modelo
                            chm_resized = chm_img.resize((W_pred, H_pred), resample=Image.BILINEAR)
                            chm_arr = np.array(chm_resized).astype("float32")

                            if chm_arr.ndim == 3:
                                chm_arr = chm_arr[..., 0]

                            st.info(
                                f"El CHM real original tenía tamaño {chm_img.size[1]}×{chm_img.size[0]} "
                                f"y se redimensionó a {H_pred}×{W_pred} para poder compararlo con la predicción.",
                                icon="ℹ️",
                            )

                        # Ahora sí: ya deberían coincidir las dimensiones
                        if chm_arr.shape != (H_pred, W_pred):
                            # Último chequeo por si acaso algo raro
                            st.warning(
                                f"Aún hay diferencias de tamaño entre la predicción {chm_pred_up.shape} "
                                f"y el CHM real {chm_arr.shape}. No se calcularán métricas.",
                                icon="⚠️",
                            )
                            chm_gt_up = None
                            metrics_up = None
                        else:
                            chm_gt_up = chm_arr
                            metrics_up = compute_all_metrics(chm_pred_up, chm_gt_up)

                    # --- Visualización ---
                    st.markdown('<div class="section-title">Resultado de la inferencia</div>', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown('<div class="result-card"><div class="result-title">Imagen RGB usada por el modelo</div>', unsafe_allow_html=True)
                        st.image(img_rgb_up)
                        if resized_msg is not None:
                            st.caption(resized_msg)
                        else:
                            st.caption(f"Tamaño de inferencia: {H}×{W} píxeles.")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Rango de alturas para normalizar colormap
                    if chm_gt_up is not None:
                        vmin_up = float(min(chm_gt_up.min(), chm_pred_up.min()))
                        vmax_up = float(max(chm_gt_up.max(), chm_pred_up.max()))
                    else:
                        vmin_up = float(chm_pred_up.min())
                        vmax_up = float(chm_pred_up.max())

                    eps = 1e-6

                    chm_pred_vis_up = (chm_pred_up - vmin_up) / (vmax_up - vmin_up + eps)
                    chm_pred_vis_up = np.clip(chm_pred_vis_up, 0.0, 1.0)
                    chm_pred_rgb_up = chm_to_rgb(chm_pred_vis_up, cmap_name="viridis")

                    with c2:
                        st.markdown('<div class="result-card"><div class="result-title">CHM predicho (m)</div>', unsafe_allow_html=True)
                        st.image(chm_pred_rgb_up)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with c3:
                        if chm_gt_up is not None:
                            chm_gt_vis_up = (chm_gt_up - vmin_up) / (vmax_up - vmin_up + eps)
                            chm_gt_vis_up = np.clip(chm_gt_vis_up, 0.0, 1.0)
                            chm_gt_rgb_up = chm_to_rgb(chm_gt_vis_up, cmap_name="viridis")
                            st.markdown('<div class="result-card"><div class="result-title">CHM real (m)</div>', unsafe_allow_html=True)
                            st.image(chm_gt_rgb_up)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-card"><div class="result-title">CHM real (no disponible)</div>', unsafe_allow_html=True)
                            st.info(
                                "No se cargó un CHM real, por lo que solo se muestra la predicción.",
                                icon="ℹ️",
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    # --- Métricas si hay CHM real ---
                    if metrics_up is not None:
                        st.markdown('<div class="section-title">Métricas (comparación con CHM real)</div>', unsafe_allow_html=True)
                        import pandas as pd

                        df_metrics_up = pd.DataFrame(
                            {
                                "Métrica": ["MAE", "RMSE", "R² (pixel)", "R² (bloques)", "Bias"],
                                "Valor": [
                                    f"{metrics_up['mae']:.3f} m",
                                    f"{metrics_up['rmse']:.3f} m",
                                    f"{metrics_up['r2']:.3f}",
                                    f"{metrics_up['r2_block']:.3f}",
                                    f"{metrics_up['bias']:.3f} m",
                                ],
                            }
                        )
                        st.table(df_metrics_up)
                    else:
                        st.info(
                            "No se calcularon métricas porque no se proporcionó un CHM real.",
                            icon="ℹ️",
                        )

                except Exception as e:
                    st.error(
                        f"Ocurrió un error al procesar los archivos: {e}",
                    )
                    # Para ver la traza completa durante las pruebas
                    st.code(traceback.format_exc())
    # (si run_btn es False, no hacemos nada extra aquí)

# CIERRE DEL CONTENEDOR
st.markdown("</div>", unsafe_allow_html=True)
