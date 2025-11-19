# app/streamlit_landing_CHM_app.py
import streamlit as st
from pathlib import Path
from base64 import b64encode
import mimetypes
from streamlit.components.v1 import html as st_html

st.set_page_config(
    page_title="CHM • Landing",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="collapsed",  # colapsa sidebar por si acaso
)

# === Ocultar por completo la sidebar y eliminar el hueco izquierdo ===
st.markdown("""
<style>
/* 1) Oculta la sidebar y el botón de menú */
section[data-testid="stSidebar"]{ display:none !important; }
div[aria-label="Main menu"]{ display:none !important; }

/* 2) Quita cualquier desplazamiento/margen residual de Streamlit */
[data-testid="stAppViewContainer"]{ margin-left:0 !important; }
[data-testid="stAppViewContainer"] > .main{ margin-left:0 !important; }

/* 3) Usa flex para empujar el footer al fondo de la pantalla */
html, body, .stApp { height: 100%; }
[data-testid="stAppViewContainer"]{
  display:flex; flex-direction:column; min-height:100%;
}
.content-grow{ flex:1 0 auto; }   /* contenido crece y empuja el footer */
footer.site-footer{ flex-shrink:0; }

/* (No tocar navbar) */
</style>
""", unsafe_allow_html=True)

# ---------- Utilidades ----------
ASSETS = Path(__file__).parent / "assets"

def data_uri(p: Path, fallback: str = "") -> str:
    if not p.exists():
        return fallback
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"
    return f"data:{mime};base64,{b64encode(p.read_bytes()).decode()}"

# Imágenes
HERO    = data_uri(ASSETS / "hero.webp")
GLOBO   = data_uri(ASSETS / "globo.webp")
OV1     = data_uri(ASSETS / "overlay-buscador.png")
OV2     = data_uri(ASSETS / "overlay-fecha.png")
LISTADO = data_uri(ASSETS / "listado-thumbs.png")
STEP1   = data_uri(ASSETS / "step1.png")
STEP2   = data_uri(ASSETS / "step2.png")
STEP3   = data_uri(ASSETS / "step3.png")
STEP4   = data_uri(ASSETS / "step4.png")

# ---------- CSS Global ----------
st.markdown(
    """
<style>
:root{
  --bg:#0b1220; --panel:#0e1624; --fg:#eaf2ff; --muted:#b8c7d8; --ok:#22d3ee; --ok2:#34d399; --cta:#f97316;
  --ring:#65b2ff; --panel-2:#101a2a; --panel-3:#162131; --border:rgba(255,255,255,.10); --border-soft:rgba(255,255,255,.06);
  --nav-h: 56px;
}

/* Fondo unificado */
html, body {
  background: var(--panel) !important; color: var(--fg);
  font-family: ui-sans-serif,system-ui,-apple-system,Inter,Segoe UI,Roboto,sans-serif;
  padding-top: var(--nav-h);
}
#root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main,
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"], .block-container {
  background: var(--panel) !important;
}

[data-testid="stHeader"]{height:0;visibility:hidden}
.block-container{padding-top:0 !important; padding-left:0 !important; padding-right:0 !important}
*{scroll-behavior:smooth}

/* Ajuste para anclas: que no queden debajo del nav fijo */
[id] { scroll-margin-top: calc(var(--nav-h) + 12px); }

/* ===== NAVBAR FIJO ===== */
.topbar{
  position:fixed; inset:0 0 auto 0; z-index:9999; width:100vw; margin-left:calc(50% - 50vw);
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding:10px 18px; transition:background .25s, box-shadow .25s, backdrop-filter .25s;
  background:linear-gradient(180deg, rgba(14,22,36,.65), rgba(14,22,36,.25));
  backdrop-filter: blur(6px);
  border-bottom: 1px solid rgba(255,255,255,.06);
}
.scrolled .topbar{ background:rgba(14,22,36,.92); box-shadow:0 4px 18px rgba(0,0,0,.35); }
.brand{display:flex; align-items:center; gap:10px; font-weight:800}
.brand .logo{width:28px;height:28px;border-radius:8px;display:grid;place-items:center;background:#0b2b3d;color:#8bd3ff;border:1px solid rgba(140,210,255,.35)}
.brand span{opacity:.9}
.nav{display:flex; gap:10px}
.nav .btn{padding:10px 14px;border-radius:999px;font-weight:700;border:1px solid var(--border); text-decoration:none;color:var(--fg)}
.nav .btn:hover{border-color:rgba(255,255,255,.22)}

/* ===== Bandas ===== */
.full-bleed{width:100vw;margin-left:calc(50% - 50vw);margin-right:calc(50% - 50vw)}
.band{
  width:100vw; margin-left:calc(50% - 50vw); margin-right:calc(50% - 50vw);
  background:var(--panel) !important; border-top:none !important; border-bottom:none !important;
  padding:28px 0; position:relative;
}

/* ===== HERO ===== */
.hero{
  position:relative;display:grid;grid-template-columns:0.95fr 1.05fr;min-height:560px;
  background:var(--panel); overflow:hidden;
}
.hero-left{padding:48px;display:flex;align-items:center;z-index:2}
.card{background:linear-gradient(180deg,var(--panel-2),var(--panel-2));border:1px solid rgba(255,255,255,.07);
  border-radius:18px;padding:24px 26px; box-shadow:inset 0 0 0 1px rgba(255,255,255,.03)}
.kicker{display:inline-block;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#8bd3ff;background:#0b2b3d;
  border:1px solid rgba(140,210,255,.35);padding:6px 10px;border-radius:999px;margin-bottom:12px}
.hero h1{font-size:46px;line-height:1.05;margin:6px 0 10px}
.sub{color:var(--muted);max-width:640px}
.bullets{margin-top:18px;display:grid;gap:12px}
.b{display:flex;gap:10px;align-items:flex-start}
.chk{width:22px;height:22px;border-radius:999px;display:grid;place-items:center;background:rgba(52,211,153,.16);
  border:1px solid rgba(52,211,153,.6);color:var(--ok2)}
.ctas{display:flex;gap:12px;margin-top:22px}
.btn{padding:12px 18px;border-radius:12px;font-weight:800;text-decoration:none;display:inline-block;border:1px solid transparent}
.btn.primary{background:var(--cta);color:#111}
.btn.primary:hover{filter:brightness(1.06)}
.btn.ghost{background:transparent;color:var(--fg);border-color:rgba(255,255,255,.14)}

.hero-right{position:relative;border-left:1px solid var(--border-soft);overflow:hidden}
.hero-img{position:absolute;inset:0;background:url('__HERO__') center/cover no-repeat;background-position:65% center;
  filter:contrast(1.05) saturate(1.05) brightness(.98)}
.hero-img:after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,rgba(15,23,42,.40) 0%,rgba(15,23,42,.18) 30%,rgba(15,23,42,0) 60%)}

/* ===== Sección Opciones ===== */
.section{max-width:1250px;margin:28px auto 8px;padding:0 24px}
.options{display:grid;grid-template-columns:0.98fr 1.02fr;gap:28px;align-items:center}
.panel{background:var(--panel);border:1px solid rgba(255,255,255,.07);border-radius:18px;padding:24px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.03)}
.panel h2{margin:0 0 10px}
.pillok{width:24px;height:24px;border-radius:999px;background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.6);
  color:#67e8f9;display:grid;place-items:center}
.row{display:flex;gap:12px;align-items:flex-start;margin:14px 0}
.globe-wrap{position:relative;min-height:520px;background:radial-gradient(ellipse at center,rgba(34,197,94,.03),transparent 60%);
  border:1px solid var(--border-soft);border-radius:18px}
.globe{position:absolute;inset:0;background:url('__GLOBO__') center/contain no-repeat;opacity:.95}
.ov1{position:absolute;top:90px;right:100px;width:min(520px,46%);background:url('__OV1__') center/contain no-repeat}
.ov2{position:absolute;bottom:120px;right:140px;width:min(420px,38%);background:url('__OV2__') center/contain no-repeat}
.listado{position:absolute;top:220px;left:54%;width:min(360px,32%);background:url('__LISTADO__') center/contain no-repeat}

/* Responsive */
@media (max-width:1100px){
  .hero{grid-template-columns:1fr}
  .hero-right{height:320px}
  .options{grid-template-columns:1fr}
  .ov1,.ov2,.listado{position:absolute;transform:scale(.9)}
}
</style>
""".replace("__HERO__", HERO)
     .replace("__GLOBO__", GLOBO or "")
     .replace("__OV1__", OV1 or "")
     .replace("__OV2__", OV2 or "")
     .replace("__LISTADO__", LISTADO or ""),
    unsafe_allow_html=True,
)

# ---------- NAVBAR (fijo) ----------
st.markdown(
    """
<div class="topbar" id="nav">
  <div class="brand"><div class="logo">🌳</div><span>CHM</span></div>
  <div class="nav">
    <a href="#inicio" class="btn">Inicio</a>
    <a href="Metodología" class="btn">Metodología</a>
    <a href="Demostración" class="btn">Demostración</a>
  </div>
</div>

<script>
(function(){
  const nav = document.getElementById('nav');
  function applyHeights(){
    const nh = nav ? (nav.getBoundingClientRect().height||56) : 56;
    document.documentElement.style.setProperty('--nav-h', nh + 'px');
  }
  const onScroll = () => {
    const sc = window.scrollY || document.documentElement.scrollTop || 0;
    document.documentElement.classList.toggle('scrolled', sc > 24);
  };
  window.addEventListener('resize', applyHeights);
  document.addEventListener('scroll', onScroll, {passive:true});
  applyHeights(); onScroll();
})();
</script>
""",
    unsafe_allow_html=True,
)

# ========= CONTENIDO (envuelto para que el footer quede abajo) =========
st.markdown('<div class="content-grow">', unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div id="inicio"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="full-bleed">
  <section class="hero">
    <div class="hero-left">
      <div class="card">
        <span class="kicker">DINOV2 • DPT • GEDI</span>
        <h1>Altura de dosel submétrica desde imágenes RGB</h1>
        <p class="sub">
          Mapas de altura de dosel (CHM) por píxel generados desde mosaicos Maxar (~0.59 m GSD, 2017–2020)
          con un encoder ViT auto-supervisado (DINOv2) y un decoder DPT entrenado con CHM de LiDAR aéreo (1 m),
          refinados con un modelo basado en GEDI para mejorar la generalización regional.
        </p>
        <div class="bullets">
          <div class="b"><span class="chk">✓</span>Pre-entrenamiento auto-supervisado con 18 M recortes 256×256.</div>
          <div class="b"><span class="chk">✓</span>Decoder DPT sobre etiquetas ALS (1 m) con salida en 256 bins + Sigloss.</div>
          <div class="b"><span class="chk">✓</span>Corrección densa con GEDI (RH95) mediante factor de escala.</div>
        </div>
        <div class="ctas">
          <a class="btn primary" href="Demostración">Empezar</a>
          <a class="btn ghost" href="Metodología">Cómo funciona</a>
        </div>
      </div>
    </div>
    <div class="hero-right"><div class="hero-img"></div></div>
  </section>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- STEPper Interactivo ----------
FALLBACK_IMG = HERO
S1 = STEP1 or FALLBACK_IMG
S2 = STEP2 or FALLBACK_IMG
S3 = STEP3 or FALLBACK_IMG
S4 = STEP4 or FALLBACK_IMG

STEP_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
:root{--bg:#0b1220;--panel:#0e1624;--fg:#eaf2ff;--muted:#b8c7d8;--ok:#22d3ee;--ok2:#34d399;--cta:#f97316;--ring:#65b2ff;--card:#111b2a;--active:#2b3d4f;--br:18px;--border:rgba(255,255,255,.10)}
html,body{margin:0;background:transparent;color:var(--fg);font-family:ui-sans-serif,system-ui,Inter,Segoe UI,Roboto,sans-serif}
.mk-sec{max-width:1250px;margin:0 auto;padding:0 24px}
.mk-title{font-size:28px;text-align:center;margin:0 0 18px}
.mk-grid{display:grid;grid-template-columns:0.40fr 1.60fr;gap:28px;align-items:start}
.steps{list-style:none;margin:0;padding:0;position:relative}
.steps:after{content:"";position:absolute;left:31px;top:6px;bottom:0;border-left:2px dashed rgba(255,255,255,.18)}
.step{display:flex;gap:12px;align-items:stretch;margin:12px 0;cursor:pointer}
.badge{flex:0 0 auto;width:46px;height:46px;border-radius:999px;display:grid;place-items:center;font-weight:800;background:rgba(255,255,255,.06);border:2px solid rgba(255,255,255,.45);color:#cfe9ff}
.box{flex:1;background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px 18px;transition:background .25s,border-color .25s,box-shadow .25s}
.box h3{margin:0 0 6px;font-size:18px}
.box .desc{margin:0;color:#cfd9ea;line-height:1.55;max-height:0;opacity:.0;overflow:hidden;transition:max-height .35s ease,opacity .25s ease}
.step:hover .box{border-color:rgba(255,255,255,.22)}
.step.active .badge{background:#0e2a44;border-color:var(--ring);color:#e8f6ff}
.step.active .box{background:var(--active);border-color:rgba(101,178,255,.7);box-shadow:0 0 0 1px rgba(101,178,255,.15) inset}
.step.active .box .desc{max-height:500px;opacity:1;margin-top:8px}
.viewer{background:var(--panel);border:1px solid rgba(255,255,255,.07);border-radius:var(--br);box-shadow:inset 0 0 0 1px rgba(255,255,255,.03);overflow:visible;}
.viewer .img-wrap{width:100%;display:flex;justify-content:center;align-items:center;max-height:min(70vh,900px);margin:0 auto;border-radius:inherit;overflow:hidden;background:rgba(255,255,255,.03)}
.viewer img{max-width:100%;max-height:100%;width:auto;height:auto;object-fit:contain;image-rendering:auto}
.viewer .cta{display:flex;justify-content:center;padding:14px}
.viewer .cta a{background:var(--cta);color:#111;text-decoration:none;font-weight:800;padding:12px 18px;border-radius:12px}
.viewer .cta a:hover{filter:brightness(1.06)}
@media(max-width:1100px){
  .mk-grid{grid-template-columns:1fr}
  .steps:after{display:none}
  .viewer .img-wrap{max-height:min(60vh,720px);}
}
</style>
</head>
<body>
<section class="mk-sec" id="metodologia">
  <h2 class="mk-title">¿Cómo se genera?</h2>
  <div class="mk-grid">
    <ol class="steps" id="mkSteps">
      <li class="step active" data-img="__STEP1__" tabindex="0">
        <div class="badge">1</div>
        <div class="box"><h3>Auto-supervisión (DINOv2)</h3>
          <p class="desc">Encoder ViT pre-entrenado con DINOv2 usando ~18 M recortes 256×256.</p></div>
      </li>
      <li class="step" data-img="__STEP2__" tabindex="0">
        <div class="badge">2</div>
        <div class="box"><h3>Regresión CHM (DPT + ALS)</h3>
          <p class="desc">Decoder DPT sobre el encoder, entrenado con CHM ALS (1 m).</p></div>
      </li>
      <li class="step" data-img="__STEP3__" tabindex="0">
        <div class="badge">3</div>
        <div class="box"><h3>Modelo global (GEDI)</h3>
          <p class="desc">Ajuste con GEDI (p.ej., RH95) para mejorar generalización.</p></div>
      </li>
      <li class="step" data-img="__STEP4__" tabindex="0">
        <div class="badge">4</div>
        <div class="box"><h3>Corrección densa y salida</h3>
          <p class="desc">Corrección por bloques y entrega CHM en GeoTIFF/COG.</p></div>
      </li>
    </ol>
    <div class="viewer" id="demostracion">
      <div class="img-wrap"><img id="mkImg" src="__STEP1__" alt="Paso activo"/></div>
      <div class="cta"><a href="Demostración">PROBAR AHORA</a></div>
    </div>
  </div>
</section>
<script>
(function(){
  const list=document.getElementById('mkSteps');const img=document.getElementById('mkImg');
  function activate(li){list.querySelectorAll('.step').forEach(s=>s.classList.remove('active'));li.classList.add('active');const u=li.getAttribute('data-img');if(u) img.src=u;}
  list.addEventListener('click',e=>{const li=e.target.closest('.step');if(!li) return;activate(li);},{passive:true});
  list.querySelectorAll('.step').forEach(li=>li.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate(li);}}));
})();
</script>
</body>
</html>
""".replace("__STEP1__", S1).replace("__STEP2__", S2).replace("__STEP3__", S3).replace("__STEP4__", S4)

st.markdown('<div class="band">', unsafe_allow_html=True)
st_html(STEP_HTML, height=900, scrolling=False)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Sección Resultados ----------
st.markdown('<div class="band">', unsafe_allow_html=True)
st.markdown(
    """
<section class="section">
  <h2 class="title" style="text-align:center;margin:0 0 14px">Resultados disponibles</h2>
  <div class="options">
    <div class="panel">
      <h2 style="margin-top:0">Imágenes Disponibles</h2>
      <p>Mapas de altura de dosel (CHM) obtenidos desde mosaicos Maxar ~0.59 m GSD (2017–2020) con un encoder auto-supervisado (DINOv2) y un decoder DPT entrenado con LiDAR aéreo (ALS, 1 m); las predicciones se refinan con un modelo basado en GEDI.</p>
      <div class="row"><span class="pillok">✓</span><div>Cobertura demostrativa en California y São Paulo sobre mosaicos Maxar.</div></div>
      <div class="row"><span class="pillok">✓</span><div>COGs con cutlines y fechas; también visibles en Google Earth Engine.</div></div>
      <div class="row"><span class="pillok">✓</span><div>Validación con LiDAR aéreo a ~30 m: MAE, RMSE y R²-block.</div></div>
      <div class="row"><span class="pillok">✓</span><div>Evaluación árbol/no-árbol con dataset anotado (~9000 thumbs).</div></div>
    </div>

    <div class="globe-wrap">
      <div class="globe"></div>
      <div class="ov1"></div>
      <div class="ov2"></div>
      <div class="listado"></div>
    </div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)  # cierra .content-grow


# ---------- FOOTER (limpio, 3 columnas) ----------
st.markdown(
    """
<style>
.site-footer{
  background:linear-gradient(180deg,var(--panel-2),var(--panel-3));
  border-top:1px solid rgba(255,255,255,.06);
}
.site-footer .wrap{
  max-width:1250px; margin:0 auto;
  padding:32px 24px;
  display:grid; grid-template-columns:1.2fr 1fr 1fr; gap:28px;
}
.site-footer h4{margin:0 0 10px;font-size:16px;letter-spacing:.02em}
.site-footer p,.site-footer a,.site-footer small{color:var(--muted)}
.site-footer a{text-decoration:none}
.site-footer a:hover{color:var(--fg)}
.site-footer .brand{display:flex;gap:10px;align-items:center}
.site-footer .logo{width:32px;height:32px;border-radius:10px;display:grid;place-items:center;background:#0b2b3d;color:#8bd3ff;border:1px solid rgba(140,210,255,.35)}
.site-footer .bottom{
  border-top:1px solid rgba(255,255,255,.06);
  padding:12px 24px 22px;
  display:flex;justify-content:space-between;align-items:center;gap:10px;
  max-width:1250px;margin:0 auto;
}
/* Responsive */
@media (max-width:900px){ .site-footer .wrap{grid-template-columns:1fr 1fr} }
@media (max-width:600px){ .site-footer .wrap{grid-template-columns:1fr}.site-footer .bottom{flex-direction:column;align-items:flex-start} }
</style>

<footer class="site-footer">
  <div class="wrap">
    <div>
      <div class="brand"><div class="logo">🌳</div><strong>CHM</strong></div>
      <p style="margin-top:10px">Estimación de altura de dosel submétrica con DINOv2 + DPT. Demo académica para Analítica de Datos.</p>
    </div>
    <div>
      <h4>Explorar</h4>
      <div><a href="#inicio">Inicio</a></div>
      <div><a href="Metodología">Metodología</a></div>
      <div><a href="Demostración">Demostración</a></div>
    </div>
    <div>
      <h4>Recursos</h4>
      <div><a href="https://github.com/facebookresearch/HighResCanopyHeight" target="_blank">Repo HighResCanopyHeight</a></div>
      <div><a href="#" target="_blank">Paper</a></div>
      <div><a href="#" target="_blank">Dataset</a></div>
    </div>
  </div>
  <div class="bottom">
    <small>© 2025 CHM • Demo académica</small>
    <small>Hecho con Streamlit</small>
  </div>
</footer>
""",
    unsafe_allow_html=True,
)