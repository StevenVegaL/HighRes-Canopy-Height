# -*- coding: utf-8 -*-
import streamlit as st
from pathlib import Path
from base64 import b64encode
import mimetypes

st.set_page_config(page_title="CHM • Metodología", page_icon="🧪", layout="wide")

# ---------- Utilidades ----------
ASSETS = Path(__file__).resolve().parent.parent / "assets"

def data_uri(p: Path, fallback: str = "") -> str:
    if not p.exists():
        return fallback
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"
    return f"data:{mime};base64,{b64encode(p.read_bytes()).decode()}"

HERO    = data_uri(ASSETS / "hero.webp")
GLOBO   = data_uri(ASSETS / "globo.webp")
OV1     = data_uri(ASSETS / "overlay-buscador.png")
OV2     = data_uri(ASSETS / "overlay-fecha.png")
LISTADO = data_uri(ASSETS / "listado-thumbs.png")

# ---------- CSS (idéntico al landing: navbar fijo + footer grande) ----------
st.markdown(
    """
<style>
:root{
  --bg:#0b1220; --panel:#0e1624; --fg:#eaf2ff; --muted:#b8c7d8; --ok:#22d3ee; --ok2:#34d399; --cta:#f97316;
  --ring:#65b2ff; --panel-2:#101a2a; --panel-3:#162131; --border:rgba(255,255,255,.10); --border-soft:rgba(255,255,255,.06);
  --nav-h: 56px;
}
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
[id] { scroll-margin-top: calc(var(--nav-h) + 12px); }

/* NAVBAR FIJO (igual al landing) */
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
.nav{display:flex; gap:10px}
.nav .btn{padding:10px 14px;border-radius:999px;font-weight:700;border:1px solid var(--border); text-decoration:none;color:var(--fg)}
.nav .btn:hover{border-color:rgba(255,255,255,.22)}

/* Secciones */
.section{max-width:1250px;margin:24px auto;padding:0 24px}
.card{background:linear-gradient(180deg,var(--panel-2),var(--panel-2));
  border:1px solid rgba(255,255,255,.07); border-radius:18px; padding:22px 24px}

/* FOOTER grande (igual al landing) */
.site-footer{margin-top:56px;background:linear-gradient(180deg,var(--panel-2),var(--panel-3));border-top:1px solid rgba(255,255,255,.06)}
.site-footer .wrap{max-width:1250px;margin:0 auto;padding:32px 24px;display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr;gap:28px}
.site-footer h4{margin:0 0 10px;font-size:16px;letter-spacing:.02em}
.site-footer p,.site-footer a,.site-footer small{color:var(--muted)}
.site-footer a{text-decoration:none}
.site-footer a:hover{color:var(--fg)}
.site-footer .brand{display:flex;gap:10px;align-items:center}
.site-footer .logo{width:32px;height:32px;border-radius:10px;display:grid;place-items:center;background:#0b2b3d;color:#8bd3ff;border:1px solid rgba(140,210,255,.35)}
.site-footer .newsletter{display:flex;gap:8px;margin-top:6px}
.site-footer input[type="email"]{flex:1;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:10px;padding:10px 12px;color:var(--fg)}
.site-footer button{background:var(--cta);color:#111;font-weight:800;border:0;border-radius:10px;padding:10px 14px;cursor:pointer}
.site-footer .bottom{border-top:1px solid rgba(255,255,255,.06);padding:12px 24px 22px;display:flex;justify-content:space-between;align-items:center;gap:10px;max-width:1250px;margin:0 auto}
.site-footer .social{display:flex;gap:12px;margin-top:10px}
@media (max-width:900px){.site-footer .wrap{grid-template-columns:1fr 1fr}}
@media (max-width:600px){.site-footer .wrap{grid-template-columns:1fr}.site-footer .bottom{flex-direction:column;align-items:flex-start}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- NAV ----------
st.markdown(
    """
<div class="topbar" id="nav">
  <div class="brand"><div class="logo">🌳</div><span>CHM</span></div>
  <div class="nav">
    <a href="/streamlit_landing_CHM_app" class="btn">Inicio</a>
    <a href="#" class="btn">Metodología</a>
    <a href="/pages/2_Demostración" class="btn">Demostración</a>
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

# ---------- CONTENIDO PLACEHOLDER ----------
st.markdown('<section class="section">', unsafe_allow_html=True)
st.markdown('<div class="card"><h2>Metodología</h2><p>Funciona ✅ — página de metodología lista para contenido.</p></div>', unsafe_allow_html=True)
st.markdown('</section>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    f"""
<footer class="site-footer">
  <div class="wrap">
    <div>
      <div class="brand"><div class="logo">🌳</div><strong>CHM</strong></div>
      <p style="margin-top:10px">Estimación de altura de dosel submétrica con DINOv2 + DPT. Demo académica para Analítica de Datos.</p>
      <div class="social"><a href="#">Twitter/X</a><a href="#">GitHub</a><a href="#">YouTube</a></div>
    </div>
    <div>
      <h4>Explorar</h4>
      <div><a href="/streamlit_landing_CHM_app">Inicio</a></div>
      <div><a href="#">Metodología</a></div>
      <div><a href="/pages/2_Demostración">Demostración</a></div>
    </div>
    <div>
      <h4>Recursos</h4>
      <div><a href="https://github.com/facebookresearch/HighResCanopyHeight" target="_blank">Repo HighResCanopyHeight</a></div>
      <div><a href="#" target="_blank">Paper</a></div>
      <div><a href="#" target="_blank">Dataset</a></div>
    </div>
    <div>
      <h4>Boletín</h4>
      <div class="newsletter">
        <input type="email" placeholder="tu@email.com"/>
        <button>Suscribirme</button>
      </div>
      <small>No enviamos spam. Cancelas cuando quieras.</small>
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
