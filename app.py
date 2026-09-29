import base64
import html
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(
    page_title="Clasificador de Dentición en Llamas",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------------------------
# Datos de las etapas (en orden cronológico)
# ---------------------------------------------------------------------------

ETAPAS = [
    {
        "clave": "DL_menor",
        "nombre": "Diente de leche menor",
        "categoria": "Cría",
        "edad": "Menor a 1 año",
        "detalle": "Dientes de leche sin desgaste",
        "color": "#D9A441",
    },
    {
        "clave": "DL_Mayor",
        "nombre": "Diente de leche mayor",
        "categoria": "Tui",
        "edad": "1 a 2 años",
        "detalle": "Dientes de leche con desgaste",
        "color": "#D07A34",
    },
    {
        "clave": "2D",
        "nombre": "2 dientes",
        "categoria": None,
        "edad": "2.5 a 3 años",
        "detalle": "Primer par de incisivos permanentes",
        "color": "#B5532F",
    },
    {
        "clave": "4D",
        "nombre": "4 dientes",
        "categoria": None,
        "edad": "3.5 a 4 años",
        "detalle": "Segundo par de incisivos permanentes",
        "color": "#8E3B35",
    },
    {
        "clave": "BLL",
        "nombre": "Boca llena",
        "categoria": None,
        "edad": "4 años y medio a más",
        "detalle": "Todos los incisivos permanentes",
        "color": "#5E2F3A",
    },
]
ETAPA_POR_CLAVE = {etapa["clave"]: etapa for etapa in ETAPAS}


# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #F5EFE6;
    --surface: #FFFCF7;
    --surface-2: #EFE6D8;
    --ink: #2B211B;
    --muted: #75685C;
    --line: #E3D7C5;
    --accent: #B5532F;
    --accent-dark: #8E3B35;
    --radius: 18px;
    --shadow: 0 1px 2px rgba(43, 33, 27, .04), 0 8px 24px rgba(43, 33, 27, .07);
}

/* Base */
.stApp {
    background:
        radial-gradient(1200px 500px at 100% -10%, rgba(181, 83, 47, .08), transparent 60%),
        var(--bg);
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--ink);
}
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1240px;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
h1, h2, h3, h4 { font-family: 'Fraunces', Georgia, serif !important; color: var(--ink) !important; }
p, li, label { color: var(--ink); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface-2);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

.sb-brand { display: flex; align-items: center; gap: .7rem; margin-bottom: 1.2rem; }
.sb-logo {
    width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
    background: var(--ink); font-size: 1.35rem;
}
.sb-brand b { display: block; font-family: 'Fraunces', serif; font-size: 1.05rem; line-height: 1.15; }
.sb-brand small { color: var(--muted); font-size: .78rem; }
.sb-section {
    font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
    color: var(--accent); margin: 1.4rem 0 .6rem;
}
.sb-text { font-size: .9rem; line-height: 1.55; color: var(--ink); }
.sb-steps { list-style: none; padding: 0; margin: 0; counter-reset: paso; }
.sb-steps li {
    counter-increment: paso; display: flex; gap: .7rem; align-items: flex-start;
    font-size: .9rem; line-height: 1.45; margin-bottom: .65rem;
}
.sb-steps li::before {
    content: counter(paso); flex: 0 0 24px; height: 24px; border-radius: 50%;
    background: var(--accent); color: #fff; font-size: .75rem; font-weight: 700;
    display: grid; place-items: center; margin-top: 1px;
}
.sb-card {
    background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
    padding: .9rem 1rem; font-size: .86rem; line-height: 1.55;
}
.sb-card .label { color: var(--muted); font-size: .74rem; }
.sb-card a { color: var(--accent); text-decoration: none; font-weight: 600; }
.sb-foot { color: var(--muted); font-size: .76rem; margin-top: 1rem; line-height: 1.5; }

/* Franja textil andina */
.band {
    height: 12px;
    background:
        linear-gradient(135deg, #D9A441 25%, transparent 25%) -6px 0 / 12px 12px,
        linear-gradient(225deg, #D9A441 25%, transparent 25%) -6px 0 / 12px 12px,
        linear-gradient(315deg, #D9A441 25%, transparent 25%) 0 0 / 12px 12px,
        linear-gradient(45deg, #D9A441 25%, transparent 25%) 0 0 / 12px 12px,
        var(--accent);
}

/* Hero */
.hero {
    position: relative; overflow: hidden; border-radius: 24px; margin-bottom: 1.8rem;
    background:
        radial-gradient(600px 300px at 85% 20%, rgba(217, 164, 65, .22), transparent 70%),
        radial-gradient(500px 300px at 0% 100%, rgba(181, 83, 47, .35), transparent 70%),
        #2B211B;
    box-shadow: 0 20px 50px rgba(43, 33, 27, .25);
}
.hero-inner {
    display: grid; grid-template-columns: 1.5fr 1fr; gap: 2rem; align-items: center;
    padding: 2.6rem 2.8rem 2.4rem;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: .5rem;
    font-size: .74rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
    color: #E9C27A; margin-bottom: .9rem;
}
.hero-eyebrow::before { content: ""; width: 22px; height: 2px; background: #E9C27A; }
.hero-title {
    font-family: 'Fraunces', Georgia, serif; color: #FBF4EA; font-size: clamp(2rem, 3.6vw, 3.1rem);
    line-height: 1.08; font-weight: 600; margin: 0 0 1rem;
}
.hero-title em { font-style: italic; color: #E9A37A; }
.hero p.lead { color: #D9CCBC; font-size: 1.04rem; line-height: 1.6; max-width: 34rem; margin: 0 0 1.5rem; }
.hero-stats { display: flex; flex-wrap: wrap; gap: .6rem; }
.hero-stat {
    border: 1px solid rgba(251, 244, 234, .16); background: rgba(251, 244, 234, .06);
    border-radius: 999px; padding: .45rem .95rem; color: #F2E7D8; font-size: .84rem;
}
.hero-stat b { color: #fff; font-weight: 700; margin-right: .25rem; }
.hero-art {
    background: rgba(251, 244, 234, .05); border: 1px solid rgba(251, 244, 234, .12);
    border-radius: 20px; padding: 1.4rem 1.2rem 1rem; text-align: center;
}
.hero-art svg { width: 100%; height: auto; max-width: 300px; }
.hero-art .cap { color: #BFAF9C; font-size: .78rem; margin-top: .4rem; letter-spacing: .03em; }

/* Encabezados de sección */
.sec-head { display: flex; align-items: baseline; gap: .7rem; margin: .2rem 0 .9rem; }
.sec-num {
    font-family: 'Fraunces', serif; font-size: .95rem; font-weight: 600; color: var(--accent);
    border: 1.5px solid var(--accent); border-radius: 999px; padding: .05rem .55rem;
}
.sec-title { font-family: 'Fraunces', serif; font-size: 1.4rem; font-weight: 600; color: var(--ink); }
.sec-sub { color: var(--muted); font-size: .88rem; margin: -.5rem 0 1rem; }

/* Uploader */
[data-testid="stFileUploader"] > label { display: none; }
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 2px dashed #D4B89A !important;
    border-radius: var(--radius) !important;
    padding: 1.6rem 1.2rem !important;
    transition: border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: #FFF8EF !important;
}
[data-testid="stFileUploaderDropzone"] svg { color: var(--accent) !important; fill: var(--accent) !important; }
[data-testid="stFileUploaderDropzone"] button:not([kind="minimal"]):not([kind="borderlessIcon"]) {
    background: var(--ink) !important; color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button:not([kind="minimal"]):not([kind="borderlessIcon"]) * { color: #fff !important; }
[data-testid="stFileUploaderDropzone"] button[kind="minimal"],
[data-testid="stFileUploaderDropzone"] button[kind="borderlessIcon"] { color: var(--muted) !important; }
[data-testid="stFileUploaderFile"] { color: var(--ink); }

/* Botón principal */
[data-testid="stElementContainer"]:has(> .stButton), .stButton { width: 100% !important; }
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #C45F37, #9C4029) !important;
    color: #fff !important; border: none !important; border-radius: 14px !important;
    padding: .85rem 1.2rem !important; font-weight: 600 !important; font-size: 1rem !important;
    letter-spacing: .01em; box-shadow: 0 8px 20px rgba(181, 83, 47, .3);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 26px rgba(181, 83, 47, .38); }
.stButton > button:active { transform: translateY(0); }
.stButton > button p { color: #fff !important; font-weight: 600; }

/* Tarjetas */
.card {
    background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 1.4rem 1.5rem;
}

/* Foto */
.photo {
    background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: .6rem; margin: 1rem 0;
}
.photo img { width: 100%; max-height: 460px; object-fit: contain; border-radius: 12px; display: block; background: var(--surface-2); }
.photo-meta {
    display: flex; justify-content: space-between; gap: 1rem; padding: .6rem .4rem .2rem;
    font-size: .8rem; color: var(--muted);
}
.photo-meta span:first-child { color: var(--ink); font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Estado vacío */
.empty { text-align: center; padding: 2.4rem 1.6rem; }
.empty svg { width: 170px; height: auto; opacity: .9; }
.card-title { font-family: 'Fraunces', Georgia, serif; font-weight: 600; color: var(--ink); line-height: 1.25; }
.empty .card-title { font-size: 1.25rem; margin: .8rem 0 .4rem; }
.empty p { color: var(--muted); font-size: .92rem; line-height: 1.55; max-width: 26rem; margin: 0 auto; }
.tips { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; margin-top: 1.4rem; text-align: left; }
.tip {
    background: var(--surface-2); border-radius: 12px; padding: .7rem .8rem; font-size: .82rem;
    line-height: 1.4; color: var(--ink);
}
.tip b { display: block; font-size: .78rem; color: var(--accent); margin-bottom: .15rem; }

/* Resultado */
.result { position: relative; overflow: hidden; padding: 0; }
.result-accent { height: 6px; }
.result-body { padding: 1.5rem 1.6rem 1.3rem; }
.result-top { display: flex; justify-content: space-between; gap: 1.2rem; align-items: flex-start; }
.eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: var(--muted); }
.result-stage {
    font-family: 'Fraunces', serif; font-size: clamp(1.8rem, 3vw, 2.4rem); font-weight: 600;
    line-height: 1.1; margin: .3rem 0 .8rem;
}
.chips { display: flex; flex-wrap: wrap; gap: .45rem; }
.chip {
    display: inline-flex; align-items: center; gap: .35rem; border-radius: 999px;
    padding: .32rem .75rem; font-size: .82rem; font-weight: 600;
}
.ring {
    flex: 0 0 auto; width: 108px; height: 108px; border-radius: 50%;
    display: grid; place-items: center;
}
.ring-inner {
    width: 84px; height: 84px; border-radius: 50%; background: var(--surface);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.ring-inner b { font-family: 'Fraunces', serif; font-size: 1.3rem; line-height: 1; letter-spacing: -.02em; }
.ring-inner small { font-size: .66rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; margin-top: .25rem; }
.teeth-box {
    display: flex; align-items: center; gap: 1.2rem; margin-top: 1.3rem;
    background: var(--surface-2); border-radius: 14px; padding: .9rem 1.1rem;
}
.teeth-box svg { flex: 0 0 150px; width: 150px; height: auto; }
.teeth-box .t-title { font-weight: 600; font-size: .95rem; }
.teeth-box .t-text { color: var(--muted); font-size: .84rem; line-height: 1.45; margin-top: .15rem; }
.note {
    display: flex; gap: .6rem; align-items: flex-start; margin-top: 1rem;
    border-radius: 12px; padding: .75rem .9rem; font-size: .86rem; line-height: 1.45;
}
.note-alta { background: #E8F1EA; color: #24553A; }
.note-media { background: #FBF0DA; color: #7A5410; }
.note-baja { background: #F8E3DE; color: #8A2E1F; }

/* Probabilidades */
.probs .card-title { font-size: 1.1rem; margin: 0 0 1rem; }
.prob-row { margin-bottom: .85rem; }
.prob-row:last-child { margin-bottom: 0; }
.prob-head { display: flex; justify-content: space-between; font-size: .88rem; margin-bottom: .35rem; }
.prob-name { display: inline-flex; align-items: center; gap: .5rem; color: var(--ink); }
.prob-name i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.prob-val { font-variant-numeric: tabular-nums; color: var(--muted); }
.prob-row.top .prob-name, .prob-row.top .prob-val { font-weight: 700; color: var(--ink); }
.prob-track { height: 10px; border-radius: 999px; background: var(--surface-2); overflow: hidden; }
.prob-fill { height: 100%; border-radius: 999px; min-width: 3px; }
.prob-row:not(.top) .prob-fill { opacity: .45; }

/* Guía de etapas */
.guide-head { margin: 2.6rem 0 1rem; }
.guide { display: grid; grid-template-columns: repeat(5, 1fr); gap: .9rem; }
.stage {
    position: relative; background: var(--surface); border: 1px solid var(--line);
    border-radius: 16px; padding: 1rem 1rem 1.1rem; transition: transform .2s, box-shadow .2s;
}
.stage:hover { transform: translateY(-3px); box-shadow: var(--shadow); }
.stage svg { width: 100%; height: auto; background: var(--surface-2); border-radius: 10px; padding: .5rem .4rem .2rem; box-sizing: border-box; }
.stage .s-num { font-size: .7rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin-top: .8rem; }
.stage .s-name { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.08rem; margin: .15rem 0 .3rem; color: var(--ink); }
.stage .s-age { font-size: .84rem; font-weight: 600; color: var(--ink); }
.stage .s-detail { font-size: .8rem; color: var(--muted); line-height: 1.4; margin-top: .25rem; }
.stage.active { border-width: 2px; box-shadow: 0 12px 30px rgba(43, 33, 27, .14); transform: translateY(-4px); }
.stage.dim { opacity: .55; }
.stage .badge {
    position: absolute; top: -10px; right: 12px; color: #fff; font-size: .68rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; border-radius: 999px; padding: .2rem .6rem;
}

/* Pie */
.footer {
    margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--line);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: .6rem;
    color: var(--muted); font-size: .8rem;
}

/* Error de modelo */
.error-card { border-left: 5px solid var(--accent); }
.error-card .card-title { font-size: 1.15rem; margin: 0 0 .4rem; }
.error-card code { background: var(--surface-2); padding: .1rem .35rem; border-radius: 6px; }

/* Responsivo */
@media (max-width: 1000px) {
    .guide { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 760px) {
    .hero-inner { grid-template-columns: 1fr; padding: 1.8rem 1.4rem; }
    .hero-art { display: none; }
    .guide { grid-template-columns: repeat(2, 1fr); }
    .tips { grid-template-columns: 1fr; }
    .result-top { flex-direction: column-reverse; }
    .teeth-box { flex-direction: column; align-items: flex-start; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------

def mostrar_html(bloque):
    """Renderiza HTML sin sangrías ni líneas vacías (evita que Markdown lo rompa)."""
    lineas = [linea.strip() for linea in bloque.splitlines() if linea.strip()]
    st.markdown("\n".join(lineas), unsafe_allow_html=True)


def svg_incisivos(nivel, color="#B5532F"):
    """Dibuja los seis incisivos inferiores según la etapa (0 = cría … 4 = boca llena)."""
    pares_permanentes = [0, 0, 1, 2, 3][nivel]
    posiciones = [15, 33, 51, 69, 87, 105]
    pares = [3, 2, 1, 1, 2, 3]  # I3, I2, I1, I1, I2, I3

    dientes = []
    for x, par in zip(posiciones, pares):
        base = 47 - 4 * (1 - ((x - 60) / 58) ** 2)
        if par <= pares_permanentes:
            ancho, alto, radio = 15, 31, 6.5
            relleno, borde = "#FFFDF7", color
        else:
            ancho = 12
            alto, radio = (13, 3) if nivel == 1 else (19, 5.5)
            relleno, borde = "#F4E9D6", "#C9B28E"
        dientes.append(
            f'<rect x="{x - ancho / 2:.1f}" y="{base - alto:.1f}" width="{ancho}" '
            f'height="{alto + 10}" rx="{radio}" fill="{relleno}" stroke="{borde}" stroke-width="1.6"/>'
        )

    encia = '<path d="M2 48 Q60 38 118 48 L118 58 Q60 66 2 58 Z" fill="#E9B0A4" stroke="#D0897C" stroke-width="1.2"/>'
    return (
        '<svg viewBox="0 0 120 64" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Incisivos inferiores, etapa {nivel + 1}">'
        + "".join(dientes) + encia + "</svg>"
    )


def imagen_a_base64(imagen, lado_max=1400):
    vista = imagen.copy()
    vista.thumbnail((lado_max, lado_max))
    buffer = BytesIO()
    vista.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode()


def nivel_confianza(porcentaje):
    if porcentaje >= 80:
        return "alta", "Confianza alta", "La predicción es consistente. Aun así, confirma en campo si el animal está en el límite entre dos etapas."
    if porcentaje >= 50:
        return "media", "Confianza moderada", "Revisa la distribución de probabilidades: la etapa vecina podría ser también plausible."
    return "baja", "Confianza baja", "Se recomienda tomar otra fotografía más nítida y frontal de los incisivos antes de concluir."


def encabezado_seccion(numero, titulo, subtitulo=None):
    sub = f'<div class="sec-sub">{subtitulo}</div>' if subtitulo else ""
    mostrar_html(f"""
    <div class="sec-head"><span class="sec-num">{numero}</span><span class="sec-title">{titulo}</span></div>
    {sub}
    """)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

with st.sidebar:
    mostrar_html("""
    <div class="sb-brand">
        <div class="sb-logo">🦙</div>
        <div><b>Dentición en Llamas</b><small>Visión computacional · YOLO</small></div>
    </div>
    <div class="sb-section">Sobre el proyecto</div>
    <div class="sb-text">
        Herramienta de visión computacional que automatiza la evaluación de la cronología
        dentaria en llamas, optimizando el diagnóstico de edad en campo.
    </div>
    <div class="sb-section">Cómo usarla</div>
    <ol class="sb-steps">
        <li><span>Toma una fotografía clara y frontal de los incisivos.</span></li>
        <li><span>Súbela en el panel principal.</span></li>
        <li><span>Presiona <b>Analizar dentición</b>.</span></li>
    </ol>
    <div class="sb-section">Investigación y desarrollo</div>
    <div class="sb-card">
        <div class="label">Investigadores</div>
        <div><b>Josue Pari</b> · <b>Zaid Alonso</b></div>
        <div class="label" style="margin-top:.55rem">Contacto</div>
        <a href="mailto:jjosuepco@gmail.com">jjosuepco@gmail.com</a>
    </div>
    <div class="sb-foot">Desarrollado para la investigación en Producción Animal — UNA-Puno.</div>
    """)


# ---------------------------------------------------------------------------
# Encabezado principal
# ---------------------------------------------------------------------------

mostrar_html(f"""
<div class="hero">
    <div class="hero-inner">
        <div>
            <div class="hero-eyebrow">Producción Animal · UNA-Puno</div>
            <div class="hero-title">Clasificador automatizado de <em>dentición en llamas</em></div>
            <p class="lead">Sube una fotografía de los incisivos y el modelo identificará la etapa
            zootécnica del animal junto con su edad aproximada.</p>
            <div class="hero-stats">
                <span class="hero-stat"><b>5</b>etapas dentarias</span>
                <span class="hero-stat"><b>YOLO</b>clasificación</span>
                <span class="hero-stat"><b>&lt; 1 año</b>a boca llena</span>
            </div>
        </div>
        <div class="hero-art">
            {svg_incisivos(4, "#E9A37A")}
            <div class="cap">Incisivos inferiores · vista frontal</div>
        </div>
    </div>
    <div class="band"></div>
</div>
""")


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

RUTA_MODELO = Path(__file__).parent / "modelo" / "best.pt"

@st.cache_resource
def cargar_modelo():
    return YOLO(str(RUTA_MODELO))

try:
    modelo = cargar_modelo()
except Exception:
    mostrar_html("""
    <div class="card error-card">
        <div class="card-title">No se pudo cargar el modelo</div>
        <div>No se encontró el archivo <code>modelo/best.pt</code>.
        Colócalo en la carpeta <code>modelo</code> junto a <code>app.py</code> y recarga la página.</div>
    </div>
    """)
    st.stop()


# ---------------------------------------------------------------------------
# Contenido principal
# ---------------------------------------------------------------------------

if "resultado" not in st.session_state:
    st.session_state.resultado = None

col_imagen, col_resultados = st.columns([1.05, 1], gap="large")

with col_imagen:
    encabezado_seccion("01", "Fotografía", "Formatos JPG o PNG. Idealmente con los incisivos centrados y bien iluminados.")
    archivo_subido = st.file_uploader(
        "Selecciona o arrastra la fotografía aquí",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    imagen = None
    if archivo_subido is not None:
        id_archivo = getattr(archivo_subido, "file_id", None) or f"{archivo_subido.name}-{archivo_subido.size}"
        if st.session_state.resultado and st.session_state.resultado["archivo"] != id_archivo:
            st.session_state.resultado = None

        imagen = ImageOps.exif_transpose(Image.open(archivo_subido)).convert("RGB")
        mostrar_html(f"""
        <div class="photo">
            <img src="data:image/jpeg;base64,{imagen_a_base64(imagen)}" alt="Fotografía ingresada">
            <div class="photo-meta">
                <span>{html.escape(archivo_subido.name)}</span>
                <span>{imagen.width} × {imagen.height} px</span>
            </div>
        </div>
        """)

        if st.button("Analizar dentición", type="primary"):
            with st.spinner("Analizando características morfológicas..."):
                resultados = modelo(imagen)
                probs = resultados[0].probs
                nombres = resultados[0].names
                st.session_state.resultado = {
                    "archivo": id_archivo,
                    "clave": nombres[probs.top1],
                    "confianza": probs.top1conf.item() * 100,
                    "probabilidades": {nombres[i]: p for i, p in enumerate(probs.data.tolist())},
                }
    else:
        st.session_state.resultado = None

resultado = st.session_state.resultado

with col_resultados:
    encabezado_seccion("02", "Diagnóstico", "Etapa dentaria, edad estimada y confianza del modelo.")

    if resultado is None:
        if imagen is None:
            titulo_vacio = "Esperando una fotografía"
            texto_vacio = "Sube una imagen de los incisivos de la llama para iniciar el análisis."
        else:
            titulo_vacio = "Imagen lista para analizar"
            texto_vacio = "Presiona <b>Analizar dentición</b> para obtener la etapa y la edad estimada."
        mostrar_html(f"""
        <div class="card empty">
            {svg_incisivos(2)}
            <div class="card-title">{titulo_vacio}</div>
            <p>{texto_vacio}</p>
            <div class="tips">
                <div class="tip"><b>Encuadre</b>Incisivos inferiores de frente, labios retraídos.</div>
                <div class="tip"><b>Luz</b>Luz natural difusa, sin sombras fuertes.</div>
                <div class="tip"><b>Enfoque</b>Imagen nítida, sin movimiento.</div>
                <div class="tip"><b>Distancia</b>La boca debe ocupar gran parte de la foto.</div>
            </div>
        </div>
        """)
    else:
        etapa = ETAPA_POR_CLAVE.get(resultado["clave"])
        if etapa is None:
            etapa = {"clave": resultado["clave"], "nombre": resultado["clave"], "categoria": None,
                     "edad": "Edad no determinada", "detalle": "", "color": "#75685C"}
        nivel = ETAPAS.index(etapa) if etapa in ETAPAS else 2
        color = etapa["color"]
        confianza = resultado["confianza"]
        tipo_nota, titulo_nota, texto_nota = nivel_confianza(confianza)
        texto_confianza = f"{confianza:.0f}%" if confianza >= 99.95 else f"{confianza:.1f}%"

        chips = ""
        if etapa["categoria"]:
            chips += f'<span class="chip" style="background:{color}22;color:{color}">{etapa["categoria"]}</span>'
        chips += f'<span class="chip" style="background:#2B211B;color:#FBF4EA">🕑 {etapa["edad"]}</span>'

        mostrar_html(f"""
        <div class="card result">
            <div class="result-accent" style="background:{color}"></div>
            <div class="result-body">
                <div class="result-top">
                    <div>
                        <div class="eyebrow">Etapa dentaria</div>
                        <div class="result-stage" style="color:{color}">{html.escape(etapa["nombre"])}</div>
                        <div class="chips">{chips}</div>
                    </div>
                    <div class="ring" style="background:conic-gradient({color} {confianza * 3.6:.1f}deg, #EFE6D8 0)">
                        <div class="ring-inner"><b>{texto_confianza}</b><small>confianza</small></div>
                    </div>
                </div>
                <div class="teeth-box">
                    {svg_incisivos(nivel, color)}
                    <div>
                        <div class="t-title">Edad estimada: {etapa["edad"]}</div>
                        <div class="t-text">{etapa["detalle"]}</div>
                    </div>
                </div>
                <div class="note note-{tipo_nota}"><div><b>{titulo_nota}.</b> {texto_nota}</div></div>
            </div>
        </div>
        """)

        filas = ""
        claves_conocidas = [e["clave"] for e in ETAPAS]
        claves_extra = [c for c in resultado["probabilidades"] if c not in claves_conocidas]
        for clave in claves_conocidas + claves_extra:
            if clave not in resultado["probabilidades"]:
                continue
            prob = resultado["probabilidades"][clave] * 100
            info = ETAPA_POR_CLAVE.get(clave, {"nombre": clave, "color": "#75685C"})
            clase_top = " top" if clave == resultado["clave"] else ""
            filas += f"""
            <div class="prob-row{clase_top}">
                <div class="prob-head">
                    <span class="prob-name"><i style="background:{info['color']}"></i>{html.escape(info['nombre'])}</span>
                    <span class="prob-val">{prob:.1f}%</span>
                </div>
                <div class="prob-track"><div class="prob-fill" style="width:{prob:.1f}%;background:{info['color']}"></div></div>
            </div>
            """

        mostrar_html(f"""
        <div class="card probs" style="margin-top:1rem">
            <div class="card-title">Distribución de probabilidades</div>
            {filas}
        </div>
        """)


# ---------------------------------------------------------------------------
# Guía de etapas
# ---------------------------------------------------------------------------

clave_activa = resultado["clave"] if resultado else None
tarjetas = ""
for i, etapa in enumerate(ETAPAS):
    clases = "stage"
    estilo = ""
    insignia = ""
    if clave_activa:
        if etapa["clave"] == clave_activa:
            clases += " active"
            estilo = f' style="border-color:{etapa["color"]}"'
            insignia = f'<span class="badge" style="background:{etapa["color"]}">Resultado</span>'
        else:
            clases += " dim"
    categoria = f' · {etapa["categoria"]}' if etapa["categoria"] else ""
    tarjetas += f"""
    <div class="{clases}"{estilo}>
        {insignia}
        {svg_incisivos(i, etapa["color"])}
        <div class="s-num" style="color:{etapa['color']}">Etapa {i + 1}{categoria}</div>
        <div class="s-name">{etapa["nombre"]}</div>
        <div class="s-age">{etapa["edad"]}</div>
        <div class="s-detail">{etapa["detalle"]}</div>
    </div>
    """

mostrar_html(f"""
<div class="guide-head">
    <div class="sec-head"><span class="sec-num">03</span><span class="sec-title">Guía de etapas dentarias</span></div>
    <div class="sec-sub">Cronología de erupción de los incisivos inferiores permanentes en llamas.</div>
</div>
<div class="guide">{tarjetas}</div>
<div class="footer">
    <span>Clasificador de Dentición en Llamas · Producción Animal, UNA-Puno</span>
    <span>Las edades son aproximadas y deben complementarse con la evaluación en campo.</span>
</div>
""")
