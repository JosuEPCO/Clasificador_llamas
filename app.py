import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Clasificador de Dentición",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# --- TEMA VISUAL: PASTEL AZULADO ---
# =========================================================
st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background: linear-gradient(180deg, #eaf3fb 0%, #dceefb 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #cfe6f7 0%, #bcdcf2 100%);
        border-right: 1px solid #a9cfe8;
    }
    section[data-testid="stSidebar"] * {
        color: #1f3b57 !important;
    }

    /* Títulos principales */
    h1, h2, h3, h4 {
        color: #2c5c8a !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Texto general */
    p, span, label, .stMarkdown {
        color: #2c4a63;
    }

    /* Contenedor de columnas tipo tarjeta */
    div[data-testid="column"] {
        background-color: #ffffffb3;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(90, 140, 180, 0.15);
        border: 1px solid #cfe4f5;
    }

    /* Botón principal */
    div.stButton > button {
        background: linear-gradient(135deg, #7fb8e0, #5a9bd4);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 3px 8px rgba(90, 155, 212, 0.35);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #6aaad9, #4a8cc7);
        transform: translateY(-2px);
        box-shadow: 0 5px 12px rgba(90, 155, 212, 0.45);
    }

    /* File uploader */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #eef7fd !important;
        border: 2px dashed #8fc1e8 !important;
        border-radius: 14px;
    }
    div[data-testid="stFileUploaderDropzone"] * {
        color: #2c5c8a !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] svg {
        fill: #5a9bd4 !important;
    }
    section[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button {
        background-color: #ffffff !important;
        color: #2c5c8a !important;
        border: 1px solid #8fc1e8 !important;
        border-radius: 8px !important;
    }

    /* Header superior */
    header[data-testid="stHeader"] {
        background-color: #eaf3fb !important;
    }
    header[data-testid="stHeader"] * {
        color: #2c5c8a !important;
        fill: #2c5c8a !important;
    }

    /* Barra inferior "Manage app" y otros overlays oscuros del entorno */
    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        background-color: transparent !important;
    }

    /* Mensajes success / info */
    div[data-testid="stNotification"] {
        border-radius: 12px;
    }
    .stAlert {
        border-radius: 12px !important;
    }

    /* Barras de progreso */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #a8d4f0, #5a9bd4) !important;
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-top: 1px solid #b6d8ef;
    }

    /* Caption */
    .stCaption, small {
        color: #4d7a9a !important;
    }
</style>
""", unsafe_allow_html=True)

# --- TRADUCTOR INTERNO ---
mapa_nombres = {
    "DL_menor": "Diente de Leche menor",
    "DL_Mayor": "Diente de Leche Mayor",
    "2D": "2 Dientes",
    "4D": "4 Dientes",
    "BLL": "Boca Llena"
}

# --- DICCIONARIO DE EDADES ZOOTÉCNICAS ---
diccionario_edades = {
    "Diente de Leche menor": "Cría (Aprox. menor a 1 año) - Dientes de leche sin desgaste",
    "Diente de Leche Mayor": "Tui (Aprox. 1 a 2 años) - Dientes de leche con desgaste",
    "2 Dientes": "2.5 a 3 años - Primer par de incisivos permanentes",
    "4 Dientes": "3.5 a 4 años - Segundo par de incisivos permanentes",
    "Boca Llena": "Más de 4.5 años - Todos los incisivos permanentes"
}

with st.sidebar:
    st.title("🦙 Sobre el Proyecto")
    st.write(
        "Esta herramienta de visión computacional automatiza la evaluación de la "
        "cronología dentaria en llamas, optimizando el diagnóstico de edad en campo."
    )

    st.subheader("📋 Instrucciones")
    st.write("1️⃣ Toma una fotografía clara de los incisivos.")
    st.write("2️⃣ Sube la imagen usando el panel principal.")
    st.write("3️⃣ Presiona **Procesar Imagen**.")

    st.divider()

    st.subheader("🔬 Investigación y Desarrollo")
    st.write("**Investigador:** Josue Pari")
    st.write("**Contacto:** jjosuepco@gmail.com")
    st.caption("Desarrollado para la investigación en Zootecnia y Producción Animal.")

st.title("🦙 Clasificador Automatizado de Dentición en Llamas")
st.markdown("Clasifica imágenes fotográficas en cinco etapas zootécnicas y estima la edad aproximada del animal.")

@st.cache_resource
def cargar_modelo():
    return YOLO('best.pt')

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error("Error: No se encontró el archivo 'best.pt' en el directorio.")
    st.stop()

archivo_subido = st.file_uploader("Selecciona o arrastra la fotografía aquí (JPG, PNG)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    col_imagen, col_resultados = st.columns(2)

    imagen = Image.open(archivo_subido)

    with col_imagen:
        st.subheader("📷 Fotografía Ingresada")
        st.image(imagen, use_container_width=True, caption="Imagen lista para el análisis")

    with col_resultados:
        st.subheader("🩺 Diagnóstico del Modelo")

        if st.button("Procesar Imagen", type="primary", use_container_width=True):
            with st.spinner("Analizando características morfológicas..."):
                resultados = modelo(imagen)

                diccionario_yolo = resultados[0].names
                indice_mejor_clase = resultados[0].probs.top1

                clase_original_yolo = diccionario_yolo[indice_mejor_clase]
                clase_traducida = mapa_nombres.get(clase_original_yolo, clase_original_yolo)

                porcentaje_confianza = resultados[0].probs.top1conf.item() * 100

                edad_estimada = diccionario_edades.get(clase_traducida, "Edad no determinada")

                st.success(f"### Etapa Dentaria: {clase_traducida}")
                st.info(f"### Edad Estimada: {edad_estimada}")
                st.write(f"**Nivel de Confianza de la Red Neuronal:** {porcentaje_confianza:.2f}%")

                st.divider()
                st.markdown("#### 📊 Distribución de Probabilidades:")
                probabilidades = resultados[0].probs.data.tolist()

                for i, prob in enumerate(probabilidades):
                    nombre_yolo = diccionario_yolo[i]
                    nombre_barra = mapa_nombres.get(nombre_yolo, nombre_yolo)
                    st.progress(prob, text=f"{nombre_barra}: {prob*100:.1f}%")
