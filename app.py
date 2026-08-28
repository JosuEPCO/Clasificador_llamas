import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Configuración principal de la página (Debe ser la primera línea)
st.set_page_config(
    page_title="Clasificador de Dentición - Llamas",
    layout="wide", # Usa todo el ancho de la pantalla para un aspecto moderno
    initial_sidebar_state="expanded"
)

# 2. Barra Lateral (Sidebar) para contexto, instrucciones y autoría
with st.sidebar:
    st.title("Sobre el Proyecto")
    st.write(
        "Esta herramienta de visión computacional automatiza la evaluación de la "
        "cronología dentaria en llamas, optimizando el diagnóstico en campo."
    )
    
    st.subheader("Instrucciones")
    st.write("1. Toma una fotografía clara y bien iluminada de los incisivos.")
    st.write("2. Sube la imagen usando el panel principal.")
    st.write("3. Presiona 'Procesar Imagen' para ejecutar la Red Neuronal Convolucional.")
    
    st.divider() # Línea divisoria elegante
    
    # Sección de autoría solicitada
    st.subheader("Investigación y Desarrollo")
    st.write("**Investigadores:** Josue Pari - Zaid Alonso")
    st.write("**Contacto:** jjosuepco@gmail.com")
    st.caption("Desarrollado para la investigación Producción Animal - UNA-PUNO.")

# 3. Panel Principal
st.title("Clasificador Automatizado de Dentición en Llamas")
st.markdown("Clasifica imágenes fotográficas en cinco etapas zootécnicas: **DL Menor, DL Mayor, 2D, 4D y BLL**.")

# 4. Carga del Modelo
@st.cache_resource
def cargar_modelo():
    return YOLO('best.pt')

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error("⚠️ Error: No se encontró el archivo 'best.pt' en el directorio.")
    st.stop()

# 5. Área de subida de archivos
archivo_subido = st.file_uploader("Selecciona o arrastra la fotografía aquí (JPG, PNG)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Dividimos la pantalla en dos columnas para una interfaz web moderna
    col_imagen, col_resultados = st.columns(2)
    
    imagen = Image.open(archivo_subido)
    
    with col_imagen:
        st.subheader("Fotografía Ingresada")
        # Mostramos la imagen ocupando el ancho de su columna
        st.image(imagen, use_container_width=True, caption="Imagen lista para el análisis")
        
    with col_resultados:
        st.subheader("Diagnóstico del Modelo")
        
        # Botón principal, ancho completo
        if st.button("Procesar Imagen", type="primary", use_container_width=True):
            with st.spinner("Analizando características morfológicas..."):
                # Realizar inferencia
                resultados = modelo(imagen)
                
                diccionario_nombres = resultados[0].names
                indice_mejor_clase = resultados[0].probs.top1
                clase_predicha = diccionario_nombres[indice_mejor_clase]
                porcentaje_confianza = resultados[0].probs.top1conf.item() * 100
                
                # Tarjetas de resultado de alto contraste
                st.success(f"### 🦷 Resultado: {clase_predicha}")
                st.info(f"**Nivel de Confianza:** {porcentaje_confianza:.2f}%")
                
                # Barras de progreso para las probabilidades de todas las clases
                st.markdown("#### Distribución de Probabilidades:")
                probabilidades = resultados[0].probs.data.tolist()
                for i, prob in enumerate(probabilidades):
                    st.progress(prob, text=f"{diccionario_nombres[i]}: {prob*100:.1f}%")