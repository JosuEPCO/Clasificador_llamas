import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Clasificador de Dentición - Josue Pari",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DICCIONARIO DE EDADES ZOOTÉCNICAS ---
# Nota: Puedes ajustar los textos entre comillas según los parámetros exactos de tu investigación
diccionario_edades = {
    "DL_menor": "Cría (Aprox. menor a 1 año) - Dientes de leche sin desgaste",
    "DL_Mayor": "Tui (Aprox. 1 a 2 años) - Dientes de leche con desgaste",
    "2D": "2.5 a 3 años - Primer par de incisivos permanentes",
    "4D": "3.5 a 4 años - Segundo par de incisivos permanentes",
    "BLL": "Más de 4.5 años (Boca Llena) - Todos los incisivos permanentes"
}

with st.sidebar:
    st.title("Sobre el Proyecto")
    st.write(
        "Esta herramienta de visión computacional automatiza la evaluación de la "
        "cronología dentaria en llamas, optimizando el diagnóstico de edad en campo."
    )
    
    st.subheader("Instrucciones")
    st.write("1. Toma una fotografía clara de los incisivos.")
    st.write("2. Sube la imagen usando el panel principal.")
    st.write("3. Presiona 'Procesar Imagen'.")
    
    st.divider()
    
    st.subheader("Investigación y Desarrollo")
    st.write("**Investigador:** Josue Pari")
    st.write("**Contacto:** jjosuepco@gmail.com")
    st.caption("Desarrollado para la investigación en Zootecnia y Producción Animal.")

st.title("Clasificador Automatizado de Dentición en Llamas")
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
        st.subheader("Fotografía Ingresada")
        st.image(imagen, use_container_width=True, caption="Imagen lista para el análisis")
        
    with col_resultados:
        st.subheader("Diagnóstico del Modelo")
        
        if st.button("Procesar Imagen", type="primary", use_container_width=True):
            with st.spinner("Analizando características morfológicas..."):
                resultados = modelo(imagen)
                
                diccionario_nombres = resultados[0].names
                indice_mejor_clase = resultados[0].probs.top1
                clase_predicha = diccionario_nombres[indice_mejor_clase]
                porcentaje_confianza = resultados[0].probs.top1conf.item() * 100
                
                # Buscamos la edad correspondiente en nuestro diccionario
                # Usamos .get() por si el nombre de la clase devuelta por el modelo varía ligeramente
                edad_estimada = diccionario_edades.get(clase_predicha, "Edad no determinada")
                
                # Mostrar los resultados ampliados
                st.success(f"### Etapa Dentaria: {clase_predicha}")
                st.info(f"### Edad Estimada: {edad_estimada}")
                st.write(f"**Nivel de Confianza de la Red Neuronal:** {porcentaje_confianza:.2f}%")
                
                st.divider()
                st.markdown("#### Distribución de Probabilidades:")
                probabilidades = resultados[0].probs.data.tolist()
                for i, prob in enumerate(probabilidades):
                    st.progress(prob, text=f"{diccionario_nombres[i]}: {prob*100:.1f}%")
