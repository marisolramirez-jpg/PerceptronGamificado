import streamlit as st
import numpy as np
import plotly.graph_objects as go


st.set_config = st.set_page_config(page_title="Simulador de Perceptrón - Laboratorio IA", layout="centered")

st.title("🧠 Laboratorio de Inteligencia Artificial: El Perceptrón")
st.write("Desarrollado por: [Tu Nombre] - Semestre 4")
st.markdown("---")

# --- PANEL DE CONTROL DE PARÁMETROS (SIDEBAR) ---
st.sidebar.header("🕹️ Calibración de Pesos y Sesgo")
st.sidebar.write("Ajusta los parámetros manualmente para modificar la frontera de decisión.")

peso_x1 = st.sidebar.slider("Peso Entrada 1 ($w_1$)", -5.0, 5.0, 0.0, 0.1)
peso_x2 = st.sidebar.slider("Peso Entrada 2 ($w_2$)", -5.0, 5.0, 0.0, 0.1)
sesgo_b = st.sidebar.slider("Sesgo del Sistema ($bias$)", -5.0, 5.0, 0.0, 0.1)


# --- ARQUITECTURA DEL PERCEPTRÓN ---
def evaluar_perceptron(x1, x2, w1, w2, b):
    # Combinación lineal (Suma ponderada)
    suma_neta = (x1 * w1) + (x2 * w2) + b
    # Función de transferencia: Escalón Unitario
    salida = 1 if suma_neta >= 0 else 0
    return suma_neta, salida


# --- INTERFAZ DE ENTRADAS Y TARGETS (MATRIZ DE DATOS) ---
st.subheader("📊 Matriz de Entrenamiento y Etiquetas Deseadas")
st.write("Selecciona los estados activos y define el comportamiento lógico esperado:")

# Diseñamos una cuadrícula de 2x2 para las opciones
fila1_col1, fila1_col2 = st.columns(2)
fila2_col1, fila2_col2 = st.columns(2)

with fila1_col1:
    st.markdown("#### Patrón A")
    check_00 = st.checkbox("Entradas activas (1,1)", value=False, key="c00")
    target_00 = st.selectbox("Clase objetivo (0,0):", [0, 1], index=0) # Por defecto 0 para OR

with fila1_col2:
    st.markdown("#### Patrón B")
    check_01 = st.checkbox("Entradas activas (0,1)", value=True, key="c01")
    target_01 = st.selectbox("Clase objetivo (0,1):", [0, 1], index=1) # Por defecto 1 para OR

with fila2_col1:
    st.markdown("#### Patrón C")
    check_10 = st.checkbox("Entradas activas (1,0)", value=True, key="c10")
    target_10 = st.selectbox("Clase objetivo (1,0):", [0, 1], index=1) # Por defecto 1 para OR

with fila2_col2:
    st.markdown("#### Patrón D")
    check_11 = st.checkbox("Entradas activas (1,1)", value=True, key="c11")
    target_11 = st.selectbox("Clase objetivo (1,1):", [0, 1], index=1) # Por defecto 1 para OR

# Empaquetamos los datos en estructuras iterables
matriz_entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
valores_esperados = [target_00, target_01, target_10, target_11]


# --- PROCESAMIENTO Y EVALUACIÓN ---
tabla_metricas = []
aciertos = 0

for (x1, x2), objetivo in zip(matriz_entradas, valores_esperados):
    net, prediccion = evaluar_perceptron(x1, x2, peso_x1, peso_x2, sesgo_b)
    correcto = (prediccion == objetivo)
    
    if correcto:
        aciertos += 1
        
    tabla_metricas.append({
        "Coordenadas": f"X₁={x1}, X₂={x2}",
        "Target": objetivo,
        "Suma Neta (z)": round(net, 3),
        "Salida Red": prediccion,
        "Resultado": "🎯 Acertado" if correcto else "❌ Error"
    })

# --- INDICADOR DE RENDIMIENTO ---
st.markdown("### Rendimiento del Modelo")
progreso = aciertos / 4
st.progress(progreso)

if aciertos == 4:
    st.success(f"🔥 ¡Convergencia lograda! El modelo clasificó el 100% de los patrones ({aciertos}/4).")
else:
    st.info(f"Ajustando parámetros... Patrones resueltos: {aciertos} de 4.")

st.table(tabla_metricas)


# --- RENDERIZADO DE LA FRONTERA DE DECISIÓN (PLOTLY) ---
st.subheader("📈 Espacio de Características y Límite de Decisión")

grafica = go.Figure()

# Ecuación de la recta: w1*x1 + w2*x2 + b = 0  =>  x2 = (-w1*x1 - b) / w2
eje_x = np.linspace(-1.0, 2.0, 50)

if peso_x2 != 0:
    eje_y = (-peso_x1 * eje_x - sesgo_b) / peso_x2
    # Mantener el gráfico limpio limitando el desborde visual de la línea
    mascara = (eje_y >= -1.0) & (eje_y <= 2.0)
    grafica.add_trace(go.Scatter(
        x=eje_x[mascara], y=eje_y[mascara],
        mode='lines', name='Hiperplano Separador',
        line=dict(color='#00FFCC', width=3)
    ))
else:
    if peso_x1 != 0:
        linea_vertical = -sesgo_b / peso_x1
        grafica.add_vline(x=linea_vertical, line_width=3, line_color="#00FFCC", name="Hiperplano Separador")

# Pintar los patrones en el plano cartesiano
for (x1, x2), obj in zip(matriz_entradas, valores_esperados):
    # Código de colores: Verde esmeralda para activos (1), Magenta para inactivos (0)
    color_nodo = '#00FF66' if obj == 1 else '#FF0066'
    tipo_marca = 'diamond' if obj == 1 else 'square'
    
    grafica.add_trace(go.Scatter(
        x=[x1], y=[x2],
        mode='markers+text',
        marker=dict(color=color_nodo, size=18, symbol=tipo_marca, line=dict(width=1, color='white')),
        text=[f"({x1},{x2}) T:{obj}"],
        textposition="bottom right",
        name=f"Clase {obj}"
    ))

# Configuración del contenedor estético (Tema Oscuro para código moderno)
grafica.update_layout(
    template="plotly_dark",
    xaxis=dict(range=[-0.5, 1.5], title="Eje X₁ (Entrada 1)", gridcolor="rgba(255,255,255,0.1)"),
    yaxis=dict(range=[-0.5, 1.5], title="Eje X₂ (Entrada 2)", gridcolor="rgba(255,255,255,0.1)"),
    height=500,
    showlegend=False,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(grafica, use_container_width=True)