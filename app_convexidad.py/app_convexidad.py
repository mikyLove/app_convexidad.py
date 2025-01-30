import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Configuración de la página (opcional)
st.set_page_config(page_title="Demostración de Convexidad", layout="centered")

# --------------------------------------------------------------------------------
# 1. Título y descripción
# --------------------------------------------------------------------------------
st.title("Demostración de Convexidad de Funciones")

st.markdown(
    """
    Esta aplicación permite:
    
    - Ingresar una **función** en términos de \\(x\\).
    - Elegir un **intervalo** para verificar automáticamente la convexidad usando 
      el **criterio de la segunda derivada** \\(f''(x) \\ge 0\\).
    - **Verificar** la definición de convexidad eligiendo \\( x_1, x_2 \\) y \\( \\lambda \\).
    - Visualizar la función y su comportamiento (convexo o no convexo) en el rango definido.
    
    ---
    """
)

# --------------------------------------------------------------------------------
# 2. Menú lateral con:
#    a) Ejemplos predefinidos
#    b) Entrada de función personalizada
# --------------------------------------------------------------------------------
st.sidebar.header("Menú Lateral")

# 2.a. Diccionario de ejemplos
func_examples = {
    "Cuadrática (convexa)": "x**2",
    "Exponencial (convexa)": "exp(x)",
    "Logaritmo (convexa en x>0)": "log(x)",
    "Seno (no convexa en todo ℝ)": "sin(x)",
    "Valor absoluto (convexa, no diferenciable en 0)": "Abs(x)"
}

example_choice = st.sidebar.selectbox("Ejemplos disponibles", list(func_examples.keys()))
if st.sidebar.button("Usar ejemplo"):
    # Reemplaza el contenido del input con el ejemplo seleccionado
    st.session_state["func_input"] = func_examples[example_choice]

# 2.b. Entrada de función personalizada
st.sidebar.subheader("Definir Función Personalizada")
default_func = "x**2"  # valor por defecto
func_input = st.sidebar.text_input("Función en términos de x", value=default_func, key="func_input")

# --------------------------------------------------------------------------------
# 3. Conversión a expresión simbólica con Sympy
# --------------------------------------------------------------------------------
x_sym = sp.Symbol('x', real=True)
try:
    func_sym = sp.sympify(func_input)  # convierte la cadena en objeto simbólico
    # Lambdify para evaluación numérica
    f = sp.lambdify(x_sym, func_sym, "numpy")
    f_prime = sp.lambdify(x_sym, sp.diff(func_sym, x_sym), "numpy")
    f_second_derivative = sp.lambdify(x_sym, sp.diff(func_sym, (x_sym, 2)), "numpy")
except Exception as e:
    st.error(f"Ocurrió un error al procesar la función: {e}")
    st.stop()

# Mostrar la función actual
st.write("### Función actual:")
st.latex(rf"f(x) = {sp.latex(func_sym)}")

# --------------------------------------------------------------------------------
# 4. Selección del intervalo de análisis para la convexidad
# --------------------------------------------------------------------------------
st.sidebar.subheader("Intervalo de Análisis")
x_min = st.sidebar.slider("x mínimo", -10.0, 0.0, -5.0, step=0.5)
x_max = st.sidebar.slider("x máximo", 0.0, 10.0, 5.0, step=0.5)
num_points = st.sidebar.slider("Puntos de muestreo", 50, 1000, 200)

if x_min >= x_max:
    st.error("El valor mínimo del intervalo debe ser menor que el máximo.")
    st.stop()

# Generamos el vector de puntos x_vals
x_vals = np.linspace(x_min, x_max, num_points)

# --------------------------------------------------------------------------------
# 5. Verificación de la convexidad mediante la segunda derivada
# --------------------------------------------------------------------------------
st.subheader("1. Verificación mediante la segunda derivada")

# La CORRECCIÓN está en evaluar punto a punto para construir un array
second_deriv_list = []
for x_val in x_vals:
    try:
        # Evaluamos la segunda derivada en cada punto
        val = f_second_derivative(x_val)
    except:
        val = np.nan
    second_deriv_list.append(val)

# Convertimos la lista a array de floats
second_deriv_vals = np.array(second_deriv_list, dtype=float)

# Aplicamos la máscara de finitos (para evitar problemas con log(), etc.)
finite_mask = np.isfinite(second_deriv_vals)

if not np.any(finite_mask):
    # No hay valores finitos -> dominio inválido, etc.
    st.warning("La segunda derivada no es finita en todo el intervalo. Revisa el dominio de la función.")
else:
    valid_second_deriv = second_deriv_vals[finite_mask]
    if np.all(valid_second_deriv >= 0):
        st.success(f"La función es convexa en el intervalo [{x_min}, {x_max}] (f''(x) ≥ 0).")
    else:
        st.warning(f"La función NO es convexa en todo el intervalo [{x_min}, {x_max}], pues existen puntos con f''(x) < 0.")

st.markdown(
    r"""
    *Recuerda*: Si \\(f''(x) \ge 0\\) en todo el intervalo, la función es convexa en ese rango.  
    En caso contrario, habrá regiones donde no es convexa.
    """
)

# --------------------------------------------------------------------------------
# 6. Verificación de la definición de convexidad
#    f(λx1 + (1−λ)x2) ≤ λf(x1) + (1−λ)f(x2)
# --------------------------------------------------------------------------------
st.subheader("2. Verificación de la Definición de Convexidad")

col1, col2, col3 = st.columns(3)
with col1:
    x1 = st.slider("x1", x_min, x_max, (x_min + x_max)/4, step=0.5)
with col2:
    x2 = st.slider("x2", x_min, x_max, (x_min + x_max)*3/4, step=0.5)
with col3:
    lam = st.slider("λ (lambda)", 0.0, 1.0, 0.5, step=0.1)

x_mid = lam*x1 + (1 - lam)*x2
try:
    f_mid = f(x_mid)
    f_combo = lam*f(x1) + (1 - lam)*f(x2)
    if np.isfinite(f_mid) and np.isfinite(f_combo):
        st.markdown(
            f"""
            - \\( x_1 = {x1}, x_2 = {x2}, \lambda = {lam} \\)
            - \\( \\lambda x_1 + (1-\\lambda)x_2 = {x_mid} \\)
            - \\( f(x_1) = {f(x1)}, f(x_2) = {f(x2)}, f(x_{{mid}}) = f({x_mid}) = {f_mid} \\)
            - \\( \\lambda f(x_1) + (1-\\lambda) f(x_2) = {f_combo} \\)
            """
        )
        if f_mid <= f_combo:
            st.success("¡Se cumple la desigualdad de convexidad para estos valores!")
        else:
            st.error("No se cumple la definición de convexidad para estos valores.")
    else:
        st.warning("La función no está bien definida en estos puntos (posibles valores NaN o Inf).")
except Exception as e:
    st.error(f"Error al evaluar f(x) con los valores dados: {e}")

# --------------------------------------------------------------------------------
# 7. Visualización Gráfica: f(x) y segunda derivada
#    - Coloreamos regiones convexas (f''(x)>=0) y no convexas (f''(x)<0).
#    - Incluimos también la cuerda entre (x1, f(x1)) y (x2, f(x2)).
# --------------------------------------------------------------------------------
st.subheader("3. Visualización Gráfica")

fig, ax = plt.subplots(figsize=(6, 4))

# Graficar la función en el rango
y_vals = [f(val) for val in x_vals]  # Evaluación punto a punto, por si hay dominios restringidos
ax.plot(x_vals, y_vals, label=rf"$f(x) = {func_input}$", color='blue')

# (Opcional) Graficar la segunda derivada en el mismo eje
ax.plot(x_vals, second_deriv_vals, '--', label=r"$f''(x)$ (segunda derivada)", color='purple')

# Resaltar regiones convexas / no convexas en base a la segunda derivada
convex_region = (second_deriv_vals >= 0) & np.isfinite(second_deriv_vals)
ax.fill_between(x_vals, y_vals, color='green', alpha=0.2, where=convex_region, label="Región convexa")
ax.fill_between(x_vals, y_vals, color='red', alpha=0.2, where=~convex_region, label="Región NO convexa")

# Graficar puntos x1, x2, x_mid
try:
    fx1 = f(x1)
    fx2 = f(x2)
    fx_mid = f(x_mid)
    ax.scatter(x1, fx1, color='red', zorder=5, label='(x1, f(x1))')
    ax.scatter(x2, fx2, color='red', zorder=5, label='(x2, f(x2))')
    if np.isfinite(fx_mid):
        ax.scatter(x_mid, fx_mid, color='green', zorder=5, 
                   label=r'$x_{mid} = \lambda x_1 + (1-\lambda)x_2$')
    
    # Cuerda entre (x1, f(x1)) y (x2, f(x2))
    if x2 != x1:
        x_line = np.linspace(x1, x2, 100)
        y_line = fx1 + (fx2 - fx1) * (x_line - x1)/(x2 - x1 + 1e-12)
        ax.plot(x_line, y_line, 'r--', label='Cuerda')
except:
    pass  # En caso de dominio inválido, no graficamos la cuerda

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Visualización de la Función y su Segunda Derivada")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# --------------------------------------------------------------------------------
# 8. Conclusión
# --------------------------------------------------------------------------------
st.markdown(
    """
    **Conclusiones**  
    - Si la segunda derivada \\(f''(x)\\) es no negativa en todo el intervalo, 
      \\(f\\) es convexa en dicho intervalo.  
    - Para verificar en puntos específicos la definición de convexidad, 
      comparamos \\(f(\\lambda x_1 + (1-\\lambda)x_2)\\) con 
      \\(\\lambda f(x_1) + (1-\\lambda) f(x_2)\\).  
    - La visualización final muestra la función, la segunda derivada y las 
      regiones donde la función es convexa (en verde) y no convexa (en rojo).
    """
)

st.info("Fin de la aplicación. ¡Modifica los parámetros en la barra lateral para seguir explorando!")


