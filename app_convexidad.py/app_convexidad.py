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

# Mostrar la función actual en formato LaTeX
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

# Evaluamos la segunda derivada en cada punto con manejo de errores de dominio
second_deriv_list = []
for xv in x_vals:
    try:
        val = f_second_derivative(xv)
    except:
        # Si el cálculo falla (por ejemplo, log(x) con x <= 0), se asigna NaN
        val = np.nan
    second_deriv_list.append(val)

second_deriv_vals = np.array(second_deriv_list, dtype=float)

# Creamos una máscara para valores finitos
finite_mask = np.isfinite(second_deriv_vals)

# Si ningún valor es finito, no se puede verificar la convexidad numéricamente
if not np.any(finite_mask):
    st.warning(
        "La segunda derivada no es finita o está fuera de dominio "
        f"en todo el intervalo [{x_min}, {x_max}]. "
        "Revisa el dominio de la función o ajusta el intervalo."
    )
else:
    valid_second_deriv = second_deriv_vals[finite_mask]
    # Si todos los valores válidos son >= 0, la función es convexa en esa región
    if np.all(valid_second_deriv >= 0):
        st.success(f"La función es convexa en el intervalo [{x_min}, {x_max}] (f''(x) ≥ 0).")
    else:
        st.warning(
            f"La función NO es convexa en todo el intervalo [{x_min}, {x_max}], "
            "pues existen puntos con f''(x) < 0."
        )

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

# Calculamos x_mid = λ*x1 + (1 - λ)*x2
x_mid = lam*x1 + (1 - lam)*x2

# Verificación de la definición: f(λx1 + (1−λ)x2) ≤ λf(x1) + (1−λ)f(x2)
try:
    fx1 = f(x1)
    fx2 = f(x2)
    fx_mid = f(x_mid)
    
    # Verificamos si son valores finitos
    if not (np.isfinite(fx1) and np.isfinite(fx2) and np.isfinite(fx_mid)):
        st.warning("La función no está bien definida (NaN/Inf) en x1, x2 o x_mid.")
    else:
        f_combo = lam*fx1 + (1 - lam)*fx2
        st.markdown(
            f"""
            - \\( x_1 = {x1}, x_2 = {x2}, \lambda = {lam} \\)
            - \\( \\lambda x_1 + (1-\\lambda)x_2 = {x_mid} \\)
            - \\( f(x_1) = {fx1}, \quad f(x_2) = {fx2}, \quad f(x_{{mid}}) = {fx_mid} \\)
            - \\( \\lambda f(x_1) + (1-\\lambda) f(x_2) = {f_combo} \\)
            """
        )
        if fx_mid <= f_combo:
            st.success("¡Se cumple la desigualdad de convexidad para estos valores!")
        else:
            st.error("No se cumple la definición de convexidad para estos valores.")
except Exception as e:
    st.error(f"Error al evaluar la función en x1, x2 o x_mid: {e}")

# --------------------------------------------------------------------------------
# 7. Visualización Gráfica: f(x), segunda derivada, y cuerda
# --------------------------------------------------------------------------------
st.subheader("3. Visualización Gráfica")

fig, ax = plt.subplots(figsize=(6, 4))

# Evaluamos la función f(x) en cada punto (manejo de dominio)
y_vals = []
for xv in x_vals:
    try:
        val = f(xv)
    except:
        val = np.nan
    y_vals.append(val)

y_vals = np.array(y_vals, dtype=float)

# Graficamos f(x)
ax.plot(x_vals, y_vals, label=rf"$f(x) = {func_input}$", color='blue')

# Graficamos la segunda derivada (donde sea finita)
ax.plot(x_vals, second_deriv_vals, '--', label=r"$f''(x)$ (2da deriv.)", color='purple')

# Resaltamos regiones donde f''(x) >= 0 (convexas) y f''(x) < 0 (no convexas)
convex_region = (second_deriv_vals >= 0) & np.isfinite(second_deriv_vals)
non_convex_region = (second_deriv_vals < 0) & np.isfinite(second_deriv_vals)

# Para poder "rellenar" de forma adecuada, 
# es mejor usar fill_between si la función está definida
# (También se puede filtrar con np.isfinite(y_vals) si se desea)
ax.fill_between(
    x_vals, y_vals, color='green', alpha=0.2,
    where=convex_region, label="Región convexa (f''(x)≥0)"
)
ax.fill_between(
    x_vals, y_vals, color='red', alpha=0.2,
    where=non_convex_region, label="Región NO convexa (f''(x)<0)"
)

# Graficamos los puntos x1, x2, x_mid si son finitos
try:
    fx1 = f(x1)
    fx2 = f(x2)
    fx_mid = f(x_mid)
    
    if np.isfinite(fx1):
        ax.scatter(x1, fx1, color='red', zorder=5, label='(x1, f(x1))')
    if np.isfinite(fx2):
        ax.scatter(x2, fx2, color='red', zorder=5, label='(x2, f(x2))')
    if np.isfinite(fx_mid):
        ax.scatter(x_mid, fx_mid, color='green', zorder=5, 
                   label=r'$x_{mid} = \lambda x_1 + (1-\lambda)x_2$')

    # Graficamos la cuerda entre (x1, f(x1)) y (x2, f(x2)) si x1 != x2
    if x2 != x1 and np.isfinite(fx1) and np.isfinite(fx2):
        x_line = np.linspace(x1, x2, 100)
        # Recta que une (x1, f(x1)) con (x2, f(x2))
        # Evitamos la división por cero sumando un eps pequeño
        eps = 1e-12
        y_line = fx1 + (fx2 - fx1) * (x_line - x1) / (x2 - x1 + eps)
        ax.plot(x_line, y_line, 'r--', label='Cuerda')
except:
    pass  # Si hay error de dominio, no graficamos estos puntos/cuerda

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
      \\(f\\) es convexa en ese intervalo.  
    - Para verificar puntualmente la definición de convexidad, 
      comparamos \\(f(\\lambda x_1 + (1-\\lambda)x_2)\\) con 
      \\(\\lambda f(x_1) + (1-\\lambda) f(x_2)\\).  
    - La visualización final muestra la función, la segunda derivada y las 
      regiones donde la función es convexa (en verde) y no convexa (en rojo).
    """
)

st.info("Fin de la aplicación. ¡Modifica los parámetros en la barra lateral para seguir explorando!")
