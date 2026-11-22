import json
import os
import pandas as pd
import streamlit as st
from pgmpy.inference import VariableElimination

from src.red_bayesiana import RedBayesiana
from src.verificacion import construir_pgmpy, marginal_pgmpy
from src.experimentos import ejecutar_experimentos

st.set_page_config(
    page_title="Proyecto 6 - Análisis de Retrasos (Red Bayesiana)",
    layout="wide",
)

# Cargar datos y mantener el estado de la red en sesión (útil para cambios en vivo)
@st.cache_data
def cargar_casos():
    with open("datos/casos_prueba.json", "r", encoding="utf-8") as f:
        return json.load(f)


if "red" not in st.session_state:
    st.session_state.red = RedBayesiana.desde_json("datos/red_retrasos.json")

red = st.session_state.red
casos = cargar_casos()

# Encabezado formal
st.title("Sistema de Predicción y Diagnóstico de Retrasos en Entregas")
st.caption("TransAndina Logística S.A. — Red Bayesiana Discreta (Inferencia por Enumeración y Verificación con pgmpy)")

# Barra lateral: estado del modelo y reinicio rápido
with st.sidebar:
    st.header("Control del Modelo")
    st.info(
        f"**Nodos:** {len(red.orden)}\n\n"
        f"**Variable de 3 estados:** Trafico ({', '.join(red.estados['Trafico'])})"
    )
    if st.button("Restaurar parámetros originales", use_container_width=True):
        st.session_state.red = RedBayesiana.desde_json("datos/red_retrasos.json")
        st.success("Parámetros restaurados desde datos/red_retrasos.json.")
        st.rerun()

    st.divider()
    if st.button("Regenerar resultados y gráficos", use_container_width=True):
        ejecutar_experimentos()
        st.success("Archivos actualizados en resultados/.")

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Consultas Interactivas",
    "2. Sensibilidad en Vivo",
    "3. Modelo y Tablas CPT",
    "4. Experimentos y Cálculo Manual",
])

# ==============================================================================
# PESTAÑA 1: CONSULTAS INTERACTIVAS (PREDICTIVA Y DIAGNÓSTICA)
# ==============================================================================
with tab1:
    st.subheader("Inferencia Predictiva y Diagnóstica")

    col_ctrl, col_res = st.columns([1, 1.4], gap="large")

    with col_ctrl:
        # Selector rápido para cargar los 4 casos de prueba oficiales
        opcion_caso = st.selectbox(
            "Cargar caso de prueba predefinido (opcional):",
            ["Consulta personalizada"] + [f"{k}: {v['tipo']}" for k, v in casos.items()],
        )

        caso_sel = None
        if opcion_caso != "Consulta personalizada":
            clave_caso = opcion_caso.split(":")[0]
            caso_sel = casos[clave_caso]

        idx_obj = red.orden.index(caso_sel["objetivo"]) if caso_sel else red.orden.index("Retraso")
        objetivo = st.selectbox("Variable a consultar (Objetivo):", red.orden, index=idx_obj)

        st.markdown("**Evidencia observada** *(deja en 'No observada' para evidencia incompleta)*:")
        evidencia = {}
        for var in red.orden:
            if var == objetivo:
                continue
            opciones = ["No observada"] + red.estados[var]
            val_defecto = caso_sel["evidencia"].get(var, "No observada") if caso_sel else "No observada"
            idx_def = opciones.index(val_defecto)
            seleccion = st.selectbox(f"{var}:", opciones, index=idx_def, key=f"ev_{var}")
            if seleccion != "No observada":
                evidencia[var] = seleccion

    with col_res:
        # Cálculo con algoritmo propio y verificación con pgmpy
        post_dict, p_ev = red.consulta(objetivo, evidencia)
        propia = [post_dict[(e,)] for e in red.estados[objetivo]]
        prior = red.marginal(objetivo)

        inf_pg = VariableElimination(construir_pgmpy(red))
        pgmpy_res = marginal_pgmpy(inf_pg, red, objetivo, evidencia)

        st.markdown(f"### Distribución Posterior: $P(\\text{{{objetivo}}} \\mid \\text{{evidencia}})$")
        st.write(f"**Evidencia aplicada:** `{evidencia if evidencia else 'Ninguna (A priori)'}`")
        st.metric("Probabilidad de la evidencia P(e)", f"{p_ev:.6f}")

        df_comp = pd.DataFrame({
            "Estado": red.estados[objetivo],
            "A priori": [round(x, 6) for x in prior],
            "Enumeración Propia": [round(x, 6) for x in propia],
            "Verificación (pgmpy)": [round(x, 6) for x in pgmpy_res],
            "Diferencia Abs.": [f"{abs(a - b):.2e}" for a, b in zip(propia, pgmpy_res)],
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        df_chart = pd.DataFrame({
            "A priori": prior,
            "Posterior": propia,
        }, index=red.estados[objetivo])
        st.bar_chart(df_chart)

# ==============================================================================
# PESTAÑA 2: ANÁLISIS DE SENSIBILIDAD EN VIVO
# ==============================================================================
with tab2:
    st.subheader("Modificación de Parámetros (Renormalización Automática)")
    st.write(
        "Selecciona cualquier fila de una CPT y ajusta una probabilidad. "
        "El resto de estados de esa fila se reescalarán proporcionalmente para mantener la suma en 1."
    )

    c1, c2 = st.columns(2)
    with c1:
        var_mod = st.selectbox("Variable a modificar:", red.orden, key="sens_var")
        filas = list(red.cpt[var_mod].keys())
        etiquetas_filas = [
            "Sin padres (Prior)" if f == () else ", ".join(f"{p}={v}" for p, v in zip(red.padres[var_mod], f))
            for f in filas
        ]
        idx_fila = st.selectbox(
            "Combinación de padres (fila):",
            range(len(filas)),
            format_func=lambda i: etiquetas_filas[i],
        )
        fila_sel = filas[idx_fila]

    with c2:
        idx_est = st.selectbox(
            "Estado a ajustar:",
            range(len(red.estados[var_mod])),
            format_func=lambda i: red.estados[var_mod][i],
        )
        val_actual = float(red.cpt[var_mod][fila_sel][idx_est])
        nuevo_val = st.slider(
            f"Nuevo valor para P({var_mod}={red.estados[var_mod][idx_est]}):",
            min_value=0.0,
            max_value=1.0,
            value=val_actual,
            step=0.01,
        )

    red_preview = red.con_parametro(var_mod, fila_sel, idx_est, nuevo_val)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Fila antes vs. después:**")
        df_fila = pd.DataFrame({
            "Estado": red.estados[var_mod],
            "Valor actual": [round(x, 4) for x in red.cpt[var_mod][fila_sel]],
            "Valor nuevo": [round(x, 4) for x in red_preview.cpt[var_mod][fila_sel]],
        })
        st.dataframe(df_fila, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Impacto inmediato en la red:**")
        p_ret_antes = red.marginal("Retraso")[1]
        p_ret_despues = red_preview.marginal("Retraso")[1]
        p_alto_antes = red.marginal("Trafico", {"Retraso": "Si"})[2]
        p_alto_despues = red_preview.marginal("Trafico", {"Retraso": "Si"})[2]

        st.metric("P(Retraso = Si) global", f"{p_ret_despues:.4f}", delta=f"{p_ret_despues - p_ret_antes:+.4f}")
        st.metric(
            "P(Trafico = Alto | Retraso = Si)",
            f"{p_alto_despues:.4f}",
            delta=f"{p_alto_despues - p_alto_antes:+.4f}",
        )

    if st.button("Aplicar este cambio al modelo activo", type="primary"):
        st.session_state.red = red_preview
        st.success("Parámetro actualizado. Ahora las consultas de la Pestaña 1 usarán este nuevo modelo.")
        st.rerun()

# ==============================================================================
# PESTAÑA 3: ESTRUCTURA DEL MODELO Y TABLAS CPT
# ==============================================================================
with tab3:
    st.subheader("Topología (DAG) y Tablas de Probabilidad Condicional (CPT)")

    if not os.path.exists("resultados/fig_red.png"):
        ejecutar_experimentos()

    st.image("resultados/fig_red.png", caption="Estructura de la Red Bayesiana (DAG)", width=700)

    for var in red.orden:
        with st.expander(f"CPT de {var} | Padres: {red.padres[var] if red.padres[var] else 'Ninguno'}", expanded=(var == "Trafico")):
            filas_tabla = []
            for clave, probs in red.cpt[var].items():
                fila_dict = {}
                for p, val_p in zip(red.padres[var], clave):
                    fila_dict[p] = val_p
                for est, prob in zip(red.estados[var], probs):
                    fila_dict[f"P({var}={est})"] = round(prob, 4)
                filas_tabla.append(fila_dict)
            st.dataframe(pd.DataFrame(filas_tabla), use_container_width=True, hide_index=True)

# ==============================================================================
# PESTAÑA 4: EXPERIMENTOS, GRÁFICOS Y CONSULTA MANUAL
# ==============================================================================
with tab4:
    st.subheader("Resultados Oficiales de los 4 Casos de Prueba y Cálculo Manual")

    if not os.path.exists("resultados/results.json"):
        ejecutar_experimentos()

    with open("resultados/results.json", "r", encoding="utf-8") as f:
        resultados = json.load(f)

    st.success(
        f"Verificación global (Propia vs. pgmpy): diferencia máxima = {resultados['max_dif_global']:.2e} | "
        f"Cálculo manual Q3 vs. código = {resultados['manual']['dif_vs_codigo']:.2e}"
    )

    c_img1, c_img2 = st.columns(2)
    with c_img1:
        st.image("resultados/fig_predictivas.png", caption="Consultas Predictivas (Q1 y Q2)", use_container_width=True)
    with c_img2:
        st.image("resultados/fig_sensibilidad.png", caption="Análisis de Sensibilidad (S1 y S2)", use_container_width=True)

    st.image("resultados/fig_diagnosticas.png", caption="Consultas Diagnósticas (Q3 y Q4)", use_container_width=True)

    with st.expander("Ver desarrollo paso a paso de la consulta manual Q3: P(Vehiculo | Retraso=Si)"):
        st.json(resultados["manual"])
        