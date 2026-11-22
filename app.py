import itertools
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


@st.cache_data
def cargar_casos():
    with open("datos/casos_prueba.json", "r", encoding="utf-8") as f:
        return json.load(f)


def red_a_json_str(red):
    """Convierte el estado actual de la red al formato JSON del proyecto."""
    cpt_serializable = {}
    for var, filas in red.cpt.items():
        cpt_serializable[var] = {
            ("|".join(clave) if clave else ""): [round(p, 6) for p in probs]
            for clave, probs in filas.items()
        }
    data = {
        "orden": red.orden,
        "estados": red.estados,
        "padres": red.padres,
        "cpt": cpt_serializable,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def generar_dot_dag(red):
    """Genera el grafo dirigido (DAG) dinámico en formato DOT para Streamlit."""
    lineas = ['digraph G {', '  rankdir=TB;', '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    for var in red.orden:
        color = "#fde0c5" if len(red.estados[var]) >= 3 else "#dbe9f6"
        etiqueta = f"{var}\\n({' / '.join(red.estados[var])})"
        lineas.append(f'  "{var}" [label="{etiqueta}", fillcolor="{color}"];')
    for var in red.orden:
        for p in red.padres[var]:
            lineas.append(f'  "{p}" -> "{var}";')
    lineas.append("}")
    return "\n".join(lineas)


if "red" not in st.session_state:
    st.session_state.red = RedBayesiana.desde_json("datos/red_retrasos.json")

red = st.session_state.red
casos = cargar_casos()

# Encabezado formal
st.title("Sistema de Predicción y Diagnóstico de Retrasos en Entregas")
st.caption("TransAndina Logística S.A. — Red Bayesiana Discreta (Inferencia por Enumeración y Verificación con pgmpy)")

# Barra lateral dinámica
with st.sidebar:
    st.header("Control del Modelo")
    nodos_multi = [v for v in red.orden if len(red.estados[v]) >= 3]
    texto_multi = ", ".join(f"{v} ({len(red.estados[v])} estados)" for v in nodos_multi) if nodos_multi else "Ninguno"
    st.info(f"**Nodos activos:** {len(red.orden)}\n\n**Variables de 3+ estados:** {texto_multi}")

    if st.button("Restaurar red original", use_container_width=True):
        st.session_state.red = RedBayesiana.desde_json("datos/red_retrasos.json")
        st.success("Red restaurada desde datos/red_retrasos.json.")
        st.rerun()

    st.download_button(
        label="Descargar red actual (.json)",
        data=red_a_json_str(red),
        file_name="red_modificada.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    if st.button("Regenerar resultados y gráficos", use_container_width=True):
        ejecutar_experimentos()
        st.success("Archivos actualizados en resultados/.")

# Pestañas principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Consultas Interactivas",
    "2. Sensibilidad en Vivo",
    "3. Modelo y Tablas CPT",
    "4. Experimentos y Cálculo Manual",
    "5. Editor de Red (+/- Variables)",
])

# ==============================================================================
# PESTAÑA 1: CONSULTAS INTERACTIVAS (PREDICTIVA Y DIAGNÓSTICA)
# ==============================================================================
with tab1:
    st.subheader("Inferencia Predictiva y Diagnóstica")

    col_ctrl, col_res = st.columns([1, 1.4], gap="large")

    with col_ctrl:
        # Solo mostrar casos predefinidos compatibles con las variables actuales
        casos_validos = {
            k: v for k, v in casos.items()
            if v["objetivo"] in red.orden and all(ev in red.orden for ev in v["evidencia"])
        }
        opcion_caso = st.selectbox(
            "Cargar caso de prueba predefinido (opcional):",
            ["Consulta personalizada"] + [f"{k}: {v['tipo']}" for k, v in casos_validos.items()],
        )

        caso_sel = None
        if opcion_caso != "Consulta personalizada":
            clave_caso = opcion_caso.split(":")[0]
            caso_sel = casos_validos[clave_caso]

        if caso_sel:
            idx_obj = red.orden.index(caso_sel["objetivo"])
        elif "Retraso" in red.orden:
            idx_obj = red.orden.index("Retraso")
        else:
            idx_obj = len(red.orden) - 1

        objetivo = st.selectbox("Variable a consultar (Objetivo):", red.orden, index=idx_obj)

        st.markdown("**Evidencia observada** *(deja en 'No observada' para evidencia incompleta)*:")
        evidencia = {}
        for var in red.orden:
            if var == objetivo:
                continue
            opciones = ["No observada"] + red.estados[var]
            val_defecto = caso_sel["evidencia"].get(var, "No observada") if caso_sel else "No observada"
            idx_def = opciones.index(val_defecto) if val_defecto in opciones else 0
            seleccion = st.selectbox(f"{var}:", opciones, index=idx_def, key=f"ev_{var}")
            if seleccion != "No observada":
                evidencia[var] = seleccion

    with col_res:
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
        var_ref = "Retraso" if "Retraso" in red.orden else red.orden[-1]
        est_ref = red.estados[var_ref][-1]
        st.markdown(f"**Impacto inmediato en `{var_ref}`:**")
        p_antes = red.marginal(var_ref)[-1]
        p_despues = red_preview.marginal(var_ref)[-1]
        st.metric(f"P({var_ref} = {est_ref}) global", f"{p_despues:.4f}", delta=f"{p_despues - p_antes:+.4f}")

    if st.button("Aplicar este cambio al modelo activo", type="primary"):
        st.session_state.red = red_preview
        st.success("Parámetro actualizado. Ahora las consultas de la Pestaña 1 usarán este nuevo modelo.")
        st.rerun()

# ==============================================================================
# PESTAÑA 3: ESTRUCTURA DEL MODELO Y TABLAS CPT
# ==============================================================================
with tab3:
    st.subheader("Topología Dinámica (DAG) y Tablas de Probabilidad Condicional (CPT)")
    st.graphviz_chart(generar_dot_dag(red), use_container_width=True)

    for var in red.orden:
        with st.expander(
            f"CPT de {var} | Padres: {red.padres[var] if red.padres[var] else 'Ninguno'}",
            expanded=(len(red.estados[var]) >= 3),
        ):
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

# ==============================================================================
# PESTAÑA 5: EDITOR DINÁMICO DE VARIABLES (AGREGAR / QUITAR)
# ==============================================================================
with tab5:
    st.subheader("Agregar o Quitar Variables de la Red Bayesiana")
    col_add, col_del = st.columns(2, gap="large")

    # --- AGREGAR VARIABLE ---
    with col_add:
        st.markdown("### Agregar nueva variable")
        with st.form("form_agregar_var"):
            nuevo_nombre = st.text_input("Nombre de la variable (sin espacios):", placeholder="Ej. Distancia")
            estados_txt = st.text_input("Estados separados por coma (mínimo 2):", placeholder="Ej. Corta, Larga")
            padres_sel = st.multiselect(
                "Padres de esta variable (máximo 2, según requisito del proyecto):",
                options=red.orden,
                max_selections=2,
            )
            submitted = st.form_submit_button("Agregar variable a la red", type="primary")

            if submitted:
                nombre_limpio = nuevo_nombre.strip().replace(" ", "_")
                lista_estados = [e.strip().replace(" ", "_") for e in estados_txt.split(",") if e.strip()]

                if not nombre_limpio:
                    st.error("Debes escribir un nombre para la variable.")
                elif nombre_limpio in red.orden:
                    st.error(f"La variable '{nombre_limpio}' ya existe en la red.")
                elif len(lista_estados) < 2:
                    st.error("Debes ingresar al menos 2 estados separados por coma.")
                elif len(set(lista_estados)) != len(lista_estados):
                    st.error("Los estados no pueden repetirse.")
                else:
                    # Inicializa probabilidades uniformes para todas las combinaciones de padres
                    combos = list(itertools.product(*[red.estados[p] for p in padres_sel]))
                    prob_uniforme = [1.0 / len(lista_estados)] * len(lista_estados)
                    nueva_cpt_var = {c: list(prob_uniforme) for c in combos}

                    nuevo_orden = red.orden + [nombre_limpio]
                    nuevos_estados = {**red.estados, nombre_limpio: lista_estados}
                    nuevos_padres = {**red.padres, nombre_limpio: padres_sel}
                    nueva_cpt = {**red.cpt, nombre_limpio: nueva_cpt_var}

                    st.session_state.red = RedBayesiana(nuevo_orden, nuevos_estados, nuevos_padres, nueva_cpt)
                    st.success(
                        f"Variable '{nombre_limpio}' agregada con probabilidades iniciales uniformes. "
                        "Puedes ajustar sus valores exactos en la Pestaña 2 (Sensibilidad en Vivo)."
                    )
                    st.rerun()

    # --- QUITAR VARIABLE ---
    with col_del:
        st.markdown("### Quitar una variable")
        # Solo se pueden eliminar variables que no sean padres de otros nodos
        hijos_por_nodo = {v: [h for h in red.orden if v in red.padres[h]] for v in red.orden}
        eliminables = [v for v, hijos in hijos_por_nodo.items() if len(hijos) == 0]

        if len(red.orden) <= 1:
            st.warning("La red debe mantener al menos una variable.")
        else:
            st.write("Para mantener la consistencia matemática de las CPT, solo se pueden eliminar nodos sin hijos.")
            var_eliminar = st.selectbox("Selecciona una variable para eliminar:", eliminables)

            bloqueadas = {v: h for v, h in hijos_por_nodo.items() if h}
            if bloqueadas:
                with st.expander("¿Por qué no aparecen las demás variables?"):
                    for v, h in bloqueadas.items():
                        st.write(f"- **{v}** es padre de: `{', '.join(h)}` (elimina primero sus hijos).")

            if st.button("Eliminar variable seleccionada", type="secondary"):
                nuevo_orden = [v for v in red.orden if v != var_eliminar]
                nuevos_estados = {k: v for k, v in red.estados.items() if k != var_eliminar}
                nuevos_padres = {k: v for k, v in red.padres.items() if k != var_eliminar}
                nueva_cpt = {k: v for k, v in red.cpt.items() if k != var_eliminar}

                st.session_state.red = RedBayesiana(nuevo_orden, nuevos_estados, nuevos_padres, nueva_cpt)
                st.success(f"Variable '{var_eliminar}' eliminada correctamente.")
                st.rerun()