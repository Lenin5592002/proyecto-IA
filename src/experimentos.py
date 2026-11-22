import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pgmpy.inference import VariableElimination

from src.red_bayesiana import RedBayesiana
from src.verificacion import construir_pgmpy, marginal_pgmpy


def ejecutar_experimentos():
    os.makedirs("resultados", exist_ok=True)

    red = RedBayesiana.desde_json("datos/red_retrasos.json")
    with open("datos/casos_prueba.json", "r", encoding="utf-8") as f:
        CONSULTAS = json.load(f)

    RESULT = {}
    AZUL, NARANJA, GRIS = "#1f77b4", "#d95f02", "#8c8c8c"

    modelo_pg = construir_pgmpy(red)
    inf_pg = VariableElimination(modelo_pg)

    # A. Las cuatro consultas
    max_dif_global = 0.0
    RESULT["consultas"] = {}
    for nombre, q in CONSULTAS.items():
        ev = q["evidencia"]
        ocultas = [v for v in red.orden if v not in ev]
        prior = {v: red.marginal(v) for v in ocultas}
        post, p_ev = red.consulta(q["objetivo"], ev)
        marg = {v: red.marginal(v, ev) for v in ocultas}
        pg = {v: marginal_pgmpy(inf_pg, red, v, ev) for v in ocultas}
        dif = {v: max(abs(a - b) for a, b in zip(marg[v], pg[v])) for v in ocultas}
        max_dif_global = max(max_dif_global, max(dif.values()))
        RESULT["consultas"][nombre] = {
            "tipo": q["tipo"],
            "evidencia": ev,
            "objetivo": q["objetivo"],
            "P_evidencia": p_ev,
            "marginales_propia": marg,
            "marginales_pgmpy": pg,
            "prior": prior,
            "dif_max": dif,
        }
    RESULT["max_dif_global"] = max_dif_global

    ev4 = CONSULTAS["Q4"]["evidencia"]
    conj, _ = red.consulta(["Clima", "Vehiculo"], ev4)
    RESULT["Q4_conjunta_Clima_Vehiculo"] = {f"{k[0]}|{k[1]}": v for k, v in conj.items()}
    f_pg = inf_pg.query(["Clima", "Vehiculo"], evidence=ev4, show_progress=False)
    RESULT["Q4_conjunta_dif"] = max(
        abs(f_pg.get_value(Clima=k[0], Vehiculo=k[1]) - v) for k, v in conj.items()
    )
    RESULT["P_retraso_prior"] = red.marginal("Retraso")

    # B. Consulta manual (Q3): P(Trafico | Retraso=Si)
    P, E = red.cpt, red.estados
    # Paso 1: P(T) = sum_{C,V} P(C) P(V) P(T | C, V)
    pt = {
        t: sum(P["Clima"][()][ic] * P["Vehiculo"][()][iv] * P["Trafico"][(c, v)][it]
               for ic, c in enumerate(E["Clima"]) for iv, v in enumerate(E["Vehiculo"]))
        for it, t in enumerate(E["Trafico"])
    }
    # Paso 2: P(Retraso=Si | T) = sum_P P(P) P(Si | T, P)
    rt = {
        t: sum(P["Preparacion"][()][ip] * P["Retraso"][(t, pr)][1] for ip, pr in enumerate(E["Preparacion"]))
        for t in E["Trafico"]
    }
    # Paso 3: conjunta con la evidencia y normalización
    conj_manual = {t: pt[t] * rt[t] for t in E["Trafico"]}
    z_manual = sum(conj_manual.values())
    RESULT["manual"] = {
        "P_trafico": pt,
        "P_retraso_dado_T": rt,
        "conjunta": conj_manual,
        "Z": z_manual,
        "posterior": {t: conj_manual[t] / z_manual for t in conj_manual},
    }
    post_q3, _ = red.consulta("Trafico", {"Retraso": "Si"})
    RESULT["manual"]["dif_vs_codigo"] = max(
        abs(RESULT["manual"]["posterior"][k[0]] - v) for k, v in post_q3.items()
    )

    # C. Análisis de sensibilidad
    sens = {}
    FILA_S1 = ("Lluvia", "Falla")
    EV_S1 = {"Clima": "Lluvia", "Vehiculo": "Falla"}
    s1 = []
    for a in [0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.95]:
        r2 = red.con_parametro("Trafico", FILA_S1, 2, a)
        s1.append({
            "a": a,
            "fila": r2.cpt["Trafico"][FILA_S1],
            "P_retraso_dado_escenario": r2.marginal("Retraso", EV_S1)[1],
            "P_retraso": r2.marginal("Retraso")[1],
            "P_alto_dado_retraso": r2.marginal("Trafico", {"Retraso": "Si"})[2],
        })
    sens["S1"] = s1

    s2 = []
    for t in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
        r2 = red.con_parametro("Preparacion", (), 1, t)
        s2.append({
            "theta": t,
            "P_retraso": r2.marginal("Retraso")[1],
            "P_dem_dado_retraso": r2.marginal("Preparacion", {"Retraso": "Si"})[1],
        })
    sens["S2"] = s2
    RESULT["sensibilidad"] = sens

    with open("resultados/results.json", "w", encoding="utf-8") as fh:
        json.dump(RESULT, fh, indent=2, ensure_ascii=False)

    # D. Figuras
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: Red
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pos = {"Clima": (1.2, 3.0), "Vehiculo": (4.0, 3.0), "Trafico": (2.6, 1.6),
           "Preparacion": (5.4, 1.6), "Retraso": (4.0, 0.1)}
    cajas = {}
    for v, (x, y) in pos.items():
        col = "#fde0c5" if v == "Trafico" else "#dbe9f6"
        caja = FancyBboxPatch((x - 1.05, y - 0.32), 2.1, 0.64, boxstyle="round,pad=0.02", fc=col, ec="#333", lw=1.2)
        ax.add_patch(caja)
        cajas[v] = caja
        ax.text(x, y + 0.08, v, ha="center", va="center", fontweight="bold", fontsize=9.5)
        ax.text(x, y - 0.14, " / ".join(red.estados[v]), ha="center", va="center", fontsize=7.5, color="#444")
    for v in red.orden:
        for p in red.padres[v]:
            ax.add_patch(FancyArrowPatch(pos[p], pos[v], arrowstyle="-|>", mutation_scale=14,
                                         patchA=cajas[p], patchB=cajas[v], shrinkA=2, shrinkB=2, color="#333", lw=1.2))
    ax.set_xlim(-0.2, 6.9); ax.set_ylim(-0.7, 3.6); ax.axis("off")
    ax.set_title("Estructura de la red bayesiana (naranja: variable de tres estados)")
    fig.tight_layout(); fig.savefig("resultados/fig_red.png", dpi=200); plt.close(fig)

    # Fig 2: Predictivas
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    etq = ["A priori", "Q1: lluvia y\nfalla del vehículo", "Q2: condiciones\nfavorables"]
    vals = [RESULT["P_retraso_prior"][1],
            RESULT["consultas"]["Q1"]["marginales_propia"]["Retraso"][1],
            RESULT["consultas"]["Q2"]["marginales_propia"]["Retraso"][1]]
    bars = ax.bar(etq, vals, color=[GRIS, NARANJA, AZUL], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1); ax.set_ylabel("P(Retraso = Si)"); ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Consultas predictivas: probabilidad de retraso")
    fig.tight_layout(); fig.savefig("resultados/fig_predictivas.png", dpi=200); plt.close(fig)

    # Fig 3: Diagnósticas
    def panel(ax, var):
        estados = red.estados[var]
        xs = list(range(len(estados)))
        w = 0.26
        ax.bar([x - w for x in xs], RESULT["consultas"]["Q3"]["prior"][var], w, color=GRIS, label="A priori")
        ax.bar(xs, RESULT["consultas"]["Q3"]["marginales_propia"][var], w, color=NARANJA, label="Q3: Retraso=Si")
        if var in RESULT["consultas"]["Q4"]["marginales_propia"]:
            ax.bar([x + w for x in xs], RESULT["consultas"]["Q4"]["marginales_propia"][var], w, color=AZUL,
                   label="Q4: Retraso=Si, Prep.=Optima")
        ax.set_xticks(xs); ax.set_xticklabels(estados, fontsize=8); ax.set_title(var, fontsize=9); ax.set_ylim(0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True)
    for ax, var in zip(axes, ["Vehiculo", "Clima", "Trafico", "Preparacion"]):
        panel(ax, var)
    axes[0].set_ylabel("Probabilidad")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Consultas diagnósticas: probabilidad de cada posible causa", y=1.02)
    fig.tight_layout(); fig.savefig("resultados/fig_diagnosticas.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Fig 4: Sensibilidad
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    a_vals = [d["a"] for d in s1]
    axes[0].plot(a_vals, [d["P_retraso"] for d in s1], "-o", color=NARANJA, label="P(Retraso=Si)")
    axes[0].plot(a_vals, [d["P_alto_dado_retraso"] for d in s1], "-s", color=AZUL, label="P(Trafico=Alto | Retraso=Si)")
    axes[0].axvline(0.60, color=GRIS, ls="--", lw=1); axes[0].text(0.61, 0.03, "valor base", color=GRIS, fontsize=8)
    axes[0].set_xlabel("P(Trafico=Alto | Clima=Lluvia, Vehiculo=Falla)"); axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7.5, loc="upper left"); axes[0].set_title("S1: parámetro del nodo de tres estados")

    t_vals = [d["theta"] for d in s2]
    axes[1].plot(t_vals, [d["P_retraso"] for d in s2], "-o", color=NARANJA, label="P(Retraso=Si)")
    axes[1].plot(t_vals, [d["P_dem_dado_retraso"] for d in s2], "-s", color=AZUL, label="P(Preparacion=Demorada | Retraso=Si)")
    axes[1].axvline(0.30, color=GRIS, ls="--", lw=1); axes[1].text(0.31, 0.03, "valor base", color=GRIS, fontsize=8)
    axes[1].set_xlabel("P(Preparacion=Demorada)"); axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7.5, loc="upper left"); axes[1].set_title("S2: preparación del pedido")
    fig.tight_layout(); fig.savefig("resultados/fig_sensibilidad.png", dpi=200); plt.close(fig)

    print("\n--- RESUMEN DE EXPERIMENTOS ---")
    print("Diferencia máxima (propia vs pgmpy):", max_dif_global)
    print("Diferencia manual vs código (Q3):", RESULT["manual"]["dif_vs_codigo"])
    for n, qq in RESULT["consultas"].items():
        obj = qq["objetivo"]
        print(f"{n} ({qq['tipo']}): {obj} = {[round(x, 4) for x in qq['marginales_propia'][obj]]}")
    print("Resultados y gráficos guardados en la carpeta 'resultados/'.\n")


if __name__ == "__main__":
    ejecutar_experimentos()
    