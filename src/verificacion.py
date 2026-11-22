import itertools
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


def construir_pgmpy(red):
    """Construye una red equivalente en pgmpy solo para verificar resultados."""
    aristas = [(p, v) for v in red.orden for p in red.padres[v]]
    modelo = DiscreteBayesianNetwork(aristas)
    cpds = []
    for v in red.orden:
        pads = red.padres[v]
        combos = list(itertools.product(*[red.estados[p] for p in pads]))
        filas = [[red.cpt[v][c][i] for c in combos] for i in range(len(red.estados[v]))]
        cpds.append(
            TabularCPD(
                variable=v,
                variable_card=len(red.estados[v]),
                values=filas,
                evidence=pads or None,
                evidence_card=[len(red.estados[p]) for p in pads] or None,
                state_names={**{v: red.estados[v]}, **{p: red.estados[p] for p in pads}},
            )
        )
    modelo.add_cpds(*cpds)
    assert modelo.check_model()
    return modelo


def marginal_pgmpy(inferencia, red, var, evidencia):
    """Calcula la distribución marginal usando VariableElimination de pgmpy."""
    f = inferencia.query([var], evidence=evidencia, show_progress=False)
    return [float(f.get_value(**{var: e})) for e in red.estados[var]]