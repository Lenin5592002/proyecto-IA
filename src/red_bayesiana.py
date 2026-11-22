import copy
import itertools
import json


class RedBayesiana:
    """Red bayesiana discreta con inferencia exacta por enumeración."""

    def __init__(self, orden, estados, padres, cpt):
        self.orden = list(orden)
        self.estados = copy.deepcopy(estados)
        self.padres = copy.deepcopy(padres)
        self.cpt = copy.deepcopy(cpt)
        self.validar()

    @classmethod
    def desde_json(cls, ruta_json):
        """Carga la estructura y las CPTs desde un archivo JSON."""
        with open(ruta_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        cpt_convertida = {}
        for var, filas in data["cpt"].items():
            cpt_convertida[var] = {}
            for clave_str, probs in filas.items():
                clave_tupla = () if clave_str == "" else tuple(clave_str.split("|"))
                cpt_convertida[var][clave_tupla] = probs

        return cls(data["orden"], data["estados"], data["padres"], cpt_convertida)

    def validar(self, tol=1e-9):
        """Comprueba: <=2 padres, orden topológico, CPT completas y normalizadas."""
        for var in self.orden:
            pads = self.padres[var]
            assert len(pads) <= 2, f"{var} tiene más de 2 padres"
            for p in pads:
                assert self.orden.index(p) < self.orden.index(var), (
                    f"Orden no topológico: {p} debe ir antes que {var}"
                )
            combos = list(itertools.product(*[self.estados[p] for p in pads]))
            assert set(combos) == set(self.cpt[var].keys()), f"CPT incompleta en {var}"
            for fila, probs in self.cpt[var].items():
                assert len(probs) == len(self.estados[var]), f"Fila mal formada en {var}{fila}"
                assert abs(sum(probs) - 1.0) < tol, f"Fila {var}{fila} no suma 1"
                assert all(p >= 0 for p in probs), f"Probabilidad negativa en {var}{fila}"

    def prob_condicional(self, var, asignacion):
        """P(var = asignacion[var] | padres = asignacion[padres])."""
        fila = tuple(asignacion[p] for p in self.padres[var])
        idx = self.estados[var].index(asignacion[var])
        return self.cpt[var][fila][idx]

    def prob_conjunta(self, asignacion):
        """Regla de la cadena: P(x1,...,xn) = prod_i P(xi | padres(xi))."""
        prob = 1.0
        for var in self.orden:
            prob *= self.prob_condicional(var, asignacion)
        return prob

    def consulta(self, objetivo, evidencia=None):
        """Calcula P(objetivo | evidencia) por enumeración."""
        evidencia = dict(evidencia or {})
        if isinstance(objetivo, str):
            objetivo = [objetivo]
        for v in objetivo:
            assert v not in evidencia, f"{v} no puede ser objetivo y evidencia a la vez"

        ocultas = [v for v in self.orden if v not in objetivo and v not in evidencia]

        sin_normalizar = {}
        for est_obj in itertools.product(*[self.estados[v] for v in objetivo]):
            asig = dict(evidencia)
            asig.update(zip(objetivo, est_obj))
            total = 0.0
            for est_oc in itertools.product(*[self.estados[v] for v in ocultas]):
                asig.update(zip(ocultas, est_oc))
                total += self.prob_conjunta(asig)
            sin_normalizar[est_obj] = total

        z = sum(sin_normalizar.values())
        posterior = {k: v / z for k, v in sin_normalizar.items()}
        return posterior, z

    def marginal(self, var, evidencia=None):
        """Atajo: posterior de una sola variable como lista ordenada por estados."""
        post, _ = self.consulta(var, evidencia)
        return [post[(e,)] for e in self.estados[var]]

    def con_parametro(self, var, fila, indice, nuevo_valor):
        """Devuelve una copia de la red con un parámetro modificado y la fila renormalizada."""
        red = RedBayesiana(self.orden, self.estados, self.padres, self.cpt)
        probs = red.cpt[var][fila]
        resto_antes = 1.0 - probs[indice]
        resto_ahora = 1.0 - nuevo_valor
        nuevas = []
        for i, p in enumerate(probs):
            if i == indice:
                nuevas.append(nuevo_valor)
            elif resto_antes > 0:
                nuevas.append(p * resto_ahora / resto_antes)
            else:
                nuevas.append(resto_ahora / (len(probs) - 1))
        red.cpt[var][fila] = nuevas
        red.validar()
        return red