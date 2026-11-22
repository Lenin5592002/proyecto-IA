from pgmpy.inference import VariableElimination
from src.red_bayesiana import RedBayesiana
from src.verificacion import construir_pgmpy, marginal_pgmpy
from src.experimentos import ejecutar_experimentos


def consulta_interactiva(red):
    print("\nVariables disponibles:", ", ".join(red.orden))
    objetivo = input("Variable a consultar: ").strip()
    if objetivo not in red.orden:
        print("Variable no válida.")
        return

    evidencia = {}
    print("Ingresa la evidencia (deja vacío para omitir una variable):")
    for var in red.orden:
        if var == objetivo:
            continue
        est = input(f"  {var} ({'/'.join(red.estados[var])}): ").strip()
        if est:
            if est in red.estados[var]:
                evidencia[var] = est
            else:
                print(f"  Estado '{est}' no válido, se omitirá.")

    propia = red.marginal(objetivo, evidencia)
    inf_pg = VariableElimination(construir_pgmpy(red))
    pgmpy_res = marginal_pgmpy(inf_pg, red, objetivo, evidencia)

    print(f"\nResultado para P({objetivo} | {evidencia}):")
    for estado, p1, p2 in zip(red.estados[objetivo], propia, pgmpy_res):
        print(f"  {estado:12s} -> Propia: {p1:.5f} | pgmpy: {p2:.5f} | Dif: {abs(p1 - p2):.2e}")


def modificar_parametro_interactivo(red):
    print("\nVariables:", ", ".join(red.orden))
    var = input("Variable cuya CPT deseas modificar: ").strip()
    if var not in red.orden:
        print("Variable no válida.")
        return

    filas = list(red.cpt[var].keys())
    for i, f in enumerate(filas):
        print(f"  [{i}] Padres={f} -> {red.cpt[var][f]}")
    idx_fila = int(input("Selecciona el número de fila: "))
    fila = filas[idx_fila]

    for i, est in enumerate(red.estados[var]):
        print(f"  [{i}] {est}")
    idx_est = int(input("Selecciona el número de estado a cambiar: "))
    nuevo_val = float(input("Nuevo valor de probabilidad (0 a 1): "))

    red_mod = red.con_parametro(var, fila, idx_est, nuevo_val)
    print(f"Nueva fila normalizada: {[round(x, 4) for x in red_mod.cpt[var][fila]]}")
    return red_mod


def main():
    red = RedBayesiana.desde_json("datos/red_retrasos.json")
    while True:
        print("\n=== PROYECTO 6: ANÁLISIS DE RETRASOS (REDES BAYESIANAS) ===")
        print("1. Ejecutar los 4 experimentos, verificación y generar gráficos")
        print("2. Realizar una consulta personalizada (en vivo)")
        print("3. Modificar un parámetro de CPT (en vivo)")
        print("4. Restaurar parámetros originales")
        print("5. Salir")
        op = input("Elige una opción: ").strip()

        if op == "1":
            ejecutar_experimentos()
        elif op == "2":
            consulta_interactiva(red)
        elif op == "3":
            nueva = modificar_parametro_interactivo(red)
            if nueva:
                red = nueva
        elif op == "4":
            red = RedBayesiana.desde_json("datos/red_retrasos.json")
            print("Parámetros restaurados desde datos/red_retrasos.json.")
        elif op == "5":
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()