# UNIR
# Seminario de Innovación en Inteligencia Artificial
# Equipo 3013B
# Integrantes: Durán García Sergio Israel; Mallorquín Cabral Christian Javier; Mejía Sandoval Octavio Francisco; Velásquez Irías Fausto Rafael

#!/usr/bin/env python3
"""Evalúa el registro anonimizado de validación humana de US-11 y US-12."""

import argparse
import csv
import json
from pathlib import Path


CAMPOS_SI_NO = [
    "t1_carga_drag_drop",
    "t2_explicacion_confianza",
    "t3_no_certeza_clinica",
    "t4_atajo_c",
    "t5_atajo_r",
    "t6_atajo_n",
    "t7_atajos_inactivos_en_captura",
]


def normalizar(valor: str) -> str:
    return valor.strip().lower().replace("í", "i")


def evaluar(ruta: Path) -> dict:
    with ruta.open(newline="", encoding="utf-8-sig") as archivo:
        filas = list(csv.DictReader(archivo))

    completas = []
    errores = []
    codigos = set()
    for numero, fila in enumerate(filas, start=2):
        codigo = fila.get("codigo", "").strip()
        if not codigo:
            errores.append(f"Fila {numero}: código vacío.")
            continue
        if codigo in codigos:
            errores.append(f"Código duplicado: {codigo}.")
        codigos.add(codigo)

        valores = {campo: normalizar(fila.get(campo, "")) for campo in CAMPOS_SI_NO}
        accidental = normalizar(fila.get("accion_accidental", ""))
        if any(valor not in {"si", "no"} for valor in valores.values()) or accidental not in {"si", "no"}:
            errores.append(f"Fila {numero} ({codigo}): complete todos los campos con Sí o No.")
            continue
        completas.append((codigo, valores, accidental))

    if len(filas) != 5:
        errores.append("El registro debe contener exactamente cinco filas de participantes.")
    if len(completas) != 5 or len(codigos) != 5 or errores:
        return {
            "estado_general": "PENDIENTE",
            "participantes_completos": len(completas),
            "participantes_requeridos": 5,
            "errores": errores,
            "us11": "PENDIENTE",
            "us12": "PENDIENTE",
        }

    us11_correctos = sum(
        datos["t2_explicacion_confianza"] == "si" and datos["t3_no_certeza_clinica"] == "si"
        for _, datos, _ in completas
    )
    tareas_us12 = [
        "t1_carga_drag_drop",
        "t4_atajo_c",
        "t5_atajo_r",
        "t6_atajo_n",
        "t7_atajos_inactivos_en_captura",
    ]
    us12_completos = sum(all(datos[campo] == "si" for campo in tareas_us12) for _, datos, _ in completas)
    acciones_accidentales = sum(accidental == "si" for _, _, accidental in completas)
    us11 = "APROBADA" if us11_correctos >= 4 else "NO APROBADA"
    us12 = "APROBADA" if us12_completos >= 4 and acciones_accidentales == 0 else "NO APROBADA"
    return {
        "estado_general": "LISTO_PARA_CIERRE" if us11 == us12 == "APROBADA" else "REQUIERE_AJUSTES",
        "participantes_completos": 5,
        "participantes_requeridos": 5,
        "errores": errores,
        "us11": us11,
        "us11_respuestas_correctas": us11_correctos,
        "us11_umbral": 4,
        "us12": us12,
        "us12_flujos_completos": us12_completos,
        "us12_umbral": 4,
        "us12_acciones_accidentales": acciones_accidentales,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registro", type=Path)
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()
    resultado = evaluar(args.registro)
    contenido = json.dumps(resultado, ensure_ascii=False, indent=2) + "\n"
    if args.salida:
        args.salida.write_text(contenido, encoding="utf-8")
    print(contenido, end="")
    return 0 if resultado["estado_general"] == "LISTO_PARA_CIERRE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
