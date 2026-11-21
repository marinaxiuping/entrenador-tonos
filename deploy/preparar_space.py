"""Monta la carpeta que se sube a un Space de Hugging Face con SDK Gradio (gratuito).

Los Spaces con Docker son de pago; el SDK Gradio gratuito solo ofrece el hardware ZeroGPU, que
construye con Python 3.10 y acepta torch de PyPI en versiones concretas. El app.py de
deploy/hf_space/ arranca la API de la app tal cual. La carpeta resultante contiene:
  - app.py, README.md (configuración del Space) y packages.txt (compilador para pyworld);
  - requirements.txt: deploy/hf_space/requirements.txt, resuelto para Python 3.10 (uv.lock fija
    versiones que exigen 3.11, como numpy 2.4 o pandas 3) y versionado para que el Space sea
    reproducible;
  - el paquete tonos, los modelos, el frontend compilado y las grabaciones de Tone Perfect.

Con --update-requirements regenera deploy/hf_space/requirements.txt a partir de las dependencias
directas de pyproject.toml, más el Gradio y el «spaces» que instala Hugging Face, sin paquetes
publicados después de --exclude-newer.

Requiere el frontend compilado (cd frontend; npm run build) y las grabaciones de Tone Perfect.

Uso: uv run python deploy/preparar_space.py [--update-requirements] [--out deploy/hf_space/build]
"""

import argparse
import shutil
import subprocess
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "hf_space"
TEMPLATE_FILES = ["app.py", "README.md", "packages.txt", "requirements.txt"]
SPACE_PYTHON = "3.10"
# ZeroGPU solo acepta ciertas versiones de torch de PyPI (entre ellas la 2.8.0 del lock) y no la
# build de CPU. Gradio y spaces son los que instala Hugging Face con sdk_version del README.
SPACE_PINS = ["torch==2.8.0", "torchaudio==2.8.0", "setuptools==80.9.0",
              "gradio[oauth,mcp]==6.32.0", "spaces==0.52.1"]


def space_requirements_input() -> str:
    """Dependencias directas del proyecto sin sus cotas inferiores (pensadas para Python 3.11),
    salvo las fijadas con «==», más las del Space."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pinned_names = {Requirement(pin).name for pin in SPACE_PINS}
    lines = list(SPACE_PINS)
    for dependency in project["dependencies"]:
        requirement = Requirement(dependency)
        if requirement.name in pinned_names:
            continue
        extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
        exact = str(requirement.specifier) if str(requirement.specifier).startswith("==") else ""
        lines.append(f"{requirement.name}{extras}{exact}")
    return "\n".join(lines) + "\n"


def update_requirements(exclude_newer: str) -> None:
    subprocess.run(
        ["uv", "pip", "compile", "-", "--no-config", "--python-version", SPACE_PYTHON,
         "--python-platform", "x86_64-manylinux_2_28", "--exclude-newer", exclude_newer,
         "--no-header", "--no-annotate", "-o", str(TEMPLATE / "requirements.txt")],
        input=space_requirements_input(), text=True, check=True, cwd=TEMPLATE)


def assemble_space(out: Path, tone_perfect: Path) -> None:
    frontend = ROOT / "frontend" / "dist"
    for required, hint in [(frontend / "index.html", "compila el frontend: cd frontend; npm run build"),
                           (ROOT / "models" / "laboratorio" / "detector_ompal.pt", "faltan los modelos de la demo (35 y 47)"),
                           (TEMPLATE / "requirements.txt", "genera los requisitos con --update-requirements"),
                           (tone_perfect, "faltan las grabaciones de Tone Perfect")]:
        if not required.exists():
            raise SystemExit(f"{required} no existe: {hint}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in TEMPLATE_FILES:
        shutil.copy2(TEMPLATE / name, out / name)
    shutil.copytree(ROOT / "src" / "tonos", out / "tonos", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "models", out / "models")
    shutil.copytree(frontend, out / "frontend" / "dist")
    shutil.copytree(tone_perfect, out / "data" / "raw" / "toneperfect", ignore=shutil.ignore_patterns("*.xml"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TEMPLATE / "build"))
    ap.add_argument("--tone-perfect", default=str(ROOT / "data" / "raw" / "toneperfect" / "tone_perfect"))
    ap.add_argument("--update-requirements", action="store_true")
    ap.add_argument("--exclude-newer", default="2026-11-21", help="fecha límite de publicación de los paquetes")
    args = ap.parse_args()

    if args.update_requirements:
        update_requirements(args.exclude_newer)
    out = Path(args.out)
    assemble_space(out, Path(args.tone_perfect))
    files = [path for path in out.rglob("*") if path.is_file()]
    print(f"Space montado en {out}: {len(files)} archivos, {sum(p.stat().st_size for p in files) / 1e6:.0f} MB")
