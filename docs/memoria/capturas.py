"""Capturas de la app para la memoria: la pantalla de práctica y un juicio de «¿Quién corrige?».

Necesita la app arrancada (por defecto en http://localhost:8000) y Edge instalado; Playwright no
es dependencia del proyecto, así que se ejecuta aparte. La práctica se captura ya calibrada (se
inyecta el perfil de voz de FV1 de Tone Perfect) y antes de grabar, porque grabar exige micrófono.
Las imágenes se recortan a la zona que se comenta en la memoria y se guardan en JPEG.

Uso: uv run --with playwright==1.48.0 python docs/memoria/capturas.py [--base http://localhost:8000]
"""

import argparse
import io
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

PROFILE = {"reference_hz": 233.0, "fmin": 120.0, "fmax": 450.0}
SCALE = 2

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8000")
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figuras"))
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def document_box(page: Page, selector: str) -> dict:
    """Caja del primer elemento en coordenadas del documento: bounding_box() es relativa a la
    ventana, y tras hacer clic la página puede haberse desplazado."""
    return page.evaluate("""(selector) => {
        const r = document.querySelector(selector).getBoundingClientRect();
        return {x: r.left + window.scrollX, y: r.top + window.scrollY, width: r.width, height: r.height};
    }""", selector)


def save_region(page: Page, top_selector: str, bottom_selector: str, name: str) -> None:
    """Captura la página entera y recorta, al ancho de la tarjeta, desde el inicio de un elemento
    hasta el final de otro."""
    top, bottom, card = (document_box(page, s) for s in (top_selector, bottom_selector, ".card"))
    image = Image.open(io.BytesIO(page.screenshot(full_page=True))).convert("RGB")
    box = (card["x"], top["y"] - 8, card["x"] + card["width"], bottom["y"] + bottom["height"] + 8)
    image.crop(tuple(round(v * SCALE) for v in box)).save(out / f"{name}.jpg", quality=86)


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1180, "height": 900}, device_scale_factor=SCALE)
    page.goto(args.base)
    page.evaluate(f"localStorage.setItem('tonos.speakerProfile', JSON.stringify({json.dumps(PROFILE)}))")
    page.reload()
    page.wait_for_selector(".card .step-heading")
    page.wait_for_timeout(800)
    save_region(page, ".card .step-heading", ".card .native-voice", "app_practicar")

    page.get_by_role("tab", name="¿Quién corrige?").click()
    page.wait_for_selector(".syllable-row button")
    page.locator(".syllable-row button:not([disabled])").nth(1).click()
    page.locator(".lab-tones button:not([disabled])").first.click()
    page.locator(".lab-actions button.primary").click()
    page.wait_for_selector(".verdict-table", timeout=60_000)
    page.wait_for_timeout(800)
    save_region(page, ".result", ".verdict-table", "app_quien_corrige")
    browser.close()
print(f"Capturas en {out}")
