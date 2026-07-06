"""
Helpers de saída padronizada para texto / HTML / widget.
"""

from __future__ import annotations

import html


def preformatted_html(text: str) -> str:
    """Empacota texto monoespaçado em um bloco HTML consistente."""
    return (
        '<pre style="background:#0f1416;color:#e8eaed;'
        'font-family:ui-monospace,Consolas,\'Courier New\',monospace;'
        'padding:14px;border-radius:8px;font-size:13px;'
        'line-height:1.55;overflow-x:auto;white-space:pre;margin:0">'
        + html.escape(text)
        + "</pre>"
    )


def emit_text(text: str, mode: str = "text"):
    """
    Renderiza texto de forma consistente.

    Args:
        text: conteúdo textual.
        mode:
            - "text": imprime no stdout
            - "html": exibe HTML em Jupyter, com fallback para texto
            - "raw": retorna a string sem exibir
    """
    mode = mode.lower()
    if mode == "raw":
        return text

    if mode == "html":
        try:
            from IPython.display import display
            import ipywidgets as w

            display(w.HTML(preformatted_html(text)))
            return text
        except Exception:
            pass

    print(text, end="" if text.endswith("\n") else "\n")
    return text

