"""
Helpers de saída padronizada para texto / HTML / widget.
"""

from __future__ import annotations

import html
import json
import uuid


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


def mermaid_block_html(graph: str,
                       title: str | None = None,
                       min_height: str = "420px") -> str:
    """
    Retorna um bloco HTML autocontido para renderizar Mermaid em Jupyter/Colab.
    """
    graph_id = f"ptx-mermaid-{uuid.uuid4().hex}"
    graph_json = json.dumps(graph)
    title_html = ""
    if title:
        title_html = (
            "<div style='font-family:system-ui,sans-serif;font-size:14px;"
            "font-weight:700;color:#0f172a;padding:8px 10px;"
            "border-bottom:1px solid #dbe4f0;'>"
            + html.escape(title)
            + "</div>"
        )
    iframe_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: #ffffff;
      color: #111827;
      font-family: system-ui, sans-serif;
    }}
    #container {{
      padding: 8px 10px 12px 10px;
      min-height: {min_height};
      overflow: auto;
      box-sizing: border-box;
    }}
    pre {{
      white-space: pre-wrap;
      background: #fff7ed;
      color: #7c2d12;
      padding: 12px;
      border-radius: 8px;
      overflow: auto;
      font-family: ui-monospace, Consolas, 'Courier New', monospace;
    }}
  </style>
</head>
<body>
  <div id="container"></div>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    const container = document.getElementById("container");
    const graph = {graph_json};
    const frameId = {json.dumps(graph_id)};
    const escapeHtml = (text) => text.replace(/[&<>]/g, (c) => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
    const notifyParent = () => {{
      try {{
        const body = document.body;
        const html = document.documentElement;
        const height = Math.max(
          body.scrollHeight, body.offsetHeight,
          html.clientHeight, html.scrollHeight, html.offsetHeight
        );
        parent.postMessage({{ type: 'ptx-mermaid-resize', frameId, height }}, '*');
      }} catch (err) {{
        console.error('ptx_analyzer resize notify error', err);
      }}
    }};
    const showFallback = (message) => {{
      container.innerHTML =
        '<pre>' + escapeHtml(message) + '\\n\\n' + escapeHtml(graph) + '</pre>';
      notifyParent();
    }};
    try {{
      if (!window.mermaid) {{
        showFallback('Mermaid nao carregou no Colab.');
      }} else {{
        mermaid.initialize({{
          startOnLoad: false,
          theme: 'base',
          securityLevel: 'loose',
          flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }},
          themeVariables: {{
            primaryColor: '#ffffff',
            primaryTextColor: '#111827',
            primaryBorderColor: '#60a5fa',
            lineColor: '#2563eb',
            secondaryColor: '#f8fafc',
            tertiaryColor: '#fff7ed',
            background: '#ffffff',
            mainBkg: '#ffffff',
            clusterBkg: '#f8fafc',
            clusterBorder: '#94a3b8',
            edgeLabelBackground: '#ffffff'
          }}
        }});
        mermaid.render('{graph_id}-svg', graph).then((result) => {{
          container.innerHTML = result.svg;
          notifyParent();
          if (window.ResizeObserver) {{
            const ro = new ResizeObserver(() => notifyParent());
            ro.observe(container);
          }}
          window.addEventListener('load', notifyParent);
        }}).catch((err) => {{
          showFallback('Falha ao renderizar Mermaid: ' + String(err));
          console.error('ptx_analyzer mermaid render error', err);
        }});
      }}
    }} catch (err) {{
      showFallback('Falha ao inicializar Mermaid: ' + String(err));
      console.error('ptx_analyzer mermaid init error', err);
    }}
  </script>
</body>
</html>"""
    iframe_srcdoc = html.escape(iframe_doc, quote=True)

    return (
        "<div style='border:1px solid #cbd5e1;border-radius:10px;"
        "background:#ffffff;overflow:hidden;margin:6px;'>"
        + title_html +
        f"<iframe id='{graph_id}' sandbox='allow-scripts allow-same-origin' "
        f"style='width:100%;border:0;height:{min_height};background:#ffffff;' "
        f"srcdoc=\"{iframe_srcdoc}\"></iframe>"
        "<script>"
        "(function(){"
        "  const frameId = " + json.dumps(graph_id) + ";"
        "  const iframe = document.getElementById(frameId);"
        "  if (!iframe) return;"
        "  const minHeight = " + json.dumps(min_height) + ";"
        "  const onMessage = (event) => {"
        "    const data = event.data || {};"
        "    if (data.type !== 'ptx-mermaid-resize' || data.frameId !== frameId) return;"
        "    const h = Math.max(parseInt(minHeight, 10) || 0, Number(data.height) || 0);"
        "    iframe.style.height = h + 'px';"
        "  };"
        "  window.addEventListener('message', onMessage);"
        "})();"
        "</script>"
        "</div>"
    )
