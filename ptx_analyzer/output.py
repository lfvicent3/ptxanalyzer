"""
Helpers de saída padronizada para texto / HTML.
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
            from IPython.display import HTML, display
            display(HTML(preformatted_html(text)))
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
    #toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid #dbe4f0;
      backdrop-filter: blur(3px);
    }}
    #toolbar button {{
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #0f172a;
      border-radius: 6px;
      padding: 4px 10px;
      cursor: pointer;
      font-size: 13px;
      line-height: 1.2;
    }}
    #toolbar button:hover {{
      background: #f8fafc;
    }}
    #zoom-label {{
      font-size: 12px;
      color: #475569;
      min-width: 52px;
      text-align: center;
    }}
    #viewport {{
      overflow: auto;
      padding: 8px 10px 12px 10px;
      min-height: {min_height};
      box-sizing: border-box;
      outline: none;
      cursor: grab;
    }}
    #viewport.dragging {{
      cursor: grabbing;
      user-select: none;
    }}
    #stage {{
      width: max-content;
      transform-origin: top left;
    }}
  </style>
</head>
<body>
  <div id="toolbar">
    <button id="zoom-out" type="button">-</button>
    <div id="zoom-label">100%</div>
    <button id="zoom-in" type="button">+</button>
    <button id="zoom-reset" type="button">100%</button>
    <button id="zoom-fit" type="button">Ajustar</button>
  </div>
  <div id="viewport" tabindex="0" title="Atalhos: Ctrl/Cmd +, Ctrl/Cmd -, Ctrl/Cmd 0, Ctrl/Cmd 9">
    <div id="stage">
      <div id="container"></div>
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    const container = document.getElementById("container");
    const viewport = document.getElementById("viewport");
    const stage = document.getElementById("stage");
    const graph = {graph_json};
    const frameId = {json.dumps(graph_id)};
    const zoomLabel = document.getElementById("zoom-label");
    const MIN_SCALE = 0.2;
    const MAX_SCALE = 5.0;
    let scale = 1;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let dragScrollLeft = 0;
    let dragScrollTop = 0;
    const escapeHtml = (text) => text.replace(/[&<>]/g, (c) => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
    const applyZoom = () => {{
      scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
      stage.style.transform = `scale(${{scale}})`;
      zoomLabel.textContent = `${{Math.round(scale * 100)}}%`;
      notifyParent();
    }};
    const setZoom = (nextScale, centerX = null, centerY = null) => {{
      const prevScale = scale;
      scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, nextScale));
      if (centerX !== null && centerY !== null) {{
        const rect = viewport.getBoundingClientRect();
        const offsetX = centerX - rect.left + viewport.scrollLeft;
        const offsetY = centerY - rect.top + viewport.scrollTop;
        const contentX = offsetX / prevScale;
        const contentY = offsetY / prevScale;
        applyZoom();
        viewport.scrollLeft = contentX * scale - (centerX - rect.left);
        viewport.scrollTop = contentY * scale - (centerY - rect.top);
        return;
      }}
      applyZoom();
    }};
    const zoomIn = () => {{
      setZoom(scale + 0.1);
    }};
    const zoomOut = () => {{
      setZoom(scale - 0.1);
    }};
    const zoomReset = () => {{
      setZoom(1);
    }};
    const fitToWidth = () => {{
      const svg = container.querySelector('svg');
      if (!svg) return;
      const contentWidth = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width
        ? svg.viewBox.baseVal.width
        : svg.getBBox().width || svg.getBoundingClientRect().width;
      const available = Math.max(120, viewport.clientWidth - 24);
      if (contentWidth > 0) {{
        scale = Math.min(MAX_SCALE, Math.max(0.3, available / contentWidth));
        applyZoom();
      }}
    }};
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
    document.getElementById('zoom-in').addEventListener('click', zoomIn);
    document.getElementById('zoom-out').addEventListener('click', zoomOut);
    document.getElementById('zoom-reset').addEventListener('click', zoomReset);
    document.getElementById('zoom-fit').addEventListener('click', fitToWidth);
    let resizeObserver = null;
    viewport.addEventListener('wheel', (event) => {{
      if (!(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      const delta = event.deltaY < 0 ? 0.12 : -0.12;
      setZoom(scale + delta, event.clientX, event.clientY);
    }}, {{ passive: false }});
    viewport.addEventListener('mousedown', (event) => {{
      if (event.button !== 0) return;
      isDragging = true;
      dragStartX = event.clientX;
      dragStartY = event.clientY;
      dragScrollLeft = viewport.scrollLeft;
      dragScrollTop = viewport.scrollTop;
      viewport.classList.add('dragging');
      event.preventDefault();
    }});
    window.addEventListener('mousemove', (event) => {{
      if (!isDragging) return;
      const dx = event.clientX - dragStartX;
      const dy = event.clientY - dragStartY;
      viewport.scrollLeft = dragScrollLeft - dx;
      viewport.scrollTop = dragScrollTop - dy;
    }});
    window.addEventListener('mouseup', () => {{
      isDragging = false;
      viewport.classList.remove('dragging');
    }});
    viewport.addEventListener('mouseleave', () => {{
      if (!isDragging) viewport.classList.remove('dragging');
    }});
    const handleShortcut = (event) => {{
      if (!(event.ctrlKey || event.metaKey)) return;
      const key = event.key;
      if (key === '+' || key === '=' ) {{
        event.preventDefault();
        zoomIn();
      }} else if (key === '-' || key === '_') {{
        event.preventDefault();
        zoomOut();
      }} else if (key === '0') {{
        event.preventDefault();
        zoomReset();
      }} else if (key === '9' || key === 'f' || key === 'F') {{
        event.preventDefault();
        fitToWidth();
      }}
    }};
    window.addEventListener('keydown', handleShortcut);
    viewport.addEventListener('keydown', handleShortcut);
    const renderGraph = () => {{
      if (!window.mermaid) {{
        showFallback('Mermaid nao carregou no Colab.');
        return;
      }}
      mermaid.render('{graph_id}-svg', graph).then((result) => {{
        container.innerHTML = result.svg;
        applyZoom();
        fitToWidth();
        viewport.focus();
        notifyParent();
        if (resizeObserver) {{
          resizeObserver.disconnect();
        }}
        if (window.ResizeObserver) {{
          resizeObserver = new ResizeObserver(() => notifyParent());
          resizeObserver.observe(container);
          resizeObserver.observe(viewport);
        }}
      }}).catch((err) => {{
        showFallback('Falha ao renderizar Mermaid: ' + String(err));
        console.error('ptx_analyzer mermaid render error', err);
      }});
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
        renderGraph();
        window.addEventListener('load', notifyParent);
        window.addEventListener('resize', fitToWidth);
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
