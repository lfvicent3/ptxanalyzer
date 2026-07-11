"""
Visualização dinâmica do traço de execução de um kernel PTX, pensada para
rodar bem no Google Colab.

Por que não Mermaid
--------------------
A tentativa anterior (ver `mermaid_block_html` em `output.py` e os antigos
`*_playground.html`) gerava um texto Mermaid novo a cada passo e pedia para
o Mermaid rodar `render()` + layout do zero em um `<iframe sandbox>` dentro
da própria página do Colab (que já é um iframe). Isso causava três problemas
observados na prática:

  1. "Grafo minúsculo com espaço em branco": o `fitToWidth()` calculava a
     escala a partir do `getBBox()` do SVG e do `clientWidth` do viewport,
     mas essa medição corria numa corrida com o layout assíncrono do
     Mermaid e com o próprio iframe (que começa com altura 0 e só cresce
     depois de um `postMessage` de volta pro pai) — o resultado dependia de
     timing e frequentemente `fit()` media o container antes dele ter
     tamanho real.
  2. Reinicializar o Mermaid inteiro a cada passo (`mermaid.render(...)`
     de novo) refaz o layout do zero, então a cada troca de passo o grafo
     "pisca" e pode nascer em outra escala.
  3. Iframe dentro de iframe (Colab já isola cada saída de célula em um
     iframe) soma mais uma camada de sandboxing/`postMessage`, o que é
     frágil e não necessário.

Este módulo troca Mermaid por **Cytoscape.js + layout dagre**, carregado
diretamente no HTML de saída da célula (sem iframe próprio — o Colab já
isola a célula, uma segunda camada só atrapalha). Cytoscape expõe uma API
de fit real (`cy.fit()`), que mede o bounding box do grafo já *depois* do
layout e o container já *depois* de estar no DOM — sem gambiarra manual de
escala. Trocar de passo apenas atualiza classes/estilos dos elementos
existentes (não reconstrói o grafo), o que elimina o "piscar" e a
sensibilidade a timing.

Estratégia para múltiplas threads
----------------------------------
O grafo principal mostra sempre uma **visão agregada**: intensidade de cor
= fração de threads que já passou por aquele bloco até o passo atual;
contorno destacado = threads ativas *naquele* bloco no passo atual. Uma
thread ou warp específico pode ser selecionado no painel lateral, o que
sobrepõe o caminho daquela thread/warp no grafo sem exigir uma cor
diferente por thread (isso fica ilegível com muitas threads).

Este módulo só consome as estruturas que o analisador dinâmico já produz
(`PTXAnalyzer.dynamic_flow(..., mode="data")` — chaves `control_flow` e
`dynamic_flow`); não reimplementa nem reinterpreta o traço.

Por que Cytoscape/dagre vêm embutidos (vendorizados), não via CDN
-------------------------------------------------------------------
Uma primeira versão carregava essas libs de um CDN (`<script src="https://
cdn.jsdelivr.net/...">` inserido em tempo de execução). Isso funciona bem
num navegador comum e no Colab, mas quebra em pelo menos um visualizador
importante: o renderer de output HTML do VS Code roda a saída de célula
de notebook dentro de um webview isolado que bloqueia scripts remotos
inseridos dinamicamente — mesmo com a máquina tendo internet. O sintoma é
exatamente "Cytoscape.js não carregou via CDN": o restante da UI (que só
usa JS inline, sem rede — timeline, tabela de threads) continua
funcionando, só o carregamento remoto falha silenciosamente. Em vez de
tentar detectar/contornar CSPs de cada frontend, os três arquivos
(`ptx_analyzer/vendor/*.min.js`) são lidos e embutidos como `<script>`
inline no HTML gerado — o grafo passa a funcionar em qualquer lugar que
já execute o resto do JS inline (VS Code, Jupyter clássico, JupyterLab,
Colab, arquivo `.html` aberto direto no navegador), com ou sem rede.
"""

from __future__ import annotations

import html
import json
import os
import re
import uuid
from pathlib import Path
from typing import Iterable, Optional


# ──────────────────────────────────────────────────────────────────────────
# 1) Construção do payload (dados) a partir do resultado do analisador
# ──────────────────────────────────────────────────────────────────────────

def _is_jump_stub(block: dict) -> bool:
    """Bloco que só existe para um `bra.uni` de reconvergência/else — não
    carrega nenhuma decisão nem efeito visível, então é "esticado" para o
    alvo real (mesma convenção usada pelo Mermaid estático em analyzer.py:
    `_resolve_visual_target`)."""
    name = block.get("display_name") or ""
    return name.startswith("Salto") and len(block.get("exits") or []) == 1


def _resolve_visual_target(blocks: dict, label: str) -> str:
    seen: set[str] = set()
    current = label
    while current in blocks and current not in seen:
        seen.add(current)
        block = blocks[current]
        if _is_jump_stub(block):
            target = (block.get("exits") or [{}])[0].get("target")
            if not target:
                break
            current = target
            continue
        break
    return current


def _block_summary(label: str, block: dict) -> dict:
    raw_instruction_name = (block.get("raw_instruction_name") or "").strip()
    ptx_name = raw_instruction_name.split(" + ", 1)[0].strip() if raw_instruction_name else ""
    return {
        "id": label,
        "label": block.get("display_name") or block.get("title") or label,
        "instruction_name": block.get("instruction_name") or "",
        "raw_instruction_name": raw_instruction_name,
        "ptx_name": ptx_name,
        "description": block.get("description") or "",
        "source_line": block.get("source_line") or 0,
        "source_code": (block.get("source_code") or "").strip(),
        "inline_source_line": block.get("inline_source_line") or 0,
        "inline_source_code": (block.get("inline_source_code") or "").strip(),
        "is_entry": bool(block.get("is_entry")),
        "is_terminal": bool(block.get("is_terminal")),
        "instruction_count": block.get("instruction_count", 0),
        "repeated_source_instance": int(block.get("repeated_source_instance") or 0),
    }


def _decision_source_key(block: dict) -> tuple[str, int]:
    source_line = int(block.get("source_line") or 0)
    inline_line = int(block.get("inline_source_line") or 0)
    source_code = (block.get("source_code") or "").strip()
    inline_source_code = (block.get("inline_source_code") or "").strip()
    reference_code = source_code or inline_source_code
    has_compound_operator = ("&&" in reference_code) or ("||" in reference_code)
    if reference_code and not has_compound_operator:
        return ("none", 0)
    if source_line > 0:
        return ("source", source_line)
    if inline_line > 0:
        return ("inline", inline_line)
    return ("none", 0)


def _compute_graph_groups(blocks: dict, order: list[str], visible_labels: list[str], control_flow: dict) -> list[dict]:
    visible_set = set(visible_labels)

    def is_decision_block(label: str) -> bool:
        return any((exit_info.get("type") == "conditional") for exit_info in (blocks[label].get("exits") or []))

    groups: list[dict] = []
    grouped_labels: set[str] = set()

    current: list[str] = []
    current_key: tuple[str, int] | None = None
    compound_idx = 0
    for label in order:
        if label not in visible_set:
            if len(current) > 1:
                compound_idx += 1
                groups.append({
                    "id": f"compound_{compound_idx}",
                    "kind": "compound_decision",
                    "label": f"Decisão Composta {compound_idx}",
                    "members": current[:],
                })
                grouped_labels.update(current)
            current = []
            current_key = None
            continue
        if is_decision_block(label):
            decision_key = _decision_source_key(blocks[label])
            if decision_key == ("none", 0):
                if len(current) > 1:
                    compound_idx += 1
                    groups.append({
                        "id": f"compound_{compound_idx}",
                        "kind": "compound_decision",
                        "label": f"Decisão Composta {compound_idx}",
                        "members": current[:],
                    })
                    grouped_labels.update(current)
                current = []
                current_key = None
                continue
            if not current:
                current = [label]
                current_key = decision_key
            elif decision_key == current_key:
                current.append(label)
            else:
                if len(current) > 1:
                    compound_idx += 1
                    groups.append({
                        "id": f"compound_{compound_idx}",
                        "kind": "compound_decision",
                        "label": f"Decisão Composta {compound_idx}",
                        "members": current[:],
                    })
                    grouped_labels.update(current)
                current = [label]
                current_key = decision_key
        else:
            if len(current) > 1:
                compound_idx += 1
                groups.append({
                    "id": f"compound_{compound_idx}",
                    "kind": "compound_decision",
                    "label": f"Decisão Composta {compound_idx}",
                    "members": current[:],
                })
                grouped_labels.update(current)
            current = []
            current_key = None
    if len(current) > 1:
        compound_idx += 1
        groups.append({
            "id": f"compound_{compound_idx}",
            "kind": "compound_decision",
            "label": f"Decisão Composta {compound_idx}",
            "members": current[:],
        })
        grouped_labels.update(current)

    order_index = {label: idx for idx, label in enumerate(order)}
    loop_sites = control_flow.get("loop_sites") or []
    unroll_idx = 0
    rendered_unroll_groups: list[tuple[list[str], int, int]] = []

    def span_of(loop: dict) -> tuple[int, int] | None:
        header = loop.get("header")
        latch = loop.get("latch")
        if header not in order_index or latch not in order_index:
            return None
        header_idx = order_index[header]
        latch_idx = order_index[latch]
        if latch_idx < header_idx:
            return None
        return header_idx, latch_idx

    for loop in loop_sites:
        span_range = span_of(loop)
        if span_range is None:
            continue
        header_idx, latch_idx = span_range

        nested = False
        for other in loop_sites:
            if other is loop:
                continue
            other_range = span_of(other)
            if other_range is None:
                continue
            other_header_idx, other_latch_idx = other_range
            if header_idx <= other_header_idx and other_latch_idx <= latch_idx and other_range != span_range:
                nested = True
                break
        if nested:
            continue

        span = [label for label in order[header_idx:latch_idx + 1] if label in visible_set]
        if len(span) < 2 or any(label in grouped_labels for label in span):
            continue
        line_counts: dict[int, int] = {}
        for label in span[1:]:
            line = int(blocks[label].get("source_line") or 0)
            if line > 0:
                line_counts[line] = line_counts.get(line, 0) + 1
        if not line_counts:
            continue
        body_line, factor = max(line_counts.items(), key=lambda item: item[1])
        if factor < 2:
            continue
        unroll_idx += 1
        groups.append({
            "id": f"unroll_{unroll_idx}",
            "kind": "unroll",
            "label": f"Desenrolado por fator {factor} (linha {body_line})",
            "members": span[:],
        })
        grouped_labels.update(span)
        rendered_unroll_groups.append((span[:], factor, body_line))

    branch_by_label = {site.get("block_label"): site for site in (control_flow.get("branch_sites") or [])}
    remainder_idx = 0
    for span, factor, _body_line in rendered_unroll_groups:
        header, latch = span[0], span[-1]
        latch_block = blocks.get(latch) or {}
        exit_target = next((edge.get("target") for edge in (latch_block.get("exits") or []) if edge.get("target") != header), None)
        exit_site = branch_by_label.get(exit_target)
        join_target = exit_site.get("reconvergence_target") if exit_site else None
        if exit_target not in order_index or join_target not in order_index:
            continue
        start_idx = order_index[exit_target]
        end_idx = order_index[join_target]
        if end_idx <= start_idx:
            continue
        remainder = [label for label in order[start_idx:end_idx] if label in visible_set]
        if len(remainder) < 2 or any(label in grouped_labels for label in remainder):
            continue
        remainder_idx += 1
        extra_label = "iteração restante" if factor - 1 == 1 else "iterações restantes"
        groups.append({
            "id": f"remainder_{remainder_idx}",
            "kind": "remainder",
            "label": f"Resto do desenrolamento (até {factor - 1} {extra_label})",
            "members": remainder[:],
        })
        grouped_labels.update(remainder)

    return groups


def _load_source_lines(source_path: Optional[str]) -> list[str]:
    if not source_path:
        return []
    try:
        with open(source_path, "r", encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in handle]
    except OSError:
        return []


def _extract_kernel_source(source_lines: list[str],
                            kernel_name: Optional[str]) -> tuple[list[str], int]:
    """Recorta do arquivo-fonte só a função do kernel em estudo (um .cu
    típico desses experimentos declara vários kernels no mesmo arquivo —
    ex.: `bubble_sort_all.cu` tem a variante global/shared/register — e
    mostrar o arquivo inteiro só distrai de qual bloco é qual).

    Localiza a linha com `<kernel_name>(`, inclui o bloco de comentário
    `/** ... */`/`//` imediatamente acima (se houver) e usa contagem de
    chaves a partir daí para achar o fim da função. Se não achar o nome
    (ex.: nome mangled sem correspondência no .cu), devolve o arquivo
    inteiro sem recortar — mostrar demais é mais seguro que recortar
    errado.

    Retorna (linhas_recortadas, número_da_primeira_linha) — o número
    mantém a numeração real do arquivo, para bater com `source_line` que
    os blocos do CFG já carregam.
    """
    if not kernel_name or not source_lines:
        return source_lines, 1

    pattern = re.compile(r"\b" + re.escape(kernel_name) + r"\s*\(")
    start_idx = next((i for i, line in enumerate(source_lines) if pattern.search(line)), None)
    if start_idx is None:
        return source_lines, 1

    doc_start = start_idx
    i = start_idx - 1
    while i >= 0 and (source_lines[i].strip() == "" or source_lines[i].strip().startswith(("*", "/*", "//"))):
        doc_start = i
        i -= 1

    depth = 0
    opened = False
    end_idx = len(source_lines) - 1
    for i in range(start_idx, len(source_lines)):
        for ch in source_lines[i]:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    end_idx = i
                    break
        if opened and depth == 0:
            break

    return source_lines[doc_start:end_idx + 1], doc_start + 1


def _build_payload(result: dict,
                    warp_size: int = 32,
                    meta: Optional[dict] = None,
                    source_path: Optional[str] = None,
                    kernel_name: Optional[str] = None) -> dict:
    control_flow = result.get("control_flow") or {}
    dynamic = result.get("dynamic_flow") or {}

    blocks: dict = control_flow.get("blocks", {})
    order: list = control_flow.get("order", [])
    raw_edges: list = control_flow.get("edges", [])

    visible_labels = [l for l in order if l in blocks and not _is_jump_stub(blocks[l])]
    visible_set = set(visible_labels)

    def resolve(label: str) -> str:
        return label if label in visible_set else _resolve_visual_target(blocks, label)

    # ── arestas: só emite a partir de blocos visíveis; alvo é resolvido
    # através da cadeia de jump-stubs até um bloco real ──────────────────
    seen_edges: set[tuple] = set()
    edge_list: list[dict] = []
    for e in raw_edges:
        src = e.get("source")
        if src not in visible_set:
            continue
        dst = resolve(e.get("target"))
        if dst not in visible_set:
            continue
        etype = e.get("edge_type")
        key = (src, dst, etype)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edge_list.append({
            "id": f"e{len(edge_list)}",
            "source": src,
            "target": dst,
            "edge_type": etype,
            "is_back_edge": bool(e.get("is_back_edge")),
        })

    branch_sources = {e["source"] for e in edge_list if e["edge_type"] == "conditional"}

    nodes = []
    for label in visible_labels:
        summary = _block_summary(label, blocks[label])
        summary["is_branch"] = label in branch_sources
        if summary["is_entry"]:
            kind = "entry"
        elif summary["is_terminal"]:
            kind = "terminal"
        elif summary["is_branch"]:
            kind = "branch"
        else:
            kind = "normal"
        summary["kind"] = kind
        nodes.append(summary)
    graph_groups = _compute_graph_groups(blocks, order, visible_labels, control_flow)

    # ── caminho de cada thread, com blocos-stub colapsados no bloco real
    # que representam e visitas consecutivas repetidas fundidas ─────────
    def display_path(raw_path: list[str]) -> list[str]:
        out: list[str] = []
        for label in raw_path:
            if label not in blocks:
                continue
            resolved = resolve(label)
            if resolved not in visible_set:
                continue
            if not out or out[-1] != resolved:
                out.append(resolved)
        return out

    threads_raw = dynamic.get("threads", [])
    thread_rows = []
    max_steps = 0
    for t in threads_raw:
        dp = display_path(t.get("path", []))
        max_steps = max(max_steps, len(dp))
        thread_rows.append({
            "thread_id": t.get("thread_id"),
            "warp_id": t.get("warp_id"),
            "lane": t.get("lane"),
            "path": dp,
            "halt_reason": t.get("halt_reason") or "",
            "unsupported_ops": t.get("unsupported_ops") or [],
        })

    # ── snapshots de memória (só disponíveis para uma thread de
    # referência) alinhados/colapsados do mesmo jeito que os caminhos ───
    thread0_frames = []
    for frame in dynamic.get("step_frames", []):
        raw_labels = frame.get("active_labels") or []
        if not raw_labels or raw_labels[0] not in blocks:
            continue
        label = resolve(raw_labels[0])
        if label not in visible_set:
            continue
        state = frame.get("state", {})
        if thread0_frames and thread0_frames[-1]["label"] == label:
            thread0_frames[-1]["state"] = state
        else:
            thread0_frames.append({"label": label, "state": state})
    for idx, item in enumerate(thread0_frames):
        item["step"] = idx

    frame_label_seq = [f["label"] for f in thread0_frames]
    reference_thread_id = None
    for row in thread_rows:
        if row["path"] == frame_label_seq and frame_label_seq:
            reference_thread_id = row["thread_id"]
            break

    full_source_lines = _load_source_lines(source_path)
    source_lines, source_line_offset = _extract_kernel_source(full_source_lines, kernel_name)

    payload = {
        "meta": dict(meta or {}),
        "nodes": nodes,
        "graph_groups": graph_groups,
        "edges": edge_list,
        "threads": thread_rows,
        "total_threads": len(thread_rows),
        "warp_size": warp_size,
        "max_steps": max_steps,
        "reference_thread_id": reference_thread_id,
        "reference_frames": thread0_frames,
        "sample_input": dynamic.get("sample_input", {}),
        "sample_output": dynamic.get("sample_output", {}),
        "validation": dynamic.get("validation", []),
        "notes": dynamic.get("notes", []),
        "branch_activity": dynamic.get("branch_activity", []),
        "unsupported_ops": dynamic.get("unsupported_ops", []),
        "source_name": os.path.basename(source_path) if source_path else "",
        "source_lines": source_lines,
        "source_line_offset": source_line_offset,
    }
    return payload


# ──────────────────────────────────────────────────────────────────────────
# 2) Renderização HTML/CSS/JS (Cytoscape.js embutido, sem CDN)
# ──────────────────────────────────────────────────────────────────────────

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def _read_vendor_js(name: str) -> str:
    return (_VENDOR_DIR / name).read_text(encoding="utf-8").replace("</script", "<\\/script")


_CYTOSCAPE_JS = _read_vendor_js("cytoscape.min.js")
_DAGRE_JS = _read_vendor_js("dagre.min.js")
_CYTOSCAPE_DAGRE_JS = _read_vendor_js("cytoscape-dagre.min.js")

# Os três arquivos acima já deixam `window.cytoscape`/`window.dagre`/
# `window.cytoscapeDagre` definidos só de serem executados (são embutidos
# como <script> antes deste bloco, em ordem) — não tem fetch nem espera
# assíncrona envolvida. Só falta registrar o layout dagre no cytoscape:
# o build UMD de cytoscape-dagre carregado via <script> simples não faz
# isso sozinho, só expõe a factory em `window.cytoscapeDagre`.
_LIB_REGISTER_JS = r"""
(function () {
  if (window.cytoscape && window.cytoscapeDagre && !window.__ptxvizDagreRegistered) {
    window.cytoscape.use(window.cytoscapeDagre);
    window.__ptxvizDagreRegistered = true;
  }
})();
"""

_APP_JS = r"""
(function () {
  var root = document.getElementById("__UID__-root");
  if (!root) return;
  var DATA = JSON.parse(document.getElementById("__UID__-data").textContent);

  var els = {
    cy: document.getElementById("__UID__-cy"),
    fallback: root.querySelector(".ptxviz-fallback"),
    slider: document.getElementById("__UID__-slider"),
    stepLabel: document.getElementById("__UID__-steplabel"),
    scopeSelect: document.getElementById("__UID__-scope"),
    threadInfo: document.getElementById("__UID__-threadinfo"),
    threadTableBody: document.querySelector("#__UID__-threadtable tbody"),
    memBox: document.getElementById("__UID__-membox"),
    aggList: document.getElementById("__UID__-agglist"),
    codeChips: document.getElementById("__UID__-codechips"),
    codeFallback: document.getElementById("__UID__-codefallback"),
    codeBody: document.getElementById("__UID__-codebody"),
    playBtn: root.querySelector('[data-act="play"]'),
    tabs: root.querySelectorAll(".ptxviz-tabs button"),
    panels: root.querySelectorAll(".ptxviz-tabpanel"),
    timelineGroupSelect: document.getElementById("__UID__-tl-group"),
    timelineInfo: document.getElementById("__UID__-tl-info"),
    timelineTable: document.getElementById("__UID__-tl-table"),
  };

  var KIND_COLOR = {
    entry: "#0f766e",
    terminal: "#b91c1c",
    branch: "#b45309",
    normal: "#334155",
  };
  var EDGE_COLOR = {
    conditional: "#d97706",
    fallthrough: "#94a3b8",
    jump: "#2563eb",
  };

  function nodeById(id) {
    for (var i = 0; i < DATA.nodes.length; i++) if (DATA.nodes[i].id === id) return DATA.nodes[i];
    return null;
  }

  // Cor estável por bloco (não por "tipo de bloco"): dois blocos diferentes
  // quase nunca caem na mesma cor (ângulo áureo distribui o matiz), então
  // na timeline dá pra notar "essas duas threads estão no mesmo bloco"
  // (mesma cor na mesma coluna) sem precisar ler o rótulo.
  function blockColor(id) {
    var hash = 0;
    for (var i = 0; i < id.length; i++) { hash = (hash * 31 + id.charCodeAt(i)) | 0; }
    var hue = Math.abs(hash * 137.508) % 360;
    return "hsl(" + hue.toFixed(1) + ", 62%, 55%)";
  }

  function threadBlockAtStep(thread, step) {
    if (!thread.path.length) return null;
    var idx = Math.min(step, thread.path.length - 1);
    return thread.path[idx];
  }

  // Agrupa quem está em cada bloco num dado passo: blockId -> [thread_ids].
  // Mais de uma chave = divergência (o grupo se separou). Usado pela linha
  // "Diverg." da timeline, pelas células de warp e pela aba "Código".
  function computeBlockGroups(threads, step) {
    var groups = {};
    threads.forEach(function (t) {
      var label = threadBlockAtStep(t, step);
      if (!label) return;
      (groups[label] = groups[label] || []).push(t.thread_id);
    });
    return groups;
  }

  // ── escopo (todas / warp N / thread N) ──────────────────────────────
  function buildScopeOptions() {
    var sel = els.scopeSelect;
    if (!sel) return;
    sel.innerHTML = "";
    var optAll = document.createElement("option");
    optAll.value = "all";
    optAll.textContent = "Todas as threads (agregado)";
    sel.appendChild(optAll);

    var byWarp = {};
    DATA.threads.forEach(function (t) {
      (byWarp[t.warp_id] = byWarp[t.warp_id] || []).push(t);
    });
    Object.keys(byWarp).sort(function (a, b) { return Number(a) - Number(b); }).forEach(function (warpId) {
      var warpOpt = document.createElement("option");
      warpOpt.value = "warp:" + warpId;
      warpOpt.textContent = "Warp " + warpId + " (" + byWarp[warpId].length + " threads)";
      sel.appendChild(warpOpt);
    });
    var group = document.createElement("optgroup");
    group.label = "Thread individual";
    DATA.threads.forEach(function (t) {
      var o = document.createElement("option");
      o.value = "thread:" + t.thread_id;
      o.textContent = "Thread " + t.thread_id + " (warp " + t.warp_id + ", lane " + t.lane + ")";
      group.appendChild(o);
    });
    sel.appendChild(group);
  }

  function threadsInScope(scope) {
    if (scope === "all") return DATA.threads;
    if (scope.indexOf("warp:") === 0) {
      var warpId = scope.slice(5);
      return DATA.threads.filter(function (t) { return String(t.warp_id) === warpId; });
    }
    if (scope.indexOf("thread:") === 0) {
      var tid = Number(scope.slice(7));
      return DATA.threads.filter(function (t) { return t.thread_id === tid; });
    }
    return DATA.threads;
  }

  // ── estado ────────────────────────────────────────────────────────────
  var state = { step: 0, scope: "all", playing: false, playTimer: null };

  function computeCounts(step, threads) {
    var active = {};        // threads distintas paradas neste bloco agora
    var completed = {};     // threads distintas que já passaram por aqui (bounded por threads.length)
    var visits = {};        // total de passagens (conta laço revisitando o mesmo bloco várias vezes)
    var activeThreadIds = {};
    var activeEdges = {};
    var completedEdges = {};
    threads.forEach(function (t) {
      var path = t.path;
      if (!path.length) return;
      var upto = Math.min(step, path.length - 1);
      var seenNodes = {};
      var seenEdges = {};
      for (var i = 0; i <= upto; i++) {
        var label = path[i];
        visits[label] = (visits[label] || 0) + 1;
        if (!seenNodes[label]) {
          seenNodes[label] = true;
          completed[label] = (completed[label] || 0) + 1;
        }
        if (i > 0) {
          var key = path[i - 1] + "->" + label;
          if (!seenEdges[key]) {
            seenEdges[key] = true;
            completedEdges[key] = (completedEdges[key] || 0) + 1;
          }
          if (i === upto) activeEdges[key] = (activeEdges[key] || 0) + 1;
        }
      }
      var current = path[upto];
      active[current] = (active[current] || 0) + 1;
      (activeThreadIds[current] = activeThreadIds[current] || []).push(t.thread_id);
    });
    return {
      active: active, completed: completed, visits: visits, activeThreadIds: activeThreadIds,
      activeEdges: activeEdges, completedEdges: completedEdges,
    };
  }

  var cy = null;

  function initCytoscape() {
    var elements = [];
    var groupByMember = {};
    (DATA.graph_groups || []).forEach(function (g) {
      elements.push({
        data: {
          id: g.id,
          label: g.label,
          kind: g.kind,
          is_group: 1,
          heat: 0,
          active: 0,
          baseLabel: g.label,
          display: g.label,
        },
      });
      (g.members || []).forEach(function (memberId) {
        groupByMember[memberId] = g.id;
      });
    });
    DATA.nodes.forEach(function (n) {
      var titleLine = n.label;
      var detailLine = n.description || n.instruction_name || "";
      var ptxLine = n.ptx_name ? ("PTX: " + n.ptx_name) : "";
      var baseLabel = [titleLine, detailLine, ptxLine].filter(Boolean).join("\n");
      elements.push({
        data: {
          id: n.id, kind: n.kind, heat: 0, active: 0,
          parent: groupByMember[n.id] || "",
          baseLabel: baseLabel, display: baseLabel,
          is_group: 0,
        },
      });
    });
    DATA.edges.forEach(function (e) {
      elements.push({
        data: {
          id: e.id, source: e.source, target: e.target,
          edge_type: e.edge_type, is_back_edge: e.is_back_edge, heat: 0,
        },
      });
    });

    cy = window.cytoscape({
      container: els.cy,
      elements: elements,
      wheelSensitivity: 0.25,
      style: [
        {
          selector: "node",
          style: {
            shape: "round-rectangle",
            label: "data(display)",
            "text-wrap": "wrap",
            "font-size": "11px",
            color: "#0f172a",
          },
        },
        {
          selector: "node[is_group = 1]",
          style: {
            shape: "round-rectangle",
            "background-color": "#f8fafc",
            "background-opacity": 0.22,
            "border-width": 2,
            "border-color": "#94a3b8",
            "border-style": "solid",
            "text-valign": "top",
            "text-halign": "center",
            "font-size": "12px",
            "font-weight": 700,
            color: "#1e293b",
            padding: "18px",
          },
        },
        {
          selector: 'node[kind = "compound_decision"]',
          style: {
            "border-color": "#ef4444",
            "background-color": "#fff1f2",
          },
        },
        {
          selector: 'node[kind = "unroll"]',
          style: {
            "border-color": "#60a5fa",
            "background-color": "#eff6ff",
          },
        },
        {
          selector: 'node[kind = "remainder"]',
          style: {
            "border-color": "#f59e0b",
            "background-color": "#fffbeb",
          },
        },
        {
          selector: "node[is_group = 0]",
          style: {
            shape: "round-rectangle",
            "background-color": function (ele) { return KIND_COLOR[ele.data("kind")] || "#334155"; },
            "background-opacity": 0.12,
            "border-width": 2,
            "border-color": function (ele) { return KIND_COLOR[ele.data("kind")] || "#334155"; },
            "text-max-width": "230px",
            "font-size": "11px",
            color: "#0f172a",
            "text-valign": "center",
            "text-halign": "center",
            padding: "10px",
            width: "label",
            height: "label",
          },
        },
        {
          selector: "node[heat > 0]",
          style: {
            "background-opacity": "mapData(heat, 0, 1, 0.12, 0.85)",
            color: function (ele) { return ele.data("heat") > 0.55 ? "#ffffff" : "#0f172a"; },
          },
        },
        {
          selector: "node[active > 0]",
          style: {
            "border-width": 4,
            "border-color": "#f59e0b",
            "border-style": "solid",
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.6,
            "line-color": function (ele) { return EDGE_COLOR[ele.data("edge_type")] || "#94a3b8"; },
            "target-arrow-color": function (ele) { return EDGE_COLOR[ele.data("edge_type")] || "#94a3b8"; },
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "line-style": function (ele) { return ele.data("is_back_edge") ? "dashed" : "solid"; },
            opacity: 0.55,
          },
        },
        {
          selector: "edge[heat > 0]",
          style: {
            width: "mapData(heat, 0, 1, 1.6, 5)",
            opacity: 1,
          },
        },
        {
          selector: ".ptxviz-thread-path",
          style: {
            "line-color": "#7c3aed",
            "target-arrow-color": "#7c3aed",
            width: 4,
            opacity: 1,
            "z-index": 999,
          },
        },
        {
          selector: ".ptxviz-thread-node",
          style: {
            "border-color": "#7c3aed",
            "border-width": 4,
          },
        },
      ],
      layout: { name: "preset" },
    });

    var layout = cy.layout({
      name: "dagre",
      rankDir: "LR",
      nodeSep: 24,
      rankSep: 70,
      padding: 20,
    });
    layout.one("layoutstop", function () {
      // O container tem uma altura fixa via CSS, mas o CFG pode ser bem
      // largo e raso (poucos ramos) ou mais alto (muitos branches
      // paralelos). Ajustar a altura do container à proporção real do
      // conteúdo evita tanto "grafo minúsculo com espaço em branco sobrando"
      // quanto "conteúdo minúsculo dentro de um canvas gigante" — cy.fit()
      // sozinho preserva a proporção, então sem isso um grafo bem largo e
      // raso ficaria com uma faixa fina no meio de uma área alta vazia.
      cy.resize();
      var bbox = cy.elements().boundingBox();
      var availableWidth = els.cy.clientWidth || 900;
      var aspect = bbox.h / Math.max(bbox.w, 1);
      var targetHeight = Math.max(220, Math.min(640, Math.round(availableWidth * aspect) + 60));
      els.cy.style.height = targetHeight + "px";
      cy.resize();
      cy.fit(undefined, 24);
    });
    layout.run();

    if (window.ResizeObserver) {
      new ResizeObserver(function () { cy.resize(); }).observe(els.cy.parentElement);
    }
    window.addEventListener("resize", function () { cy.resize(); });
  }

  function applyStep() {
    var threads = threadsInScope(state.scope);
    var counts = computeCounts(state.step, threads);
    var totalInScope = Math.max(threads.length, 1);

    if (cy) {
      cy.batch(function () {
        cy.nodes().forEach(function (n) {
          if (n.data("is_group")) return;
          var id = n.id();
          var done = counts.completed[id] || 0;
          var visits = counts.visits[id] || 0;
          var nowIds = counts.activeThreadIds[id] || [];
          n.data("heat", done / totalInScope);
          n.data("active", nowIds.length);
          var statusBits = [done + "/" + totalInScope];
          if (visits > done) statusBits.push(visits + "x");
          if (nowIds.length) {
            var shown = nowIds.slice(0, 4).map(function (tid) { return "T" + tid; }).join(",");
            statusBits.push("agora: " + shown + (nowIds.length > 4 ? " +" + (nowIds.length - 4) : ""));
          }
          n.data("display", n.data("baseLabel") + "\n" + statusBits.join(" · "));
        });
        cy.edges().forEach(function (e) {
          var key = e.data("source") + "->" + e.data("target");
          var c = counts.completedEdges[key] || 0;
          e.data("heat", c / totalInScope);
        });
        cy.elements(".ptxviz-thread-path, .ptxviz-thread-node").removeClass("ptxviz-thread-path ptxviz-thread-node");
        if (state.scope.indexOf("thread:") === 0) {
          highlightSingleThreadPath(threads[0], state.step);
        }
      });
    }

    if (els.stepLabel) els.stepLabel.textContent = "Passo " + (state.step + 1) + " de " + Math.max(DATA.max_steps, 1);
    renderThreadTable(threads);
    renderMemory();
    renderCodePanel();
    updateTimelineHighlight();
  }

  function highlightSingleThreadPath(thread, step) {
    if (!thread) return;
    var upto = Math.min(step, thread.path.length - 1);
    for (var i = 0; i <= upto; i++) {
      var n = cy.getElementById(thread.path[i]);
      if (n) n.addClass("ptxviz-thread-node");
      if (i > 0) {
        var edge = cy.edges('[source = "' + thread.path[i - 1] + '"][target = "' + thread.path[i] + '"]');
        edge.addClass("ptxviz-thread-path");
      }
    }
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
    });
  }

  function renderThreadTable(threads) {
    if (!els.threadTableBody) return;
    var rows = threads.slice(0, 500).map(function (t) {
      var idx = Math.min(state.step, t.path.length - 1);
      var current = idx >= 0 ? t.path[idx] : "-";
      var finished = state.step >= t.path.length - 1;
      var node = nodeById(current);
      var currentLabel = node ? node.label : current;
      return (
        "<tr>" +
        "<td>" + t.thread_id + "</td>" +
        "<td>" + t.warp_id + "</td>" +
        "<td>" + t.lane + "</td>" +
        "<td>" + escapeHtml(currentLabel) + "</td>" +
        "<td>" + (finished ? escapeHtml(t.halt_reason || "concluída") : "em execução") + "</td>" +
        "</tr>"
      );
    });
    els.threadTableBody.innerHTML = rows.join("") || "<tr><td colspan='5'>Sem threads.</td></tr>";
    if (threads.length > 500) {
      els.threadInfo.textContent = "Mostrando 500 de " + threads.length + " threads no escopo atual.";
    } else {
      els.threadInfo.textContent = threads.length + " thread(s) no escopo atual.";
    }
  }

  function renderMemory() {
    if (!els.memBox) return;
    var frames = DATA.reference_frames;
    if (!frames.length) {
      els.memBox.innerHTML = "<div class='ptxviz-muted'>Nenhum snapshot de memória disponível.</div>";
      return;
    }
    var idx = Math.min(state.step, frames.length - 1);
    var frame = frames[idx];
    var refLabel = DATA.reference_thread_id !== null && DATA.reference_thread_id !== undefined
      ? ("thread " + DATA.reference_thread_id)
      : "thread de referência";
    var rows = Object.keys(frame.state).map(function (key) {
      return "<div class='ptxviz-kv'><div class='ptxviz-kv-key'>" + escapeHtml(key) + "</div>" +
        "<div class='ptxviz-kv-val'>" + escapeHtml(JSON.stringify(frame.state[key])) + "</div></div>";
    });
    els.memBox.innerHTML =
      "<div class='ptxviz-muted'>Estado de memória em '" + escapeHtml(frame.label) + "' (" + refLabel + ")</div>" +
      rows.join("");
  }

  var CODE_GROUP_PALETTE = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#be185d", "#65a30d"];

  // Mostra, no passo atual, em que bloco cada grupo de threads está —
  // e se houver mais de um bloco ao mesmo tempo (divergência), destaca
  // cada linha de código correspondente com uma cor diferente e lista
  // quais threads estão em cada uma. Granularidade é por bloco básico
  // (o que o CFG/trace rastreiam), não por instrução PTX individual.
  function renderCodePanel() {
    if (!els.codeChips) return;
    var threads = threadsInScope(state.scope);
    var groups = computeBlockGroups(threads, state.step);
    var blockIds = Object.keys(groups);
    var lineColorByNumber = {};

    var chips = blockIds.map(function (id, idx) {
      var node = nodeById(id);
      var color = CODE_GROUP_PALETTE[idx % CODE_GROUP_PALETTE.length];
      if (node && node.source_line) lineColorByNumber[node.source_line] = color;
      var ids = groups[id];
      var shown = ids.slice(0, 6).map(function (tid) { return "T" + tid; }).join(", ");
      var extra = ids.length > 6 ? " +" + (ids.length - 6) : "";
      var label = node ? node.label : id;
      var lineInfo = node && node.source_line ? ("linha " + node.source_line) : "sem linha .cu associada";
      var instr = node && node.instruction_name ? escapeHtml(node.instruction_name) : "";
      var lineAttr = node && node.source_line ? (" data-goto-line='" + node.source_line + "'") : "";
      return (
        "<div class='ptxviz-code-chip'" + lineAttr + " style='border-left:4px solid " + color + "'>" +
        "<b>" + escapeHtml(label) + "</b> <span class='ptxviz-muted'>(" + escapeHtml(lineInfo) + ")</span><br>" +
        ids.length + " thread(s): " + escapeHtml(shown + extra) +
        (instr ? "<div class='ptxviz-muted'>" + instr + "</div>" : "") +
        "</div>"
      );
    });
    els.codeChips.innerHTML = chips.join("") || "<div class='ptxviz-muted'>Nenhuma thread no escopo atual.</div>";
    if (blockIds.length > 1) {
      els.codeChips.insertAdjacentHTML("afterbegin", "<div class='ptxviz-code-diverge-banner'>⚠ Divergência: " + blockIds.length + " blocos diferentes neste passo</div>");
    }
    els.codeChips.querySelectorAll("[data-goto-line]").forEach(function (chip) {
      chip.style.cursor = "pointer";
      chip.addEventListener("click", function () {
        var row = els.codeBody.querySelector('.ptxviz-code-line[data-line="' + chip.getAttribute("data-goto-line") + '"]');
        if (row) row.scrollIntoView({ block: "center" });
      });
    });

    if (!DATA.source_lines || !DATA.source_lines.length) {
      els.codeFallback.hidden = false;
      els.codeBody.innerHTML = "";
      return;
    }
    els.codeFallback.hidden = true;

    if (!els.codeBody.dataset.built) {
      var offset = DATA.source_line_offset || 1;
      els.codeBody.innerHTML = DATA.source_lines.map(function (text, idx) {
        var lineNo = idx + offset;
        return (
          "<div class='ptxviz-code-line' data-line='" + lineNo + "'>" +
          "<span class='ptxviz-code-lineno'>" + lineNo + "</span>" +
          "<span class='ptxviz-code-text'>" + escapeHtml(text) + "</span>" +
          "</div>"
        );
      }).join("");
      els.codeBody.dataset.built = "1";
    }
    els.codeBody.querySelectorAll(".ptxviz-code-line").forEach(function (row) {
      row.style.background = "";
      row.style.borderLeft = "";
      row.classList.remove("ptxviz-code-current");
    });
    Object.keys(lineColorByNumber).forEach(function (lineNo) {
      var row = els.codeBody.querySelector('.ptxviz-code-line[data-line="' + lineNo + '"]');
      if (!row) return;
      var color = lineColorByNumber[lineNo];
      row.style.background = color + "1f";
      row.style.borderLeft = "4px solid " + color;
      row.classList.add("ptxviz-code-current");
    });
  }

  function renderStaticPanels() {
    if (els.aggList) {
      var aggRows = DATA.nodes.map(function (n) {
        var hits = 0;
        DATA.threads.forEach(function (t) { if (t.path.indexOf(n.id) !== -1) hits++; });
        return { n: n, hits: hits };
      }).sort(function (a, b) { return b.hits - a.hits; });
      els.aggList.innerHTML = aggRows.map(function (row) {
        var pct = DATA.total_threads ? Math.round((row.hits / DATA.total_threads) * 100) : 0;
        return (
          "<div class='ptxviz-agg-row'>" +
          "<div class='ptxviz-agg-label'>" + escapeHtml(row.n.label) + "</div>" +
          "<div class='ptxviz-agg-bar'><div class='ptxviz-agg-fill' style='width:" + pct + "%'></div></div>" +
          "<div class='ptxviz-agg-count'>" + row.hits + "/" + DATA.total_threads + "</div>" +
          "</div>"
        );
      }).join("");
    }

    var notesEl = root.querySelector(".ptxviz-notes");
    if (notesEl) {
      notesEl.innerHTML = DATA.notes.map(function (n) { return "<li>" + escapeHtml(n) + "</li>"; }).join("");
    }

    var valEl = root.querySelector(".ptxviz-validation");
    if (valEl) {
      if (!DATA.validation.length) {
        valEl.innerHTML = "<div class='ptxviz-muted'>Nenhuma saída esperada informada.</div>";
      } else {
        valEl.innerHTML = DATA.validation.map(function (v) {
          var status = v.match ? "OK" : "DIVERGE";
          return (
            "<div class='ptxviz-kv'><div class='ptxviz-kv-key'>" + escapeHtml(v.buffer) + " [" + status + "]</div>" +
            "<div class='ptxviz-kv-val'>esperado=" + escapeHtml(JSON.stringify(v.expected)) +
            " obtido=" + escapeHtml(JSON.stringify(v.actual)) + "</div></div>"
          );
        }).join("");
      }
    }

    var ioEl = root.querySelector(".ptxviz-io");
    if (ioEl) {
      var ioRows = [];
      Object.keys(DATA.sample_input).forEach(function (k) {
        ioRows.push("<div class='ptxviz-kv'><div class='ptxviz-kv-key'>entrada: " + escapeHtml(k) + "</div><div class='ptxviz-kv-val'>" + escapeHtml(JSON.stringify(DATA.sample_input[k])) + "</div></div>");
      });
      Object.keys(DATA.sample_output).forEach(function (k) {
        ioRows.push("<div class='ptxviz-kv'><div class='ptxviz-kv-key'>saída: " + escapeHtml(k) + "</div><div class='ptxviz-kv-val'>" + escapeHtml(JSON.stringify(DATA.sample_output[k])) + "</div></div>");
      });
      ioEl.innerHTML = ioRows.join("");
    }
  }

  // ── linha do tempo (swimlane): todas as threads/warps de uma vez, uma
  // coluna por passo lógico. Diferente do grafo, aqui uma volta de laço
  // vira colunas repetidas em sequência em vez de uma aresta sobreposta
  // nela mesma — não tem ambiguidade sobre "quantas vezes" ou "em que
  // ordem" um bloco foi revisitado. Construída uma vez (o conteúdo de
  // cada célula não depende do passo selecionado); só o destaque da
  // coluna atual muda a cada passo. ────────────────────────────────────
  var MAX_TIMELINE_ROWS = 500;
  var MAX_TIMELINE_STEPS = 400;

  function buildTimeline() {
    if (!els.timelineTable) return;
    var group = els.timelineGroupSelect.value;
    var stepsShown = Math.min(Math.max(DATA.max_steps, 1), MAX_TIMELINE_STEPS);

    var theadCells = ['<th class="ptxviz-tl-rowlabel ptxviz-tl-corner">' + (group === "warp" ? "Warp" : "Thread") + "</th>"];
    for (var s = 0; s < stepsShown; s++) {
      theadCells.push('<th class="ptxviz-tl-step" data-step="' + s + '">' + (s + 1) + "</th>");
    }

    var rowsHtml = [];
    if (group === "warp") {
      var byWarp = {};
      DATA.threads.forEach(function (t) { (byWarp[t.warp_id] = byWarp[t.warp_id] || []).push(t); });
      Object.keys(byWarp).sort(function (a, b) { return Number(a) - Number(b); }).forEach(function (warpId) {
        var members = byWarp[warpId];
        var cells = ['<th class="ptxviz-tl-rowlabel" data-warp="' + warpId + '" title="Selecionar warp ' + warpId + '">W' + warpId + "</th>"];
        for (var s = 0; s < stepsShown; s++) {
          var groups = computeBlockGroups(members, s);
          var keys = Object.keys(groups);
          var bg = "#e2e8f0";
          var title = "";
          if (keys.length === 1) {
            var node = nodeById(keys[0]);
            bg = blockColor(keys[0]);
            title = (node ? node.label : keys[0]) + " — " + members.length + " thread(s)";
          } else if (keys.length > 1) {
            var stripes = [];
            var segw = 10;
            keys.forEach(function (k, i) {
              var c = blockColor(k);
              stripes.push(c + " " + (i * segw) + "px", c + " " + ((i + 1) * segw) + "px");
            });
            bg = "repeating-linear-gradient(45deg," + stripes.join(",") + ")";
            title = "Divergência no warp: " + keys.map(function (k) {
              var n = nodeById(k);
              return (n ? n.label : k) + "=" + groups[k].length;
            }).join(", ");
          }
          cells.push('<td class="ptxviz-tl-cell" data-step="' + s + '" data-warp="' + warpId + '" style="background:' + bg + '" title="' + escapeHtml(title) + '"></td>');
        }
        rowsHtml.push("<tr>" + cells.join("") + "</tr>");
      });
    } else {
      var shownThreads = DATA.threads.slice(0, MAX_TIMELINE_ROWS);

      // Linha-resumo: marca em vermelho qualquer coluna onde as threads
      // mostradas NÃO estão todas no mesmo bloco — sem isso, notar 2-3
      // colunas divergentes entre dezenas de células com cores parecidas
      // é fácil de passar batido.
      var divergeCells = ['<th class="ptxviz-tl-rowlabel" style="cursor:default;font-weight:400;color:#64748b;">Diverg.</th>'];
      for (var ds = 0; ds < stepsShown; ds++) {
        var groups = computeBlockGroups(shownThreads, ds);
        var keys = Object.keys(groups);
        var diverges = keys.length > 1;
        var dTitle = diverges
          ? "Divergência: " + keys.map(function (k) { var n = nodeById(k); return (n ? n.label : k) + "=" + groups[k].length; }).join(", ")
          : "todas as threads no mesmo bloco";
        divergeCells.push('<td class="ptxviz-tl-cell" data-step="' + ds + '" style="background:' + (diverges ? "#dc2626" : "#e5e7eb") + '" title="' + escapeHtml(dTitle) + '"></td>');
      }
      rowsHtml.push("<tr>" + divergeCells.join("") + "</tr>");

      shownThreads.forEach(function (t) {
        var cells = ['<th class="ptxviz-tl-rowlabel" data-thread="' + t.thread_id + '" title="Selecionar thread ' + t.thread_id + '">T' + t.thread_id + "</th>"];
        for (var s = 0; s < stepsShown; s++) {
          var label = threadBlockAtStep(t, s);
          var node = label ? nodeById(label) : null;
          var finished = label && s >= t.path.length - 1;
          var title = node ? (node.label + (finished ? " (finalizada aqui)" : "")) : "";
          cells.push('<td class="ptxviz-tl-cell" data-step="' + s + '" data-thread="' + t.thread_id + '" style="background:' + (label ? blockColor(label) : "#e2e8f0") + '" title="' + escapeHtml(title) + '"></td>');
        }
        rowsHtml.push("<tr>" + cells.join("") + "</tr>");
      });
    }

    els.timelineTable.innerHTML = "<thead><tr>" + theadCells.join("") + "</tr></thead><tbody>" + rowsHtml.join("") + "</tbody>";

    var infoBits = [];
    if (group === "thread" && DATA.total_threads > MAX_TIMELINE_ROWS) {
      infoBits.push("mostrando " + MAX_TIMELINE_ROWS + " de " + DATA.total_threads + " threads");
    }
    if (DATA.max_steps > MAX_TIMELINE_STEPS) {
      infoBits.push("mostrando os primeiros " + MAX_TIMELINE_STEPS + " de " + DATA.max_steps + " passos");
    }
    els.timelineInfo.textContent = infoBits.join(" · ");

    els.timelineTable.querySelectorAll("td.ptxviz-tl-cell[data-thread]").forEach(function (td) {
      td.addEventListener("click", function () { selectThreadFromTimeline(td.getAttribute("data-thread"), Number(td.getAttribute("data-step"))); });
    });
    els.timelineTable.querySelectorAll("th.ptxviz-tl-rowlabel[data-thread]").forEach(function (th) {
      th.addEventListener("click", function () { selectThreadFromTimeline(th.getAttribute("data-thread"), state.step); });
    });
    els.timelineTable.querySelectorAll("[data-warp]").forEach(function (el) {
      el.addEventListener("click", function () {
        var warpId = el.getAttribute("data-warp");
        var step = el.hasAttribute("data-step") ? Number(el.getAttribute("data-step")) : state.step;
        if (els.scopeSelect) {
          els.scopeSelect.value = "warp:" + warpId;
          state.scope = els.scopeSelect.value;
        }
        state.step = step;
        if (els.slider) els.slider.value = String(step);
        applyStep();
      });
    });

    updateTimelineHighlight();
  }

  function selectThreadFromTimeline(threadId, step) {
    if (els.scopeSelect) {
      els.scopeSelect.value = "thread:" + threadId;
      state.scope = els.scopeSelect.value;
    }
    state.step = step;
    if (els.slider) els.slider.value = String(step);
    applyStep();
    var threadTab = root.querySelector('[data-tab="thread"]');
    if (threadTab) threadTab.click();
  }

  function updateTimelineHighlight() {
    if (!els.timelineTable) return;
    els.timelineTable.querySelectorAll(".ptxviz-tl-col-current").forEach(function (el) {
      el.classList.remove("ptxviz-tl-col-current");
    });
    els.timelineTable.querySelectorAll('[data-step="' + state.step + '"]').forEach(function (el) {
      el.classList.add("ptxviz-tl-col-current");
    });
  }

  function wireControls() {
    if (els.slider) {
      els.slider.max = String(Math.max(DATA.max_steps - 1, 0));
      els.slider.value = "0";
      els.slider.addEventListener("input", function () {
        state.step = Number(els.slider.value);
        applyStep();
      });
    }
    var prevBtn = root.querySelector('[data-act="prev"]');
    if (prevBtn) {
      prevBtn.addEventListener("click", function () {
        state.step = Math.max(0, state.step - 1);
        if (els.slider) els.slider.value = String(state.step);
        applyStep();
      });
    }
    var nextBtn = root.querySelector('[data-act="next"]');
    if (nextBtn) {
      nextBtn.addEventListener("click", function () {
        state.step = Math.min(DATA.max_steps - 1, state.step + 1);
        if (els.slider) els.slider.value = String(state.step);
        applyStep();
      });
    }
    if (els.playBtn) {
      els.playBtn.addEventListener("click", function () {
        state.playing = !state.playing;
        els.playBtn.textContent = state.playing ? "⏸" : "▶";
        if (state.playing) {
          state.playTimer = setInterval(function () {
            if (state.step >= DATA.max_steps - 1) {
              state.step = 0;
            } else {
              state.step += 1;
            }
            if (els.slider) els.slider.value = String(state.step);
            applyStep();
          }, 700);
        } else {
          clearInterval(state.playTimer);
        }
      });
    }
    if (els.scopeSelect) {
      els.scopeSelect.addEventListener("change", function () {
        state.scope = els.scopeSelect.value;
        applyStep();
      });
    }
    if (els.timelineGroupSelect) {
      els.timelineGroupSelect.addEventListener("change", buildTimeline);
    }
    root.querySelectorAll('[data-act="fit"]').forEach(function (b) {
      b.addEventListener("click", function () { if (cy) { cy.resize(); cy.fit(undefined, 30); } });
    });
    root.querySelectorAll('[data-act="zoom-in"]').forEach(function (b) {
      b.addEventListener("click", function () { if (cy) cy.zoom(cy.zoom() * 1.2); });
    });
    root.querySelectorAll('[data-act="zoom-out"]').forEach(function (b) {
      b.addEventListener("click", function () { if (cy) cy.zoom(cy.zoom() / 1.2); });
    });
    els.tabs.forEach(function (btn) {
      btn.addEventListener("click", function () {
        els.tabs.forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        var tab = btn.getAttribute("data-tab");
        els.panels.forEach(function (p) {
          p.hidden = p.getAttribute("data-panel") !== tab;
        });
      });
    });
  }

  function showFallback(message) {
    els.cy.hidden = true;
    els.fallback.hidden = false;
    els.fallback.querySelector(".ptxviz-fallback-msg").textContent = message;
    var list = DATA.nodes.map(function (n) {
      return "<li><b>" + escapeHtml(n.label) + "</b> (" + escapeHtml(n.id) + ") — " + escapeHtml(n.description || "") + "</li>";
    }).join("");
    els.fallback.querySelector(".ptxviz-fallback-nodes").innerHTML = list;
  }

  buildScopeOptions();
  renderStaticPanels();
  wireControls();
  buildTimeline();
  applyStep();

  // Cytoscape/dagre vêm embutidos como <script> logo acima deste bloco
  // (não via CDN — ver docstring do módulo), então já estão disponíveis
  // de forma síncrona quando esta função roda. Só existem quando a seção
  // "graph" foi pedida (`els.cy` só existe nesse caso); "só timeline" ou
  // "só código" nem chegam a ter esse peso embutido no HTML.
  if (els.cy) {
    if (!window.cytoscape) {
      showFallback("Cytoscape.js não inicializou. A linha do tempo e a tabela de threads abaixo continuam funcionando normalmente.");
    } else {
      try {
        initCytoscape();
        applyStep();
      } catch (err) {
        console.error("ptxviz init error", err);
        showFallback("Falha ao inicializar o grafo: " + String(err) + " — a linha do tempo e a tabela de threads abaixo continuam funcionando.");
      }
    }
  }
})();
"""

_CSS_TEMPLATE = r"""
#__UID__-root { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: #ffffff; }
#__UID__-root .ptxviz-header { padding: 10px 14px; background: #f8fafc; border-bottom: 1px solid #dbe4f0; display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: baseline; }
#__UID__-root .ptxviz-title { font-weight: 700; font-size: 15px; }
#__UID__-root .ptxviz-meta { font-size: 12px; color: #475569; }
#__UID__-root .ptxviz-body { display: flex; flex-wrap: wrap; }
#__UID__-root .ptxviz-graph-wrap { flex: 2 1 520px; min-width: 320px; display: flex; flex-direction: column; border-right: 1px solid #e2e8f0; }
#__UID__-root .ptxviz-toolbar { display: flex; gap: 8px; align-items: center; padding: 8px 10px; border-bottom: 1px solid #e2e8f0; flex-wrap: wrap; }
#__UID__-root .ptxviz-toolbar button { border: 1px solid #cbd5e1; background: #fff; border-radius: 6px; padding: 4px 10px; font-size: 12px; cursor: pointer; }
#__UID__-root .ptxviz-toolbar button:hover { background: #f1f5f9; }
#__UID__-root .ptxviz-legend { display: flex; gap: 10px; font-size: 11px; color: #475569; flex-wrap: wrap; margin-left: auto; }
#__UID__-root .ptxviz-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 4px; vertical-align: middle; }
#__UID__-cy { height: 480px; }
#__UID__-root .ptxviz-fallback { padding: 16px; }
#__UID__-root .ptxviz-fallback-msg { color: #b91c1c; font-weight: 600; margin-bottom: 10px; }
#__UID__-root .ptxviz-stepbar { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; background: #f8fafc; }
#__UID__-root .ptxviz-stepbar input[type=range] { flex: 1; }
#__UID__-root .ptxviz-stepbar button { border: 1px solid #cbd5e1; background: #fff; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
#__UID__-root .ptxviz-side { flex: 1 1 300px; min-width: 280px; max-width: 420px; display: flex; flex-direction: column; }
#__UID__-root .ptxviz-tabs { display: flex; border-bottom: 1px solid #e2e8f0; overflow-x: auto; }
#__UID__-root .ptxviz-tabs button { flex: none; border: none; background: transparent; padding: 8px 10px; font-size: 12px; cursor: pointer; border-bottom: 2px solid transparent; color: #475569; }
#__UID__-root .ptxviz-tabs button.active { border-bottom-color: #2563eb; color: #0f172a; font-weight: 600; }
#__UID__-root .ptxviz-tabpanel { padding: 10px 12px; overflow: auto; max-height: 480px; font-size: 12px; }
#__UID__-root .ptxviz-muted { color: #64748b; font-size: 11px; margin-bottom: 6px; }
#__UID__-root .ptxviz-agg-row { display: grid; grid-template-columns: 1fr 80px 50px; gap: 8px; align-items: center; padding: 3px 0; }
#__UID__-root .ptxviz-agg-bar { background: #e2e8f0; border-radius: 4px; overflow: hidden; height: 8px; }
#__UID__-root .ptxviz-agg-fill { background: #2563eb; height: 100%; }
#__UID__-root .ptxviz-agg-count { text-align: right; color: #475569; }
#__UID__-root table { width: 100%; border-collapse: collapse; font-size: 11px; }
#__UID__-root th, #__UID__-root td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #eef2f7; }
#__UID__-root select { width: 100%; margin-bottom: 8px; padding: 4px; border-radius: 6px; border: 1px solid #cbd5e1; }
#__UID__-root .ptxviz-kv { display: grid; grid-template-columns: 110px 1fr; gap: 8px; padding: 4px 0; border-bottom: 1px dashed #eef2f7; }
#__UID__-root .ptxviz-kv-key { font-weight: 600; }
#__UID__-root ul.ptxviz-notes, #__UID__-root ul { margin: 0; padding-left: 16px; }
#__UID__-root .ptxviz-timeline { border-top: 1px solid #e2e8f0; width: 100%; }
#__UID__-root .ptxviz-timeline-header { display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; flex-wrap: wrap; }
#__UID__-root .ptxviz-timeline-header h3 { font-size: 13px; margin: 0; }
#__UID__-root .ptxviz-timeline-header select { width: auto; margin: 0; font-size: 12px; }
#__UID__-root .ptxviz-timeline-scroll { overflow: auto; max-height: 280px; }
#__UID__-root .ptxviz-tl-table { border-collapse: collapse; table-layout: fixed; font-size: 10px; }
#__UID__-root .ptxviz-tl-table th, #__UID__-root .ptxviz-tl-table td { border: 1px solid #f1f5f9; padding: 0; }
#__UID__-root .ptxviz-tl-rowlabel { position: sticky; left: 0; background: #f8fafc; z-index: 2; padding: 1px 6px !important; text-align: right; cursor: pointer; white-space: nowrap; color: #334155; font-weight: 600; }
#__UID__-root .ptxviz-tl-rowlabel:hover { color: #2563eb; text-decoration: underline; }
#__UID__-root .ptxviz-tl-corner { position: sticky; top: 0; left: 0; z-index: 3; cursor: default; color: #64748b; font-weight: 400; }
#__UID__-root .ptxviz-tl-corner:hover { text-decoration: none; color: #64748b; }
#__UID__-root .ptxviz-tl-step { position: sticky; top: 0; background: #f8fafc; z-index: 1; font-weight: 400; color: #94a3b8; width: 16px; min-width: 16px; text-align: center; }
#__UID__-root .ptxviz-tl-cell { width: 16px; min-width: 16px; height: 15px; cursor: pointer; }
#__UID__-root .ptxviz-tl-cell.ptxviz-tl-col-current, #__UID__-root .ptxviz-tl-step.ptxviz-tl-col-current { outline: 2px solid #0f172a; outline-offset: -1px; }
#__UID__-root .ptxviz-codeview { border-top: 1px solid #e2e8f0; width: 100%; padding: 0 0 12px 0; }
#__UID__-root .ptxviz-codeview > .ptxviz-code-diverge-banner, #__UID__-root .ptxviz-codeview > #__UID__-codechips, #__UID__-root .ptxviz-codeview > .ptxviz-code, #__UID__-root .ptxviz-codeview > .ptxviz-muted { margin-left: 12px; margin-right: 12px; }
#__UID__-codechips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; margin-bottom: 8px; }
#__UID__-root .ptxviz-code-chip { flex: 1 1 220px; max-width: 320px; border-radius: 6px; background: #f8fafc; padding: 6px 10px; font-size: 12px; }
#__UID__-root .ptxviz-code-diverge-banner { flex-basis: 100%; background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; border-radius: 6px; padding: 6px 10px; font-size: 12px; font-weight: 600; }
#__UID__-root .ptxviz-code { font-family: ui-monospace, Consolas, "Courier New", monospace; font-size: 13px; line-height: 1.5; overflow: auto; max-height: 420px; border: 1px solid #eef2f7; border-radius: 6px; }
#__UID__-root .ptxviz-code-line { display: flex; white-space: pre; padding: 0 6px; }
#__UID__-root .ptxviz-code-lineno { flex: none; width: 40px; color: #94a3b8; text-align: right; margin-right: 10px; user-select: none; }
#__UID__-root .ptxviz-code-text { white-space: pre; }
#__UID__-root .ptxviz-code-line.ptxviz-code-current { font-weight: 600; }
"""

ALL_SECTIONS = ("graph", "threads", "memory", "io", "timeline", "code")


def _normalize_sections(sections: Optional[Iterable[str]]) -> set[str]:
    if sections is None:
        return set(ALL_SECTIONS)
    normalized = {str(s).strip().lower() for s in sections}
    unknown = normalized - set(ALL_SECTIONS)
    if unknown:
        raise ValueError(
            f"sections desconhecida(s): {sorted(unknown)}. Opções válidas: {ALL_SECTIONS}."
        )
    return normalized


def _tab_button(key: str, label: str, active: bool) -> str:
    cls = ' class="active"' if active else ""
    return f'<button type="button" data-tab="{key}"{cls}>{label}</button>'


def _tab_panel(key: str, body: str, active: bool) -> str:
    hidden = "" if active else " hidden"
    return f'<div class="ptxviz-tabpanel" data-panel="{key}"{hidden}>{body}</div>'


def _render_fragment(payload: dict, title: str, height: str,
                     sections: Optional[Iterable[str]] = None) -> str:
    uid = f"ptxviz-{uuid.uuid4().hex[:10]}"
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    css = _CSS_TEMPLATE.replace("__UID__", uid)
    app_js = _APP_JS.replace("__UID__", uid)
    meta = payload.get("meta", {})
    meta_bits = []
    if meta.get("grid_dim"):
        meta_bits.append(f"grid={tuple(meta['grid_dim'])}")
    if meta.get("block_dim"):
        meta_bits.append(f"block={tuple(meta['block_dim'])}")
    meta_bits.append(f"warp_size={payload.get('warp_size')}")
    meta_bits.append(f"total_threads={payload.get('total_threads')}")
    meta_html = " · ".join(meta_bits)

    active_sections = _normalize_sections(sections)
    show_graph = "graph" in active_sections
    show_threads = "threads" in active_sections
    show_memory = "memory" in active_sections
    show_io = "io" in active_sections
    show_timeline = "timeline" in active_sections
    show_code = "code" in active_sections
    # "Agregado" é uma leitura textual do mesmo dado de calor do grafo —
    # não faz sentido como seção independente sem o grafo.
    show_agg = show_graph
    show_stepbar = payload.get("max_steps", 0) > 0

    tabs: list[tuple[str, str, str]] = []
    if show_agg:
        tabs.append(("agg", "Agregado",
            "<div class='ptxviz-muted'>Threads que já passaram por cada bloco (do total).</div>"
            f"<div id='{uid}-agglist'></div>"))
    if show_threads:
        tabs.append(("thread", "Threads",
            f"<select id='{uid}-scope'></select>"
            f"<div id='{uid}-threadinfo' class='ptxviz-muted'></div>"
            f"<table id='{uid}-threadtable'>"
            "<thead><tr><th>Thread</th><th>Warp</th><th>Lane</th><th>Bloco atual</th><th>Estado</th></tr></thead>"
            "<tbody></tbody></table>"))
    if show_memory:
        tabs.append(("mem", "Memória", f"<div id='{uid}-membox'></div>"))
    if show_io:
        tabs.append(("io", "E/S",
            "<div class='ptxviz-io'></div><h4>Validação</h4><div class='ptxviz-validation'></div>"
            "<h4>Notas</h4><ul class='ptxviz-notes'></ul>"))

    side_html = ""
    if tabs:
        buttons_html = "".join(_tab_button(k, label, idx == 0) for idx, (k, label, _b) in enumerate(tabs))
        panels_html = "".join(_tab_panel(k, body, idx == 0) for idx, (k, _label, body) in enumerate(tabs))
        side_html = (
            '<div class="ptxviz-side">'
            f'<div class="ptxviz-tabs">{buttons_html}</div>'
            f"{panels_html}"
            "</div>"
        )

    graph_wrap_html = ""
    if show_graph:
        graph_wrap_html = f"""
    <div class="ptxviz-graph-wrap">
      <div class="ptxviz-toolbar">
        <button type="button" data-act="fit">Ajustar</button>
        <button type="button" data-act="zoom-in">+</button>
        <button type="button" data-act="zoom-out">-</button>
        <div class="ptxviz-legend">
          <span><span class="ptxviz-swatch" style="background:#0f766e"></span>Entrada</span>
          <span><span class="ptxviz-swatch" style="background:#b91c1c"></span>Saída</span>
          <span><span class="ptxviz-swatch" style="background:#b45309"></span>Decisão</span>
          <span><span class="ptxviz-swatch" style="background:#334155"></span>Normal</span>
          <span><span class="ptxviz-swatch" style="background:#f59e0b;border-radius:50%"></span>Ativo agora</span>
          <span><span class="ptxviz-swatch" style="background:#7c3aed"></span>Thread selecionada</span>
        </div>
      </div>
      <div id="{uid}-cy" style="height:{_esc(height)}"></div>
      <div class="ptxviz-fallback" hidden>
        <div class="ptxviz-fallback-msg"></div>
        <ul class="ptxviz-fallback-nodes"></ul>
      </div>
    </div>"""

    body_html = ""
    if graph_wrap_html or side_html:
        body_html = f'<div class="ptxviz-body">{graph_wrap_html}{side_html}</div>'

    stepbar_html = ""
    if show_stepbar:
        stepbar_html = f"""
  <div class="ptxviz-stepbar">
    <button type="button" data-act="prev">◀</button>
    <button type="button" data-act="play">▶</button>
    <input type="range" id="{uid}-slider" min="0" value="0">
    <button type="button" data-act="next">▶</button>
    <span id="{uid}-steplabel" class="ptxviz-muted">Passo 1</span>
  </div>"""

    timeline_html = ""
    if show_timeline:
        timeline_html = f"""
  <div class="ptxviz-timeline">
    <div class="ptxviz-timeline-header">
      <h3>Linha do tempo (todas as threads)</h3>
      <label class="ptxviz-muted" style="display:flex;align-items:center;gap:6px;margin:0;">
        Agrupar por:
        <select id="{uid}-tl-group" style="width:auto;">
          <option value="thread">Thread</option>
          <option value="warp">Warp</option>
        </select>
      </label>
      <span id="{uid}-tl-info" class="ptxviz-muted"></span>
      <span class="ptxviz-muted">Clique numa célula/rótulo para selecionar essa thread ou warp.</span>
    </div>
    <div class="ptxviz-timeline-scroll">
      <table id="{uid}-tl-table" class="ptxviz-tl-table"></table>
    </div>
  </div>"""

    codeview_html = ""
    if show_code:
        codeview_html = f"""
  <div class="ptxviz-codeview">
    <div class="ptxviz-timeline-header">
      <h3>Código do kernel — linha executada agora</h3>
      <span class="ptxviz-muted">Granularidade por bloco do CFG (não por instrução individual). Mais de uma cor ao mesmo tempo = divergência.</span>
    </div>
    <div id="{uid}-codechips"></div>
    <div id="{uid}-codefallback" class="ptxviz-muted" hidden>Nenhum arquivo .cu associado a esta visualização — passe <code>source_path=kernel.source_path</code> ao gerar o HTML pra habilitar o código-fonte aqui.</div>
    <div id="{uid}-codebody" class="ptxviz-code"></div>
  </div>"""

    # Cytoscape/dagre só são embutidos (custam ~660KB) quando a seção do
    # grafo é pedida — "só timeline" ou "só código" não pagam esse peso.
    vendor_scripts_html = ""
    if show_graph:
        vendor_scripts_html = (
            f"<script>{_CYTOSCAPE_JS}</script>\n"
            f"<script>{_DAGRE_JS}</script>\n"
            f"<script>{_CYTOSCAPE_DAGRE_JS}</script>\n"
            f"<script>{_LIB_REGISTER_JS}</script>\n"
        )

    fragment = f"""
<div id="{uid}-root" class="ptxviz-root">
  <div class="ptxviz-header">
    <div class="ptxviz-title">{_esc(title)}</div>
    <div class="ptxviz-meta">{_esc(meta_html)}</div>
  </div>{stepbar_html}
  {body_html}{timeline_html}{codeview_html}
</div>
<script type="application/json" id="{uid}-data">{payload_json}</script>
<style>{css}</style>
{vendor_scripts_html}<script>{app_js}</script>
"""
    return fragment


def _esc(text: str) -> str:
    import html
    return html.escape(str(text))


# ──────────────────────────────────────────────────────────────────────────
# 3) API pública
# ──────────────────────────────────────────────────────────────────────────

def render_dynamic_trace_html(result: dict,
                              title: str = "Traço dinâmico do kernel",
                              warp_size: int = 32,
                              grid_dim: Optional[tuple] = None,
                              block_dim: Optional[tuple] = None,
                              height: str = "480px",
                              source_path: Optional[str] = None,
                              kernel_name: Optional[str] = None,
                              sections: Optional[Iterable[str]] = None) -> str:
    """Constrói o fragmento HTML/CSS/JS autocontido (sem `<html>`/`<body>`
    próprios) que renderiza o traço dinâmico de forma navegável.

    `result` é o dict devolvido por `PTXAnalyzer.dynamic_flow(..., mode="data")`
    (precisa pelo menos da chave `control_flow`; sem `dynamic_flow`, mostra
    só a estrutura estática — é assim que `PTXAnalyzer.control_flow_html()`
    reaproveita este mesmo renderizador) — nada aqui reinterpreta o traço,
    só reformata o que o analisador já calculou.

    `source_path`, se informado (ex.: `kernel.source_path`), embute o
    trecho do arquivo `.cu` correspondente ao kernel em estudo na seção
    "Código do kernel", que destaca a(s) linha(s) correspondente ao bloco
    em que cada thread/grupo está no passo atual — a granularidade é por
    bloco básico (não por instrução individual), que é o que o CFG já
    rastreia. `kernel_name`, se informado (ex.: `kernel.friendly_kernel_name`),
    recorta o arquivo pra mostrar só a função desse kernel — útil quando o
    `.cu` declara várias variantes no mesmo arquivo. Sem `kernel_name`, o
    arquivo inteiro é mostrado.

    `sections`, se informado, restringe quais blocos aparecem — um
    subconjunto de `dynamic_view.ALL_SECTIONS` = ("graph", "threads",
    "memory", "io", "timeline", "code"). Por padrão (`None`) mostra todos.
    Peças que dependem de dados que faltam (ex.: "memory" sem snapshots)
    simplesmente mostram um aviso vazio, não quebram.

    Pode ser passado direto para `IPython.display.HTML(...)` (Jupyter/Colab)
    ou embutido em uma página HTML maior.
    """
    if "control_flow" not in result:
        raise ValueError(
            "`result` precisa vir de PTXAnalyzer.control_flow(mode='data') ou "
            "PTXAnalyzer.dynamic_flow(..., mode='data') (falta a chave 'control_flow')."
        )
    meta = {}
    if grid_dim is not None:
        meta["grid_dim"] = list(grid_dim)
    if block_dim is not None:
        meta["block_dim"] = list(block_dim)
    payload = _build_payload(result, warp_size=warp_size, meta=meta,
                             source_path=source_path, kernel_name=kernel_name)
    return _render_fragment(payload, title=title, height=height, sections=sections)


def render_dynamic_trace_iframe_html(result: dict,
                                     title: str = "Traço dinâmico do kernel",
                                     warp_size: int = 32,
                                     grid_dim: Optional[tuple] = None,
                                     block_dim: Optional[tuple] = None,
                                     height: str = "480px",
                                     source_path: Optional[str] = None,
                                     kernel_name: Optional[str] = None,
                                     sections: Optional[Iterable[str]] = None) -> str:
    """Empacota a visualização dinâmica em um `<iframe srcdoc>` para
    frontends de notebook/webview que não executam de forma confiável
    múltiplos `<script>` inline irmãos no mesmo output HTML."""
    fragment = render_dynamic_trace_html(
        result,
        title=title,
        warp_size=warp_size,
        grid_dim=grid_dim,
        block_dim=block_dim,
        height=height,
        source_path=source_path,
        kernel_name=kernel_name,
        sections=sections,
    )
    frame_id = f"ptxviz-frame-{uuid.uuid4().hex}"
    iframe_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "</head><body style='margin:0;padding:0;background:#ffffff;'>"
        f"{fragment}"
        "</body></html>"
    )
    iframe_srcdoc = html.escape(iframe_doc, quote=True)
    return (
        "<div style='width:100%;background:transparent;'>"
        f"<iframe id='{frame_id}' sandbox='allow-scripts allow-same-origin' "
        f"style='width:100%;border:0;height:{_esc(height)};background:#ffffff;' "
        f"srcdoc=\"{iframe_srcdoc}\"></iframe>"
        "</div>"
    )


def show_dynamic_trace_colab(result: dict, **kwargs):
    """Chama `render_dynamic_trace_iframe_html(result, **kwargs)` e exibe o
    resultado via `IPython.display.HTML` — uso direto em uma célula do
    Colab/Jupyter:

        show_dynamic_trace_colab(result, title="bubble_sort_global_kernel")
    """
    fragment = render_dynamic_trace_iframe_html(result, **kwargs)
    from IPython.display import HTML, display
    display(HTML(fragment))
    return None


def save_dynamic_trace_html(result: dict, path: str, **kwargs) -> str:
    """Grava uma página HTML completa e independente (abre direto no
    navegador, sem precisar de Jupyter) com a visualização dinâmica."""
    fragment = render_dynamic_trace_html(result, **kwargs)
    title = kwargs.get("title", "Traço dinâmico do kernel")
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_esc(title)}</title></head>"
        f"<body style='margin:0;padding:20px;background:#eef2f7;font-family:system-ui,sans-serif;'>"
        f"{fragment}"
        "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
