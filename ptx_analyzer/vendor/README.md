Bibliotecas JS de terceiros, vendorizadas (embutidas) aqui para que
`ptx_analyzer.dynamic_view` gere HTML 100% autocontido — sem depender de
CDN em tempo de visualização. Isso importa porque alguns visualizadores
de notebook (ex.: o renderer de output HTML do VS Code) rodam a saída de
célula num webview isolado que bloqueia `<script src="https://...">`
inserido em tempo de execução, mesmo com internet disponível na máquina.
Com as libs embutidas no HTML gerado, isso deixa de ser um problema.

- `cytoscape.min.js` — Cytoscape.js 3.30.2 (MIT) — https://js.cytoscape.org/
- `dagre.min.js` — dagre 0.8.5 (MIT) — https://github.com/dagrejs/dagre
- `cytoscape-dagre.min.js` — cytoscape-dagre 2.5.0 (MIT) — https://github.com/cytoscape/cytoscape.js-dagre

Para atualizar as versões, baixe os `.min.js` correspondentes do jsDelivr
e substitua os arquivos aqui.
