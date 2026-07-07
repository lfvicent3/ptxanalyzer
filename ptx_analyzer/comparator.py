"""
Comparação textual de múltiplos kernels PTX.
"""

from __future__ import annotations

from typing import Dict, List

from .analyzer import PTXAnalyzer, _bar, _section


class PTXComparator:
    def __init__(self):
        self._analyzers: Dict[str, PTXAnalyzer] = {}
        self._order: List[str] = []

    def add(self, name: str, analyzer: PTXAnalyzer) -> "PTXComparator":
        self._analyzers[name] = analyzer
        if name not in self._order:
            self._order.append(name)
        return self

    def _metrics(self) -> Dict[str, dict]:
        return {name: self._analyzers[name].kernel.metrics_dict() for name in self._order}

    def summary(self):
        if not self._order:
            print("Nenhum kernel adicionado.")
            return

        table = self._metrics()
        names = self._order
        metric_names = [
            "Instruções",
            "Regs/Thread",
            "BranchCondicional",
            "BranchRatio",
            "ld.global",
            "st.global",
            "Shared",
            "ld/st.local",
            "Int.Aritmética",
            "PTXASSpillStores",
            "PTXASSpillLoads",
        ]

        col_w = min(14, max(10, max(len(name) for name in names) + 1))
        metric_w = max(len(metric) for metric in metric_names) + 1

        print("╔" + "═" * 66 + "╗")
        print(f"║  {'Comparação PTX':<64}║")
        print("╚" + "═" * 66 + "╝")
        print()

        header = f"  {'Métrica':<{metric_w}}"
        for name in names:
            header += f"  {name:>{col_w}}"
        print(header)
        print("  " + "─" * (metric_w + (col_w + 2) * len(names)))

        for metric in metric_names:
            row = f"  {metric:<{metric_w}}"
            for name in names:
                value = table[name].get(metric, "—")
                if isinstance(value, float):
                    if metric == "BranchRatio":
                        value = f"{value:.1%}"
                    else:
                        value = f"{value:.3f}"
                row += f"  {str(value):>{col_w}}"
            print(row)

        print(f"\n{_section('Mix de Instruções', 68)}")
        all_categories = sorted({
            category
            for analyzer in self._analyzers.values()
            for category in analyzer.kernel.category_counts
        })
        for category in all_categories:
            row = f"  {category:<14}"
            for name in names:
                kernel = self._analyzers[name].kernel
                total = max(kernel.total_instructions, 1)
                count = kernel.category_counts.get(category, 0)
                pct = count / total * 100.0
                row += f"  {pct:4.1f}% {_bar(count, max(kernel.category_counts.values(), default=1), 10)}"
            print(row)
        print()

    def generate_report_table(self) -> str:
        if not self._order:
            return ""

        table = self._metrics()
        metric_names = [
            "Instruções",
            "Regs/Thread",
            "BranchCondicional",
            "BranchRatio",
            "ld.global",
            "st.global",
            "Shared",
            "ld/st.local",
            "PTXASSpillStores",
            "PTXASSpillLoads",
            "Int.Aritmética",
        ]

        lines = []
        header = ["Métrica", *self._order]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for metric in metric_names:
            row = [metric]
            for name in self._order:
                value = table[name].get(metric, "—")
                if isinstance(value, float) and metric == "BranchRatio":
                    value = f"{value:.1%}"
                row.append(str(value))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
