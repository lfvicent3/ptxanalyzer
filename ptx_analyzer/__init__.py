"""
PTX Analyzer
"""

from .analyzer import PTXAnalyzer
from .comparator import PTXComparator
from .parser import parse_ptx
from .heuristics import run_heuristics, LEVEL_ICONS, LEVEL_COLORS, LEVEL_TEXT_COLORS
from .source_view import PTXSourceView
from .utils import compile_to_ptx, analyze_all_ptx, compare_kernels_in_ptx_file
from .core import (
    PTXKernel,
    PTXInstruction,
    BasicBlock,
    CFGEdge,
    BranchSite,
    MemoryHotspot,
    ControlFlowAnalysis,
    CATEGORIES,
    CATEGORY_COLORS,
    build_cfg,
    analyze_control_flow,
)
from .visuals import (
    plot_bra_graph,
    plot_decision_tree,
    plot_gpu_efficiency,
    plot_instruction_roofline,
    plot_metric_space_pca,
    plot_branch_efficiency_registers,
    plot_memory_hierarchy,
    plot_runtime_curves,
)
from .runtime import (
    RuntimeSample,
    RuntimeBenchmark,
    RuntimeProfile,
    compile_cuda_binary,
    parse_runtime_output,
    profile_cuda_runtime,
)

__all__ = [
    "PTXAnalyzer",
    "PTXComparator",
    "parse_ptx",
    "run_heuristics",
    "LEVEL_ICONS",
    "LEVEL_COLORS",
    "LEVEL_TEXT_COLORS",
    "PTXSourceView",
    "compile_to_ptx",
    "analyze_all_ptx",
    "compare_kernels_in_ptx_file",
    "PTXKernel",
    "PTXInstruction",
    "BasicBlock",
    "CFGEdge",
    "BranchSite",
    "MemoryHotspot",
    "ControlFlowAnalysis",
    "build_cfg",
    "analyze_control_flow",
    "plot_decision_tree",
    "plot_bra_graph",
    "plot_gpu_efficiency",
    "plot_instruction_roofline",
    "plot_metric_space_pca",
    "plot_branch_efficiency_registers",
    "plot_memory_hierarchy",
    "plot_runtime_curves",
    "CATEGORIES",
    "CATEGORY_COLORS",
    "RuntimeSample",
    "RuntimeBenchmark",
    "RuntimeProfile",
    "compile_cuda_binary",
    "parse_runtime_output",
    "profile_cuda_runtime",
]
