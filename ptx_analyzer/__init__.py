"""
PTX Analyzer
"""

from .analyzer import PTXAnalyzer
from .comparator import PTXComparator
from .parser import parse_ptx
from .ptxas import parse_ptxas_output
from .utils import compile_to_ptx, analyze_all_ptx, compare_kernels_in_ptx_file
from .core import (
    PTXKernel,
    PTXInstruction,
    KernelParamDecl,
    BasicBlock,
    CFGEdge,
    BranchSite,
    MemoryHotspot,
    ControlFlowAnalysis,
    PTXASInfo,
    CATEGORIES,
    CATEGORY_COLORS,
    build_cfg,
    analyze_control_flow,
    explain_instruction,
)
from .runtime import (
    RuntimeSample,
    RuntimeBenchmark,
    RuntimeProfile,
    BenchmarkEntry,
    BenchmarkSuite,
    compile_cuda_binary,
    parse_runtime_output,
    profile_cuda_runtime,
    parse_benchmark_output,
    load_benchmark_csv,
)
from .state import (
    KernelArg,
    KernelLaunchConfig,
    ByteMemory,
    ThreadContext,
    Frame,
    BranchDecision,
    ThreadTrace,
    KernelDynamicTrace,
    UnsupportedInstruction,
)
from .interpreter import PTXInterpreter, SUPPORTED_OPCODES
from .dynamic_view import (
    render_dynamic_trace_html,
    show_dynamic_trace_colab,
    save_dynamic_trace_html,
    ALL_SECTIONS,
)

__all__ = [
    "PTXAnalyzer",
    "PTXComparator",
    "parse_ptx",
    "parse_ptxas_output",
    "compile_to_ptx",
    "analyze_all_ptx",
    "compare_kernels_in_ptx_file",
    "PTXKernel",
    "PTXInstruction",
    "KernelParamDecl",
    "BasicBlock",
    "CFGEdge",
    "BranchSite",
    "MemoryHotspot",
    "ControlFlowAnalysis",
    "PTXASInfo",
    "build_cfg",
    "analyze_control_flow",
    "explain_instruction",
    "CATEGORIES",
    "CATEGORY_COLORS",
    "RuntimeSample",
    "RuntimeBenchmark",
    "RuntimeProfile",
    "BenchmarkEntry",
    "BenchmarkSuite",
    "compile_cuda_binary",
    "parse_runtime_output",
    "profile_cuda_runtime",
    "parse_benchmark_output",
    "load_benchmark_csv",
    "KernelArg",
    "KernelLaunchConfig",
    "ByteMemory",
    "ThreadContext",
    "Frame",
    "BranchDecision",
    "ThreadTrace",
    "KernelDynamicTrace",
    "UnsupportedInstruction",
    "PTXInterpreter",
    "SUPPORTED_OPCODES",
    "render_dynamic_trace_html",
    "show_dynamic_trace_colab",
    "save_dynamic_trace_html",
    "ALL_SECTIONS",
]
