"""
Suporte a profiling de runtime para kernels CUDA instrumentados com cudaEvent.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence


_RUNTIME_LINE_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z0-9_]+)\s+"
    r"N=(?P<n>\d+)\s+"
    r"(?P<ms>\d+(?:\.\d+)?)\s+ms"
    r"(?P<extra>.*?)"
    r"(?P<status>OK|ERRO|WARN)\s*$"
)


@dataclass
class RuntimeSample:
    label: str
    n: int
    milliseconds: float
    status: str
    extra: str = ""
    raw_line: str = ""


@dataclass
class RuntimeBenchmark:
    label: str
    samples: List[RuntimeSample] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.samples)

    @property
    def statuses(self) -> List[str]:
        return [s.status for s in self.samples]

    @property
    def min_ms(self) -> float:
        return min((s.milliseconds for s in self.samples), default=0.0)

    @property
    def max_ms(self) -> float:
        return max((s.milliseconds for s in self.samples), default=0.0)

    @property
    def mean_ms(self) -> float:
        vals = [s.milliseconds for s in self.samples]
        return statistics.fmean(vals) if vals else 0.0

    @property
    def median_ms(self) -> float:
        vals = [s.milliseconds for s in self.samples]
        return statistics.median(vals) if vals else 0.0

    @property
    def stdev_ms(self) -> float:
        vals = [s.milliseconds for s in self.samples]
        return statistics.stdev(vals) if len(vals) >= 2 else 0.0

    @property
    def ok_rate(self) -> float:
        if not self.samples:
            return 0.0
        oks = sum(1 for s in self.samples if s.status == "OK")
        return oks / len(self.samples)

    def by_size(self) -> Dict[int, dict]:
        grouped: Dict[int, List[RuntimeSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.n, []).append(sample)

        summary: Dict[int, dict] = {}
        for n, samples in sorted(grouped.items()):
            vals = [s.milliseconds for s in samples]
            ok_rate = sum(1 for s in samples if s.status == "OK") / len(samples)
            summary[n] = {
                "runs": len(samples),
                "min_ms": round(min(vals), 6),
                "max_ms": round(max(vals), 6),
                "mean_ms": round(statistics.fmean(vals), 6),
                "median_ms": round(statistics.median(vals), 6),
                "stdev_ms": round(statistics.stdev(vals), 6) if len(vals) >= 2 else 0.0,
                "ok_rate": round(ok_rate, 6),
                "samples": [asdict(s) for s in samples],
            }
        return summary

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "runs": self.runs,
            "min_ms": round(self.min_ms, 6),
            "max_ms": round(self.max_ms, 6),
            "mean_ms": round(self.mean_ms, 6),
            "median_ms": round(self.median_ms, 6),
            "stdev_ms": round(self.stdev_ms, 6),
            "ok_rate": round(self.ok_rate, 6),
            "by_size": self.by_size(),
            "samples": [asdict(s) for s in self.samples],
        }


@dataclass
class RuntimeProfile:
    source_path: str
    executable_path: str
    arch: str
    sizes: List[int]
    repeats: int
    benchmarks: Dict[str, RuntimeBenchmark]
    stdout_runs: List[str]
    stderr_runs: List[str]

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "executable_path": self.executable_path,
            "arch": self.arch,
            "sizes": self.sizes,
            "repeats": self.repeats,
            "benchmarks": {k: v.to_dict() for k, v in self.benchmarks.items()},
            "stdout_runs": self.stdout_runs,
            "stderr_runs": self.stderr_runs,
        }


def _default_executable_path(src_path: str) -> str:
    base = os.path.splitext(os.path.basename(src_path))[0]
    return os.path.join(tempfile.gettempdir(), f"{base}_ptx_analyzer.bin")


def compile_cuda_binary(
    src_path: str,
    out_path: Optional[str] = None,
    arch: str = "sm_75",
    extra_flags: Optional[Sequence[str]] = None,
) -> str:
    """
    Compila um arquivo .cu para executável nativo usando nvcc.
    """
    if out_path is None:
        out_path = _default_executable_path(src_path)

    cmd = ["nvcc", src_path, f"-arch={arch}", "-O3", "-o", out_path]
    if extra_flags:
        cmd.extend(extra_flags)

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Erro ao compilar {src_path}:\n{res.stderr}")
    return out_path


def parse_runtime_output(output: str) -> List[RuntimeSample]:
    """
    Extrai linhas de benchmark no formato:
      bubble_sort  N=1024  1.234 ms  OK
    """
    samples: List[RuntimeSample] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _RUNTIME_LINE_RE.match(line)
        if not match:
            continue
        extra = " ".join(match.group("extra").split())
        samples.append(RuntimeSample(
            label=match.group("label"),
            n=int(match.group("n")),
            milliseconds=float(match.group("ms")),
            status=match.group("status"),
            extra=extra,
            raw_line=raw_line,
        ))
    return samples


def profile_cuda_runtime(
    src_path: str,
    sizes: Sequence[int] = (1024,),
    repeats: int = 3,
    arch: str = "sm_75",
    executable_path: Optional[str] = None,
    extra_compile_flags: Optional[Sequence[str]] = None,
    extra_run_args: Optional[Sequence[str]] = None,
) -> RuntimeProfile:
    """
    Compila e executa um benchmark CUDA instrumentado para múltiplos tamanhos.
    """
    exe_path = compile_cuda_binary(src_path, executable_path, arch, extra_compile_flags)
    benchmarks: Dict[str, RuntimeBenchmark] = {}
    stdout_runs: List[str] = []
    stderr_runs: List[str] = []

    for n in sizes:
        for _ in range(repeats):
            cmd = [exe_path, str(n)]
            if extra_run_args:
                cmd.extend(extra_run_args)
            res = subprocess.run(cmd, capture_output=True, text=True)
            stdout_runs.append(res.stdout)
            stderr_runs.append(res.stderr)
            if res.returncode != 0:
                raise RuntimeError(
                    f"Benchmark falhou para N={n} em {src_path}.\n"
                    f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}"
                )

            parsed = parse_runtime_output(res.stdout)
            if not parsed:
                raise RuntimeError(
                    f"Não foi possível extrair métricas de runtime da saída:\n{res.stdout}"
                )

            for sample in parsed:
                benchmarks.setdefault(sample.label, RuntimeBenchmark(sample.label))
                benchmarks[sample.label].samples.append(sample)

    return RuntimeProfile(
        source_path=os.path.abspath(src_path),
        executable_path=os.path.abspath(exe_path),
        arch=arch,
        sizes=list(sizes),
        repeats=repeats,
        benchmarks=benchmarks,
        stdout_runs=stdout_runs,
        stderr_runs=stderr_runs,
    )
