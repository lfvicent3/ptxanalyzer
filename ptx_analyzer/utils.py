"""
Utilitários de compilação.
"""

from .analyzer import PTXAnalyzer
from .comparator import PTXComparator

# ──────────────────────────────────────────────────────────────────────────────
# 9. Utilitários de compilação (chamados no Colab via shell)
# ──────────────────────────────────────────────────────────────────────────────

def compile_to_ptx(src_path: str, out_path: str = None,
                   arch: str = "sm_75", extra_flags: str = "") -> str:
    """
    Compila um arquivo .cu para PTX usando nvcc.
    Retorna o caminho do arquivo .ptx gerado.
    Requer nvcc instalado (disponível no Colab com GPU runtime).
    """
    import subprocess, os

    if out_path is None:
        out_path = src_path.replace(".cu", ".ptx")

    cmd = f"nvcc -ptx {src_path} -arch={arch} -o {out_path} {extra_flags}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc falhou:\n{result.stderr}"
        )
    if result.stderr:
        print(f"[nvcc] {result.stderr.strip()}")

    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 11. Utilitários de lote
# ──────────────────────────────────────────────────────────────────────────────

def analyze_all_ptx(ptx_dir: str) -> PTXComparator:
    """
    Carrega todos os .ptx de um diretório e retorna um PTXComparator pronto.

    Uso:
        comp = analyze_all_ptx("./ptx")
        comp.show_table()
        comp.plot_radar()
    """
    import os

    comp = PTXComparator()
    for fname in sorted(os.listdir(ptx_dir)):
        if fname.endswith(".ptx"):
            path = os.path.join(ptx_dir, fname)
            name = fname[:-4]
            try:
                comp.add(name, PTXAnalyzer.from_file(path))
            except Exception as e:
                print(f"[aviso] {fname}: {e}")
    return comp


def compare_kernels_in_ptx_file(ptx_path: str) -> PTXComparator:
    """
    Carrega um único arquivo PTX e compara todos os kernels contidos nele.

    Exemplo:
        comp = compare_kernels_in_ptx_file("./ptx/baseline.ptx")
        comp.summary()
    """
    analyzer = PTXAnalyzer.from_file(ptx_path)
    return analyzer.compare_kernels_in_file()
