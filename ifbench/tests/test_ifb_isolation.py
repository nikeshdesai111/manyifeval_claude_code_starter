"""The IFBench work must not touch the ManyIFEval experiment."""
import ast
import hashlib
import subprocess
from pathlib import Path

IFB = Path(__file__).resolve().parents[1]
ROOT = IFB.parent
# ManyIFEval source-tree hash (src, configs, scripts, tests) at the start of the IFBench work, 2026-11-22.
MANYIFEVAL_TREE_SHA256 = "02c41f6d15e3682b9900125e0b44737e13629d1047664eec11451ebf60092d2f"


def test_manyifeval_source_tree_unchanged():
    h = hashlib.sha256()
    for d in ("src", "configs", "scripts", "tests"):
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",):
                h.update(str(p.relative_to(ROOT)).encode())
                h.update(p.read_bytes())
    assert h.hexdigest() == MANYIFEVAL_TREE_SHA256


def test_no_changes_outside_ifbench():
    out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", ".", ":(exclude)ifbench"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return  # not a git checkout; the tree hash above still guards
    assert out.stdout.strip() == "", out.stdout


def test_ifbench_code_never_imports_manyifeval():
    for p in (IFB / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n == "mife" or n.startswith("mife.") for n in names), p


def test_ifbench_seed_family_is_new():
    import yaml
    ours = yaml.safe_load((IFB / "configs" / "ifbench.yaml").read_text())["seeds"]
    theirs = (ROOT / "configs" / "experiment.yaml").read_text()
    for v in ours.values():
        assert str(v) not in theirs
