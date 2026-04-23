"""Phase 1 gate: scaffold integrity."""
import importlib
from pathlib import Path


def test_all_required_directories_exist():
    required = [
        "infinite_dom",
        "infinite_dom/generator",
        "infinite_dom/generator/templates",
        "infinite_dom/browser",
        "infinite_dom/environment",
        "infinite_dom/oracle",
        "infinite_dom/server",
        "training",
        "tests",
        "scripts",
    ]
    for d in required:
        assert Path(d).is_dir(), f"Missing directory: {d}"


def test_required_root_files_exist():
    required = [
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "Dockerfile",
        ".dockerignore",
        "openenv.yaml",
        ".gitignore",
        ".env.example",
        "BUILD_LOG.md",
    ]
    for f in required:
        assert Path(f).is_file(), f"Missing file: {f}"


def test_all_init_py_files_exist():
    packages = [
        "infinite_dom",
        "infinite_dom/generator",
        "infinite_dom/browser",
        "infinite_dom/environment",
        "infinite_dom/oracle",
        "infinite_dom/server",
        "tests",
    ]
    for p in packages:
        assert Path(f"{p}/__init__.py").is_file(), f"Missing __init__.py: {p}"


def test_core_imports_available():
    for mod in ["fastapi", "pydantic", "jinja2", "playwright", "yaml"]:
        importlib.import_module(mod)
