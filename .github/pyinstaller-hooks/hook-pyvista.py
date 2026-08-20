from PyInstaller.utils.hooks import collect_all, is_module_or_submodule

_EXCLUDED_MODULES = (
    "pyvista.conftest",
    "pyvista.examples",
    "pyvista.ext",
    "pyvista.trame",
    "pyvista.utilities.sphinx_gallery",
)


def _is_runtime_module(name: str) -> bool:
    return not any(is_module_or_submodule(name, excluded) for excluded in _EXCLUDED_MODULES)


datas, binaries, hiddenimports = collect_all(
    "pyvista",
    filter_submodules=_is_runtime_module,
    exclude_datas=["examples/**"],
    on_error="warn once",
)
