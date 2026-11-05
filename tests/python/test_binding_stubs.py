"""The native module's type stubs match what it exports.

Every public name in ``rustystats._rustystats`` has a declaration in
``_rustystats.pyi``. Every function's parameters have the runtime's names, order,
optionality and literal defaults. Every class declares each public member the runtime
class has, and nothing it lacks.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
import rustystats._rustystats as native

STUB = Path(native.__file__).with_name("_rustystats.pyi")


def _stub() -> ast.Module:
    return ast.parse(STUB.read_text(encoding="utf-8"))


def _declared() -> dict[str, ast.AST]:
    return {
        node.name: node for node in _stub().body if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }


EXPORTED = sorted(name for name in dir(native) if not name.startswith("_"))
FUNCTIONS = [name for name in EXPORTED if not inspect.isclass(getattr(native, name))]
CLASSES = [name for name in EXPORTED if inspect.isclass(getattr(native, name))]


def test_every_export_is_declared_and_every_declaration_exists() -> None:
    declared = _declared()
    assert sorted(set(EXPORTED) - set(declared)) == [], "exported but not in the stub"
    assert sorted(set(declared) - set(EXPORTED)) == [], "in the stub but not exported"


_NO_DEFAULT = object()


def _stub_parameters(node: ast.FunctionDef) -> list[tuple[str, Any]]:
    arguments = node.args
    positional = arguments.posonlyargs + arguments.args
    defaults: list[Any] = [_NO_DEFAULT] * (len(positional) - len(arguments.defaults))
    defaults += [ast.literal_eval(default) for default in arguments.defaults]
    parameters = [(arg.arg, default) for arg, default in zip(positional, defaults)]
    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        parameters.append((arg.arg, _NO_DEFAULT if default is None else ast.literal_eval(default)))
    return parameters


def _runtime_parameters(function: Any) -> list[tuple[str, Any]]:
    return [
        (name, _NO_DEFAULT if parameter.default is inspect.Parameter.empty else parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
    ]


@pytest.mark.parametrize("name", FUNCTIONS)
def test_function_signature_matches_the_stub(name: str) -> None:
    node = _declared()[name]
    assert isinstance(node, ast.FunctionDef), f"{name} is declared as a class"
    stub = _stub_parameters(node)
    runtime = _runtime_parameters(getattr(native, name))
    # A default the runtime cannot spell as a literal (``Some(0)``) shows as ``...``:
    # the stub must give one, whose value only the Rust signature states.
    stub = [
        (param, ...) if (param, ...) in runtime and default is not _NO_DEFAULT else (param, default)
        for param, default in stub
    ]
    assert stub == runtime


@pytest.mark.parametrize("name", CLASSES)
def test_class_members_match_the_stub(name: str) -> None:
    node = _declared()[name]
    assert isinstance(node, ast.ClassDef), f"{name} is declared as a function"
    declared = {item.name for item in node.body if isinstance(item, ast.FunctionDef)} | {
        item.target.id
        for item in node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }
    runtime = {member for member in dir(getattr(native, name)) if not member.startswith("_")}
    assert sorted(runtime - declared) == [], "members missing from the stub"
    assert sorted(declared - runtime - {"__init__"}) == [], "stub members the class lacks"
