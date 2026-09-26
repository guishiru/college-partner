"""Keep the dependency direction from drifting.

AGENTS.md says modules must not reach into each other, and that the statistical
engine must stay free of language and application concerns. That rule is easy
to write and easy to break three months later, so this test reads every import
in ``modules/`` and fails when one points somewhere it should not.

To allow a new dependency, add it to ``ALLOWED`` on purpose — and say why in
``docs/decisions/``.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

MODULES_ROOT = Path(__file__).resolve().parents[1] / "modules"

# package -> the ``modules.*`` prefixes it is allowed to import.
# A package may always import itself.
ALLOWED: dict[str, set[str]] = {
    # The namespace package itself. Must stay empty of imports.
    "modules": set(),
    # Runtime layout only. Depends on nothing.
    "modules.workspace": set(),
    # 大模型客户端。叶子模块，不依赖任何业务模块。
    "modules.llm": set(),
    # Infrastructure modules: leaves, so they can never create a cycle.
    "modules.users": {"modules.workspace"},
    "modules.conversations": set(),
    "modules.employees": set(),
    "modules.jobs": set(),
    "modules.files": set(),
    "modules.reports": set(),
    # The language layer must not reach the engine: it names methods and
    # extracts parameters, it never computes.
    # 语言层可以调大模型——它的职责就是把人话翻译成方法和参数。
    # 引擎依然不许：见下面 test_the_engine_never_imports_an_llm_or_the_language_layer。
    "modules.skills": {"modules.llm"},
    # The engine is deterministic and knows nothing about anything else.
    "modules.analysis.engine": set(),
    # The workflow is allowed to orchestrate the engine, the Skill layer and
    # the job context object.
    "modules.analysis": {"modules.skills", "modules.jobs"},
    # The web layer is the composition root and may import every module.
    "modules.web": {
        "modules.analysis",
        "modules.conversations",
        "modules.employees",
        "modules.files",
        "modules.jobs",
        "modules.llm",
        "modules.reports",
        "modules.skills",
        "modules.users",
        "modules.workspace",
    },
}


def _owning_package(path: Path) -> str:
    relative = path.relative_to(MODULES_ROOT.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    dotted = ".".join(parts)
    candidates = [key for key in ALLOWED if dotted == key or dotted.startswith(key + ".")]
    return max(candidates, key=len) if candidates else dotted


def _imported_modules(tree: ast.AST, package: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative imports stay inside the file's own package.
                continue
            if node.module:
                found.add(node.module)
    return {name for name in found if name == "modules" or name.startswith("modules.")}


class ModuleBoundaryTests(unittest.TestCase):
    def test_every_module_file_belongs_to_a_declared_package(self):
        unknown = [
            str(path.relative_to(MODULES_ROOT.parent))
            for path in sorted(MODULES_ROOT.rglob("*.py"))
            if _owning_package(path) not in ALLOWED
        ]
        self.assertEqual(
            unknown,
            [],
            "新增模块必须在 ALLOWED 中声明它允许依赖谁，并在 docs/decisions/ 记录原因。",
        )

    def test_imports_follow_the_declared_direction(self):
        violations: list[str] = []
        for path in sorted(MODULES_ROOT.rglob("*.py")):
            package = _owning_package(path)
            if package not in ALLOWED:
                continue
            permitted = ALLOWED[package] | {package}
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for imported in sorted(_imported_modules(tree, package)):
                if not any(
                    imported == allowed or imported.startswith(allowed + ".")
                    for allowed in permitted
                ):
                    violations.append(
                        f"{path.relative_to(MODULES_ROOT.parent)}：{package} 不允许依赖 {imported}"
                    )
        self.assertEqual(violations, [], "\n" + "\n".join(violations))

    def test_the_engine_never_imports_an_llm_or_the_language_layer(self):
        engine_root = MODULES_ROOT / "analysis" / "engine"
        forbidden = ("modules.skills", "openai", "anthropic", "langchain")
        offenders: list[str] = []
        for path in sorted(engine_root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if any(name == item or name.startswith(item + ".") for item in forbidden):
                        offenders.append(f"{path.name}: {name}")
        self.assertEqual(
            offenders,
            [],
            "统计引擎必须保持确定性：不得依赖语言层或任何大模型 SDK。",
        )


if __name__ == "__main__":
    unittest.main()
