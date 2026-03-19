"""Auto-discovery for taskbench robot agents.

Mirrors the solver discovery pattern: importing any module under
``taskbench/agents/`` triggers its ``@register_agent()`` decorator.
"""

_discovered = False


def discover_agents():
    """Import all agent modules under ``taskbench.agents``.

    Idempotent — safe to call multiple times.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    import importlib
    import pkgutil

    import taskbench.agents as agents_pkg

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        agents_pkg.__path__, prefix="taskbench.agents."
    ):
        importlib.import_module(modname)
