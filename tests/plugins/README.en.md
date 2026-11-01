# tests/plugins

Tests of the plugin registry, the conformance check and the command `python -m mst.plugin`. Plugins for the checks are declared in the test (`fakes.py`) or created in a temporary directory.

| File | What it checks |
|---|---|
| `fakes.py` | A model, a demo case and a plugin declaration for conformance checks |
| `test_discovery.py` | Discovery of the packages `mst.plugins.<id>`, checks of declarations together, a duplicate id, a foreign metric namespace, an incompatible SDK range, the model catalog from the registry |
| `test_conformance.py` | `check_plugin` on declarations and a demo calculation: an error in the demo, a non-deterministic result, a position that should calculate but did not |
| `test_boundary.py` | The core does not know concrete asset types: `mst.engine` does not import plugin packages or name their identifiers |
| `test_cli.py` | `new` creates a package that passes `check`, `demo-run` prints quantiles, a broken package is rejected naming the problem |
