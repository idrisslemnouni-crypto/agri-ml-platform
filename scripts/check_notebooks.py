from pathlib import Path

import nbformat

for path in Path("notebooks").glob("*.ipynb"):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert all(not any(o.output_type == "error" for o in c.get("outputs", [])) for c in nb.cells)
print("Notebook schemas and stored outputs valid")
