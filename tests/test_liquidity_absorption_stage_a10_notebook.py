"""Ensure published A10 notebook cells parse before expensive local archive replay."""
from pathlib import Path
import json
import unittest

class TestNotebookSyntax(unittest.TestCase):
    def test_all_python_code_cells_compile(self):
        repo = Path(__file__).resolve().parents[1]
        nb = json.loads((repo / "notebooks" / "Liquidity_Absorption" / "10_unselected_level_universe.ipynb").read_text(encoding="utf8"))
        self.assertEqual(nb["nbformat"], 4)
        codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(codes), 4)
        for i,c in enumerate(codes):
            self.assertFalse(c.get("outputs"), f"Code cell {i} must not ship old outputs")
            compile("".join(c["source"]), f"notebook cell {i}", "exec")

if __name__ == "__main__":
    unittest.main()
