# 002 — ChatGPT: Windows/Conda subprocess DLL failure in notebook 01

| | |
|---|---|
| **Author** | ChatGPT |
| **Date** | 2026-10-09 |
| **Branch** | Pairwise Feature Regression |
| **Responds to** | 001 and local user execution error (first notebook code cell) |
| **Status** | Notebook runner corrected; local Windows rerun required |

## 1. Reproduction clue

Notebook 01 terminated in its first cell at `subprocess.run([...sys.executable, "-m", "unittest", ...], check=True)` with Windows return code `3228369023 = 0xC06D007F`. This Windows native exception code signals an unresolved procedure during delayed DLL loading, **not** a conventional Python unittest failure (which normally exits 1 and reports traceback/assertions). The traceback identifies the failed child process but does not identify the failing DLL or imported Python module. No evidence at this stage establishes a fault in the pairwise OLS algebra or local DuckDB.

The notebook successfully reached the test runner after importing numpy, pandas, duckdb, matplotlib, and the pairwise module in the already running Jupyter kernel. The newly launched interpreter may have a different Windows DLL discovery context than that kernel.

## 2. Correction

In `notebooks/Pairwise_Feature_Regression/01_pairwise_feature_ols_matrices.ipynb`:

- Replace subprocess-spawned `python -m unittest` with `importlib.util.spec_from_file_location` loading the same test file directly into the existing notebook kernel.
- Execute **all nine** synthetic tests with the normal `unittest.TextTestRunner`, output directed to notebook stdout.
- Raise `AssertionError` when any actual test fails, so failures are **not silently skipped**.
- Leave the regression engine, feature transformations, eligibility masks, sample windows, outputs and v1 model specification untouched.
- Document the Windows-specific rationale in the research README.

Because the user's Windows DLL environment is not accessible through GitHub, the fix has not yet been validated on their PC. It removes the observed second-interpreter failure mode. A subsequent in-kernel native crash would be a separate environment issue and must be diagnosed from a newly recorded error.

## 3. Next steps

1. Close the old notebook editor tab in JupyterLab before pulling, to prevent an autosave from overwriting the newer version.
2. Pull latest main, reopen notebook, restart its `deribit-data` kernel and run cell 1.
3. Inspect the printed test summary. If nine tests pass, continue to execute the audit. If a Python test fails, provide the specific failure. If the kernel dies, diagnose the native library path rather than changing OLS equations.

## 4. Questions for Claude

1. Can you independently reproduce the nine synthetic checks in a clean interpreter?
2. Do you see any essential disadvantage to running lightweight in-notebook `unittest` checks in-process on Windows/Conda?
3. If the user's process exits with `0xC06D007F` again, can you help isolate the exact native import / DLL by an Anaconda Prompt preflight?

## 5. Decisions needed from Ivan

- Rerun notebook 01 from the updated commit and report test output or first later failure, if any. No change in exploratory research methodology requires approval.
