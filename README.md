# OD-QC-001 — Ekpyrotic contraction and classicality

Computational study of the starting relationship

    rho_shear / rho_phi ∝ a^{3(w-1)}

for an ekpyrotic contracting universe, plus a comparison with a specified
effective bounce model and the WKB classicality of the contracting branch.

## Layout

- `src/ekpyrotic.py` — model, integrators, diagnostics (documented conventions)
- `notebooks/OD-QC-001_ekpyrotic_classicality.ipynb` — executed study notebook
- `notebooks/make_notebook.py` — generator for the notebook
- `results/OD-QC-001_REPORT.md` — findings, assumptions, open items (step 5)
- `results/metrics.json`, `results/figs/` — quantitative outputs
- `paper/OD-QC-001_paper.tex` / `paper/OD-QC-001_paper.pdf` — the paper write-up

Rebuild the paper: `cd paper && latexmk -pdf OD-QC-001_paper.tex` (TeX Live).

## Reproduce

```bash
python3 notebooks/make_notebook.py
cd notebooks && jupyter nbconvert --to notebook --execute --inplace \
    OD-QC-001_ekpyrotic_classicality.ipynb
```

Requirements: python3 with numpy, scipy, matplotlib, jupyter/nbconvert.
