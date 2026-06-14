# IEEE BSN 2026 Paper Materials

## Files

| File | Purpose |
|---|---|
| `ieee_bsn2026.tex` | Full 4-page IEEE conference paper (compile-ready) |
| `figures/fig1_system_architecture.png` | **Fig. 1** (used in LaTeX via `\includegraphics`) |
| `generate_fig1_system_architecture.py` | Regenerate Fig. 1 PNG |
| `fig1_system_architecture.tex` | Old TikZ version (optional, not used in paper) |
| `fig1_system_diagram.md` | Mermaid version for slides/thesis |
| `build_table_i.py` | Auto-build Table I from `results/` |
| `table_i_data.json` | Latest accuracy data (generated) |

## Compile the paper

Requires a LaTeX distribution (MiKTeX or TeX Live) with `IEEEtran` class.

Uses the official IEEE conference template preamble (`\documentclass[conference]{IEEEtran}`, `\IEEEoverridecommandlockouts`).

```bash
cd paper
pdflatex ieee_bsn2026.tex
pdflatex ieee_bsn2026.tex
```

Ensure `figures/fig1_system_architecture.png` exists (run `generate_fig1_system_architecture.py` if needed).

Output: `ieee_bsn2026.pdf`

## Regenerate Table I

```bash
python paper/build_table_i.py
```

Paste the printed LaTeX rows into `ieee_bsn2026.tex` if results change.

## Before submission — fill in placeholders

- [ ] Author names and affiliations
- [ ] IRB/ethics protocol number
- [ ] Participant ages and demographics
- [ ] CTNet and FBMSNet full citations (replace `[Add full ...]`)
- [ ] Acknowledgments / funding
- [ ] Retrain CTNet/FBMSNet for S1 if you want a complete 5×3 table
- [ ] Verify paper fits **4 pages** (IEEE BSN 2026 limit, references included)

## Note on page limit

IEEE BSN 2026 accepts **4-page** technical papers (not 5). This draft targets 4 pages in two-column IEEE format.
