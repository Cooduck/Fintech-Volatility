Course research report in the NeurIPS 2026 template
==================================================

The compiled report is report.pdf. The editable manuscript is main.tex.
The report uses the instructor-provided style file and its preprint option.

From this directory, build with a TeX installation that includes Latin Modern:

  latexmk -pdf -halt-on-error -jobname=report main.tex

You can also compile with pdfLaTeX and BibTeX:

  pdflatex -halt-on-error -jobname=report main.tex
  bibtex report
  pdflatex -halt-on-error -jobname=report main.tex
  pdflatex -halt-on-error -jobname=report main.tex

For Overleaf, upload main.tex, references.bib, checklist.tex, neurips_2026.sty,
and the three PDFs under figures/. Set main.tex as the main document.

From the repository root, regenerate the figure PDFs with the project Python
environment. The script reads the archived CSV files and makes no API calls.

  python paper/figures/build_figures.py
