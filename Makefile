PYTHON ?= ../../../.venv/bin/python
TEX = pdflatex
BIB = bibtex
SRC = paper
TEXFLAGS = -interaction=nonstopmode -halt-on-error

all: $(SRC).pdf

figures:
	$(PYTHON) scripts/98_result_plots.py
	$(PYTHON) scripts/98_gqa_level2_plot.py
	$(PYTHON) scripts/98_threshold_evidence_plots.py

verify:
	$(PYTHON) scripts/98_verify_threshold_bundle.py

$(SRC).pdf: $(SRC).tex references.bib
	$(TEX) $(TEXFLAGS) $(SRC)
	$(BIB) $(SRC)
	$(TEX) $(TEXFLAGS) $(SRC)
	$(TEX) $(TEXFLAGS) $(SRC)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(SRC).pdf

.PHONY: all clean distclean figures verify
