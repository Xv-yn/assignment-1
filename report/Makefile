# Build the report.  `make` produces report.pdf; `make clean` removes the
# LaTeX build products but never the PDF.
#
# latexmk handles the bibtex/rerun cycle automatically.  If latexmk is not
# installed, `make manual` does the same thing the long way.

PDF   = report.pdf
TEX   = report.tex
BIB   = references.bib

all: $(PDF)

$(PDF): $(TEX) $(BIB)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

manual:
	pdflatex -interaction=nonstopmode $(TEX)
	bibtex report
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

# Check the page limit (the brief allows 20 pages maximum, 10-15 ideal).
pages: $(PDF)
	@pdfinfo $(PDF) | grep Pages

clean:
	latexmk -c 2>/dev/null || true
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all manual pages clean
