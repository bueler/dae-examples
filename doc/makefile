all: twoballs.pdf

twoballs.pdf: twoballs.tex twoballs.bib
	pdflatex twoballs
	bibtex twoballs
	pdflatex twoballs
	pdflatex twoballs

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
