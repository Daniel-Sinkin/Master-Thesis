TEX      = pdflatex
BIB      = bibtex
MAIN     = main
BUILDDIR = build

.PHONY: all clean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex tex/*.tex chapters/*.tex references.bib
	$(TEX) -output-directory=$(BUILDDIR) $(MAIN).tex
	$(BIB) $(BUILDDIR)/$(MAIN)
	$(TEX) -output-directory=$(BUILDDIR) $(MAIN).tex
	$(TEX) -output-directory=$(BUILDDIR) $(MAIN).tex

clean:
	rm -rf $(BUILDDIR)
	rm -f $(MAIN).pdf

# Requires: pip install watchdog
watch:
	watchmedo shell-command \
		--patterns="*.tex;*.bib" \
		--recursive \
		--command='make all 2>&1 | tail -5' .
