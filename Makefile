# Makefile
TEX      = lualatex
BIB      = bibtex
MAIN     = main
BUILDDIR = build

.PHONY: all clean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex tex/*.tex chapters/*.tex references.bib
	mkdir -p $(BUILDDIR)
	$(TEX) -output-directory=$(BUILDDIR) $(MAIN).tex
	$(BIB) $(BUILDDIR)/$(MAIN)
	$(TEX) -output-directory=$(BUILDDIR) $(MAIN).tex
	$(TEX) -output-directory=$(BUILDDIR) $(MAIN).tex

clean:
	rm -rf $(BUILDDIR)
	rm -f $(MAIN).pdf
	rm -f $(MAIN).aux $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).log \
		$(MAIN).bbl $(MAIN).blg $(MAIN).out $(MAIN).run.xml \
		$(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).lol \
		$(MAIN).synctex.gz

# Requires: pip install watchdog
watch:
	watchmedo shell-command \
		--patterns="*.tex;*.bib" \
		--recursive \
		--command='make all 2>&1 | tail -5' .
