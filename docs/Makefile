# Minimal Sphinx docs Makefile
SPHINXBUILD ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)"

.PHONY: html clean
