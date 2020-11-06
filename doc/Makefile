PAPER    := glow.pdf
TEX_SUFS := .aux .log .nav .out .snm .toc .vrb .fdb_latexmk .bbl .blg .fls
GITID    := $(shell git describe --always 2> /dev/null)
TEXSRC   := $(wildcard *.tex bolsig/*.tex)
FIGURES  := $(wildcard figures/*.pdf figures/*.jpg)
BIBS     := $(wildcard bibs/*.bib)

%.pdf:%.tex $(TEXSRC) $(BIBS) $(FIGURES) gitinfo2.pm
	echo "hello"
ifdef GITID
	./gitinfo2.pm
endif
	latexmk -pdf $<

all: $(PAPER)

clean:
	$(RM) $(foreach suf, ${TEX_SUFS}, $(PAPER:.pdf=${suf}))
ifdef GITID
	$(RM) vc.tex
endif
	$(RM) $(PAPER)

