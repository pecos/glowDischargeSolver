PAPER    := glow.pdf
TEX_SUFS := .aux .log .nav .out .snm .toc .vrb .fdb_latexmk .bbl .blg .fls
GITID    := $(shell git describe --always 2> /dev/null)
TEXSRC   := $(wildcard *.tex)
FIGURES  := $(wildcard figures/*.pdf figures/*.jpg)
BIBS     := $(wildcard bibs/*.bib)

#$(info $$GITID is [${GITID}])
$(info $$FIGURES is [${FIGURES}])
$(info $$PAPER is [${PAPER}])
$(info $$TEXSRC is [${TEXSRC}])
$(info $$FIGURES is [${FIGURES}])
$(info $$BIBS is [${BIBS}])


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

