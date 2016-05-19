all: build/main.pdf

mplfiles = ../matplotlibrc ../header-matplotlib.tex

pyoptions = MATPLOTLIBRC=$(shell pwd)/../ TEXINPUTS=$(shell pwd)/..: 


build/plots/hopg.pdf: data/hopg.nid scripts/hopg.py scripts/read_nid.py $(mplfiles) | build/plots
	$(pyoptions) python scripts/hopg.py


plots = $(addprefix build/plots/, $(addsuffix .pdf, \
	hopg \
	))


build/main.pdf: FORCE $(plots) | build
	TEXINPUTS=build:..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	main.tex

preview: FORCE $(plots) | build
	TEXINPUTS=build:..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	  -pvc \
	main.tex


build:
	mkdir -p build

build/plots:
	mkdir -p build/plots

clean:
	rm -rf build

FORCE:

.PHONY: all clean
