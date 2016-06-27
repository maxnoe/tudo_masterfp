all: build/main.pdf

mplfiles = ../matplotlibrc ../header-matplotlib.tex

pyoptions = MATPLOTLIBRC=$(shell pwd)/../ TEXINPUTS=$(shell pwd)/..:build:


build/plots/hopg_up.pdf build/plots/hopg.pdf_down.pdf: data/hopg.nid scripts/hopg.py scripts/read_nid.py $(mplfiles) | build/plots
	$(pyoptions) python scripts/hopg.py

build/plots/gold.pdf: data/gold.nid scripts/gold.py scripts/read_nid.py $(mplfiles) | build/plots
	$(pyoptions) python scripts/gold.py


plots = $(addprefix build/plots/, $(addsuffix .pdf, \
	hopg_up \
	hopg_down \
	gold \
	))


build/main.pdf: FORCE $(plots) | build
	TEXINPUTS=build:build/plots:..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	main.tex

preview: FORCE $(plots) | build
	TEXINPUTS=build:build/plots:..: \
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
