all: build/main.pdf

mplfiles = ../matplotlibrc ../header-matplotlib.tex

# hier Python-Skripte:
# build/fit_linear.pdf: scripts/fit_linear.py data/messwerte_2.csv $(mplfiles) | build
# 	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python scripts/fit_linear.py


build/main.pdf: FORCE | build
	TEXINPUTS=build:..: \
	BIBINPUTS=..: \
	max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	main.tex

preview: FORCE | build
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

clean:
	rm -rf build

FORCE:

.PHONY: all clean
