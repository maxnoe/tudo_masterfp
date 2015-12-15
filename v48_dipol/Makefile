all: build/main.pdf

mplfiles = ../matplotlibrc ../header-matplotlib.tex

# hier Python-Skripte:
build/langevin.pdf: scripts/langevin.py $(mplfiles) | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python scripts/langevin.py

build/strom.pdf: scripts/strom.py $(mplfiles) | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python scripts/strom.py

build/fit_linear.pdf: scripts/fit_linear.py data/messwerte_2.csv $(mplfiles) | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python scripts/fit_linear.py

build/fit_non_linear.pdf: scripts/fit_non_linear.py data/messwerte_2.csv $(mplfiles) | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python scripts/fit_non_linear.py

build/activation_work.pdf build/fit_parameters.tex: build/fit_linear.pdf

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/langevin.pdf build/strom.pdf build/fit_linear.pdf

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
