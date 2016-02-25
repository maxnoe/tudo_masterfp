all: build/main.pdf

mplfiles = ../matplotlibrc ../header-matplotlib.tex


# hier Python-Skripte:
# build/fit_linear.pdf: scripts/fit_linear.py data/messwerte_2.csv $(mplfiles) | build
# 	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python scripts/fit_linear.py


build/main.pdf: FORCE | build

plots = $(addprefix build/plots/, $(addsuffix .pdf, \
	bethe_air \
	pulse_with_amp \
))



build/plots/%.pdf: scripts/plot_%.py ../header-matplotlib.tex ../matplotlibrc | build/plots
	MATPLOTLIBRC=$(shell pwd)/../ TEXINPUTS=$(shell pwd)/..: python $<

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
