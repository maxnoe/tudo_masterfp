all: build/main.pdf

# hier Python-Skripte:
plots/langevin.pdf: langevin.py ../matplotlibrc ../header-matplotlib.tex | build
	TEXINPUTS=$(shell pwd)/..: python langevin.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: plots/langevin.pdf

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

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
