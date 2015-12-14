all: build/main.pdf

# hier Python-Skripte:
build/langevin.pdf: plots/langevin.py ../matplotlibrc ../header-matplotlib.tex | build/plots
	TEXINPUTS=$(shell pwd)/..: python plots/langevin.py

build/strom.pdf: plots/strom.py ../matplotlibrc ../header-matplotlib.tex | build/plots
	TEXINPUTS=$(shell pwd)/..: python plots/strom.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/langevin.pdf build/strom.pdf

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
