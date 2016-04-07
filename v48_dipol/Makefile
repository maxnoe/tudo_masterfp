all: build/main.pdf

mplfiles = ../matplotlibrc ../header-matplotlib.tex

# hier Python-Skripte:
build/langevin.pdf: scripts/langevin.py $(mplfiles) | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python scripts/langevin.py

build/strom.pdf: scripts/strom.py $(mplfiles) | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python scripts/strom.py

build/method1.pdf: scripts/method1.py build/data_corrected.csv $(mplfiles) | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python $^

build/fit_parameters_1.tex build/activation_work_1.tex: build/method1.pdf

build/data_corrected.csv: scripts/correct_data.py | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python $^

build/data_correction_fit.pdf: build/data_corrected.csv

build/data_corrected.pdf: build/data_corrected.csv

build/rate.tex build/rate.pdf: scripts/rate.py | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python $^

build/method2.pdf: scripts/method2.py build/data_corrected.csv | build
	MATPLOTLIBRC=.. TEXINPUTS=$(shell pwd)/..: python $^

$(addprefix build/, activation_work_2.tex tau_0.tex tau.tex): build/method2.pdf

build/data_1.tex: scripts/tables.py
	python scripts/tables.py

build/data_2.tex build/data_3.tex: build/data_1.tex

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/main.pdf: build/langevin.pdf build/strom.pdf

build/main.pdf: build/method1.pdf build/method2.pdf

build/main.pdf: build/data_correction_fit.pdf build/data_corrected.pdf

build/main.pdf: build/rate.pdf build/rate.tex

build/main.pdf: $(addprefix build/data_, 1.tex 2.tex 3.tex)

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
