all: build/main.pdf

mplfiles = ../matplotlibrc ../header-matplotlib.tex

pyoptions = MATPLOTLIBRC=$(shell pwd)/../ TEXINPUTS=$(shell pwd)/..: 


build/attenuation.pdf: data/att_long_102_9kHz_fft.csv data/att_short_102_9kHz_fft.csv scripts/attenuation_factor.py $(mplfiles)
	$(pyoptions) python scripts/attenuation_factor.py


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
