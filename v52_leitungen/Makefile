all: build/main.pdf

mplfiles = ../matplotlibrc ../header-matplotlib.tex

pyoptions = MATPLOTLIBRC=$(shell pwd)/../ TEXINPUTS=$(shell pwd)/..: 


build/attenuation_fft.pdf build/attenuation_signal.pdf: data/att_long_102_9kHz_fft.csv data/att_short_102_9kHz_fft.csv scripts/attenuation_factor.py $(mplfiles)
	$(pyoptions) python scripts/attenuation_factor.py

build/lcr.pdf: data/rg058-85m-lcrg.csv scripts/plot_lcr.py | build
	$(pyoptions) python scripts/plot_lcr.py

build/length_measurement.pdf: data/reflek1.csv scripts/plot_length_measurement.py | build
	$(pyoptions) python scripts/plot_length_measurement.py

build/unknown.pdf: scripts/plot_unknown.py | build
	$(pyoptions) python scripts/plot_unknown.py

build/multiple_reflection.pdf: scripts/plot_multiple_reflection.py | build
	$(pyoptions) python scripts/plot_multiple_reflection.py



plots = $(addprefix build/, $(addsuffix .pdf, \
	attenuation_signal \
	attenuation_fft \
	lcr \
	length_measurement \
	unknown \
	multiple_reflection \
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

clean:
	rm -rf build

FORCE:

.PHONY: all clean
