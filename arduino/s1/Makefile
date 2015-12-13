all: build/presi.pdf



TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build \


build/presi.pdf: FORCE beamerthemem.sty beamerinnerthememetropolis.sty beamerouterthememetropolis.sty presi.tex  | build
	max_print_line=1048576 latexmk $(TeXOptions) presi.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
