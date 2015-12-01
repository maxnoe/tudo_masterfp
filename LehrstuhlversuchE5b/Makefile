texoptions = -lualatex \
			 -output-directory=build \
			 -interaction=nonstopmode \
			 -halt-on-error


all: build/poster.pdf

build/poster.pdf: FORCE poster.tex beamerthemetudoposter.sty | build
	TEXINPUTS=images:build: latexmk $(texoptions) poster.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -rf build
