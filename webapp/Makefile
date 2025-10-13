SHELL := /bin/bash
APPNAME := webapp

REQS := sassc msgfmt
K := $(foreach r, ${REQS}, $(if $(shell command -v ${r} 2> /dev/null), '', $(error "${r} not installed")))

VERSION := $(shell cat VERSION | tr -d "[:space:]")

default: clean compile

clean:
	rm -Rf build/${APPNAME}*
	rm -f public/css/main-*.css
	cd public/css && sassc -t compact -m main.scss main-${VERSION}.css

compile:
	cd public/css && sassc -t compact -m main.scss main-${VERSION}.css