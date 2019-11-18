# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python

all: clean inplace test

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-build:
	rm -rf build

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-cache

pytest:
	rm -f .coverage
	pytest mnefun

flake:
	flake8 --count mnefun examples setup.py

pydocstyle:
	pydocstyle mnefun

test: clean pytest flake pydocstyle
