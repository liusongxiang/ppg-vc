PYTHON:= python3.7
CUDA_VERSION:= 10.1
PYTORCH_VERSION:= 1.6
DOT:= .
.PHONY: all clean

all: virtualenv

virtualenv:
	test -d venv || virtualenv -p $(PYTHON) venv
	. venv/bin/activate; pip install torch==$(PYTORCH_VERSION) \
		-f https://download.pytorch.org/whl/cu$(subst $(DOT),,$(CUDA_VERSION))/torch_stable.html
	. venv/bin/activate; cd ../; pip install -r requirements.txt
	touch venv/bin/activate

clean:
	rm -fr venv
	find -iname "*.pyc" -delete
