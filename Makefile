.PHONY: test benchmark serve clean

test:
	python3 -m unittest discover -s tests -v

benchmark:
	python3 -m routecalib.cli benchmark --out results --n-total 1500 --seed 42

serve:
	python3 -m routecalib.cli serve --host 127.0.0.1 --port 8080

clean:
	rm -rf results __pycache__ routecalib/__pycache__ tests/__pycache__ .pytest_cache
