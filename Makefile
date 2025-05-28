.PHONY: install test lint format clean docs

install:
	pip install -e .

test:
	pytest

lint:
	flake8 .
	mypy .

format:
	black .
	isort .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

docs:
	sphinx-build -b html docs/source docs/build/html

run:
	python main.py

backtest:
	python backtest.py

validate-config:
	python utils/config_validator.py

setup-env:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

update-deps:
	pip freeze > requirements.txt 