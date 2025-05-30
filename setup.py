from setuptools import setup, find_packages

setup(
    name="trading-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ccxt",
        "pandas",
        "numpy",
        "scikit-learn",
        "optuna",
        "pyyaml",
    ],
    python_requires=">=3.8",
) 