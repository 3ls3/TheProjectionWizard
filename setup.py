"""
Setup file for the Projection Wizard application.
"""

from setuptools import setup, find_packages

setup(
    name="projection-wizard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "great-expectations",
    ],
    python_requires=">=3.8",
)
