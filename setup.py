"""
Setup script for AdaptFoundation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptfoundation",
    version="0.1.0",
    author="Robin Guiavarch",
    author_email="robin.guiavarch@telecom-paris.fr",
    description="Adaptation of Foundation Models for Cortical Folding Analysis 1st lab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.telecom-paris.fr/telecom-neurospin/adaptfoundation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "adaptfoundation": ["configs/*.yaml"],
    },
)