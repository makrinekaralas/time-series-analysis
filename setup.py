import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsa",
    version="0.0.1",
    author="Makrine Karalashvili",
    author_email="maka.karalashvili@bmw.de",
    description="Univariate Time Series Analysis and Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/makrinekaralas/time-series-analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)