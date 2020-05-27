from setuptools import find_packages, setup

setup(
    name="nlp_sentiment_analysis",
    packages=find_packages(),
    version="0.1.0",
    description="nlp sample",
    author="cse",
    license="MIT",
    entry_points={
        "console_scripts": ["generate_notebook=generate_notebook:generate_notebook"],
    },
)
