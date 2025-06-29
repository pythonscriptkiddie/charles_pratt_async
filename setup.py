from setuptools import setup, find_packages

setup(
    name="pareto_ggg",
    version="0.1.0",
    description="Pareto/GGG customer lifetime model with slice sampling",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib"
    ],
    python_requires=">=3.7",
)
