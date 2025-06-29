# build_pareto_ggg.py
import os

PACKAGE_NAME = "pareto_ggg"
FILES = [
    "__init__.py",
    "model.py",
    "slice_sampling.py",
    "utils.py",
    "mcmc.py"
]
TEST_DIR = os.path.join(PACKAGE_NAME, "tests")
TEST_FILES = ["__init__.py", "test_model.py"]

REQUIREMENTS = """numpy
scipy
pandas
matplotlib
"""

SETUP_PY = f"""from setuptools import setup, find_packages

setup(
    name=\"{PACKAGE_NAME}\",
    version=\"0.1.0\",
    description=\"Pareto/GGG customer lifetime model with slice sampling\",
    author=\"Your Name\",
    packages=find_packages(),
    install_requires=[
        \"numpy\",
        \"scipy\",
        \"pandas\",
        \"matplotlib\"
    ],
    python_requires=\">=3.7",
)
"""

README = f"""# {PACKAGE_NAME}

Pareto/GGG customer lifetime model with slice sampling.
Includes tools for simulation, parameter inference, and plotting.
"""

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_file(path, content=""):
    with open(path, "w") as f:
        f.write(content)

def main():
    # Create package directory and core files
    ensure_dir(PACKAGE_NAME)
    for fname in FILES:
        write_file(os.path.join(PACKAGE_NAME, fname))

    # Create test directory and test files
    ensure_dir(TEST_DIR)
    for fname in TEST_FILES:
        write_file(os.path.join(TEST_DIR, fname))

    # Create setup.py, requirements.txt, README.md
    write_file("setup.py", SETUP_PY)
    write_file("requirements.txt", REQUIREMENTS)
    write_file("README.md", README)

    print(f"Created {PACKAGE_NAME} package scaffold.")

if __name__ == "__main__":
    main()
