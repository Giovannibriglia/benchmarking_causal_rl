import pathlib

from setuptools import find_packages, setup


def get_version():
    """Gets the vmas version."""
    path = CWD / "benchmarking_causal_rl" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


CWD = pathlib.Path(__file__).absolute().parent


setup(
    name="BenchmarkingCausalRL",
    version="0.1.0",
    description="Benchmarking Causal RL",
    long_description=open("README.md").read(),
    url="https://github.com/Giovannibriglia/benchmarking_causal_rl",
    license="GPLv3",
    author="Giovanni Briglia",
    author_email="giovanni.briglia@phd.unipi.it",
    packages=find_packages(),
    install_requires=["torch"],
    include_package_data=True,
)
