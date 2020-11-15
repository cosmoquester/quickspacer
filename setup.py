from setuptools import find_packages, setup

setup(
    name="quickspacer",
    version="0.0.1",
    description="Korean spacing correction model that aims for fast speed and moderate accuracy.",
    python_requires=">=3.7",
    install_requires=[],
    url="https://github.com/psj8252/quickspacer.git",
    author="Park Sangjun",
    packages=find_packages(exclude=["tests"]),
)
