from setuptools import setup

setup(
    name="diff-dalle",
    py_modules=["diff_dalle"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
