from setuptools import setup, find_packages

setup(
    name="dynalr",
    version="1.0.0",
    description="PID‑based adaptive learning‑rate optimizers for PyTorch",
    author="Hassan Al Subaidi",
    author_email="hassanalsubaidi1@gmail.com",
    url="https://github.com/Hassan1278/DynaLR",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
