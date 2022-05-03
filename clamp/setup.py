from setuptools import setup, find_packages


setup(
    name="clamp",
    version="0.1",
    packages=find_packages(),
    install_requires=['torch>=1.10.1',
                      'matplotlib',
                      'seaborn',
                      'numpy',
                      'ipykernel',
                      'pandas',
                      'pyyaml',
                      'tqdm']
)
