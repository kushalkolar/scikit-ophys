from setuptools import setup, find_packages


install_requires = [
    "numpy",
    "numba"
]


setup(
    name='scikit-ophys',
    version='0.0',
    packages=find_packages(),
    install_requires=install_requires,
    url='',
    license='GPL v3',
    author='Kushal Kolar',
    author_email='',
    description=''
)
