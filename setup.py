from setuptools import setup, find_packages

setup(
    name='project1_redactor',
    version='1.0',
    author='Sakshi Pandey',
    author_email='sakshi.pandey@ufl.edu',
    packages=find_packages(exclude=('tests', 'docs')),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    python_requires='>=3.10',
)
