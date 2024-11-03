from setuptools import setup, find_packages
from setuptools.command.install import install
import nltk

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

setup(
    name='project1_redactor',
    version='1.0',
    author='Sakshi Pandey',
    author_email='sakshi.pandey@ufl.edu',
    packages=find_packages(exclude=('tests', 'docs')),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'nltk',
        'spacy',
        'transformers',
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
    entry_points={
        'console_scripts': [
            'redactor=redactor:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='3.12',
)
