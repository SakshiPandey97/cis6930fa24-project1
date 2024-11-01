from setuptools import setup, find_packages

setup(
	name='project1',
	version='1.0',
	author='Sakshi Pandey',
	authour_email='sakshi.pandey@ufl.edu',
	packages=find_packages(exclude=('tests', 'docs')),
	setup_requires=['pytest-runner'],
	tests_require=['pytest']	
)