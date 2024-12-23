from setuptools import setup, find_packages

VERSION = "0.0.1"

# with open('docs/index.rst', 'r') as f:
#     LONG_DESCRIPTION = f.read()

setup(
    name='DeepField-app',
    packages=find_packages(),
    version=VERSION,
    url='https://github.com/deepfield-team/DeepField',
    license='Apache License 2.0',
    author='deepfield-team',
    author_email='',
    description='Deepfield visualization toolbox.',
    # long_description=LONG_DESCRIPTION,
    zip_safe=False,
    platforms='any',
    install_requires=[
        "DeepField @ git+https://github.com/deepfield-team/DeepField.git@d0c89164925d8450782125e73ae327fb600a3889",
        "trame",
        "trame-vuetify",
        "trame-vtk",
        "trame-components",
        "trame-matplotlib",
        "trame-plotly",
        "plotly"
    ],
    extras_require={
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Apache License 2.0',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering'
    ],
)
