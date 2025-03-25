from setuptools import setup, find_packages

VERSION = "0.0.1"

setup(
    name='DeepField-app',
    packages=find_packages(),
    version=VERSION,
    url='https://github.com/deepfield-team/DeepField-app',
    license='GNU General Public License v3.0',
    author='deepfield-team',
    author_email='',
    description='DeepField web application.',
    zip_safe=False,
    platforms='any',
    install_requires=[
        "DeepField @ git+https://github.com/deepfield-team/DeepField.git@a37648d0cc6c9ef3536fca84c2b75a8564ab77c3",
        "trame",
        "trame-vuetify",
        "trame-vtk",
        "trame-components",
        "trame-matplotlib",
        "trame-plotly",
        "plotly"
    ],
    entry_points={
    'console_scripts': [
        'deepfield-app = deepfield_app.app:server_start',
    ],
    },
    extras_require={
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: GNU General Public License v3.0',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering'
    ],
)
