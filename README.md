# DeepField-app

Web application for visualization and exploration of reservoir models in 3D, 2D, and 1D.

Lightweight. Modern. Open source.

## Features

The application allows you to read reservoir models in ECLIPSE file format,
view and explore data in 3D, 2D and 1D, write and execute custom python scripts 
containing reservoir model transformations or calculations, 
and immediately view the results of these transformations.

Main page of the application in a dark theme:

<img src="static/scene0.PNG" width="50%"/>

An example of a reservoir model in 3D view with a filter applied to scalar values:

<img src="static/scene1.PNG" width="50%"/>

2D slice view of the reservoir model and colormap options:

<img src="static/scene2.PNG" width="50%"/>

Construction of a multiline 1D plot to compare dynamic properties of grid cells and wells:

<img src="static/scene3.PNG" width="50%"/>

Plots obtained from interpolation of PVT or relative permeability tables:

<img src="static/scene4.PNG" width="50%"/>

Description of the reservoir model and a white theme of the application:

<img src="static/scene5.PNG" width="50%"/>

Script writing

<img src="static/scene6.PNG" width="50%"/>

...and the results of its execution:

<img src="static/scene7.PNG" width="50%"/>


## Installation as a package

We recommend creating a new virtual environment with python 3.11 to install the project dependencies:

	conda create -n app python=3.11

Activate the new environment:

	conda activate app

To install the project dependencies, run in the terminal:

    pip install "git+https://github.com/deepfield-team/DeepField-app.git"

After installation, run in the terminal:

	deepfield-app

This should open a new tab in your default browser to http://localhost:8080/ with the application's home page.

You can add a few optional parameters to the application start command:
* --server - use to prevent a new tab from opening in the browser
* --app - use to launch the application in a separate window rather than in the browser
* --port 1234 - to change the default port 8080 to, e.g., 1234

When the application is running, you can click on the help icon in
the upper right corner to read a brief description of the page. 
Hover over buttons and icons to see a tooltip with textual information
about them.

## Installation from source code

Another option to run the application is to clone the entire repository:

	git clone https://github.com/deepfield-team/DeepField-app.git

Addionally, you will need to clone the repository `DeepField` into the same directory as the `DeepField-app`:

	git clone https://github.com/deepfield-team/DeepField.git

Install dependencties in both repositories using

	pip install -r requirements.txt

Then navigate to the directory DeepField-app and run in the terminal

	python -m deepfield_app.app

to start the application.


## Open-source reservoir models

An example reservoir model with dynamics simulation can be found in the `open_data` directory in the `DeepField` repository [https://github.com/deepfield-team/DeepField](https://github.com/deepfield-team/DeepField),
as well as links to a number of other open source models.

## Script writing

The application allows you to write and execute python scripts for
reservoir model transformations and calculations. The script should 
be based on the `DeepField` framework 
[https://github.com/deepfield-team/DeepField](https://github.com/deepfield-team/DeepField).
Read the [documentation](https://deepfield-team.github.io/DeepField/)
and see
[examples](https://github.com/deepfield-team/DeepField/blob/main/tutorials) 
in the `DeepField` repository to prepare a script.

## Next releases

The project is developing. We are preparing new releases with new features.
Your suggestions and issues reports will help to make the application even better.

## What's inside

We use
* [trame](https://github.com/Kitware/trame) to build the web application
* [DeepField](https://github.com/deepfield-team/DeepField) to read and process reservoir models

## Citing

We hope that this project will help you in your research and you will decide to cite it as
```
DeepField web application (2024). GitHub repository, https://github.com/deepfield-team/DeepField-app.
```
