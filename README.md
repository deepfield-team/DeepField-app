# DeepField-app

Web application for visualization and exploration of reservoir models in 3D, 2D, and 1D.

Lightweight. Modern. Open source.

## Features

The application allows you to read reservoir models in ECLIPSE file format,
view and explore data in 3D, 2D and 1D, write and execute custom scripts 
containing reservoir model transformations or calculations, 
and immediately view the results of these transformations.

Example of the reservoir model in 3D view:

![img](static/scene1.png)

2D view:

Construnction of a multiline 1D plot to compare dynamic properties of grid cells and wells:



## Installation

Run in the terminal:

    pip install "git+https://github.com/deepfield-team/DeepField-app.git@setup

## Run the application

After installation, type and run in the terminal:

	deepfield-app

This should open a new tab in your default browser to http://localhost:8080/ with the application's home page.

You can add a few optional parameters to the command line:
* --server - use to prevent a new tab from opening in the browser
* --app - use to launch the application in a separate window rather than in the browser
* --port 1234 - to change the default port 8080 to, e.g., 1234