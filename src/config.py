from trame.app import get_server

from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor
)

from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleRubberBandPick,
    vtkInteractorStyleSwitch,
)

import vtkmodules.vtkRenderingOpenGL2  # noqa

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

renderer = vtkRenderer()
renderer.SetBackground(1, 1, 1)
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)

rw_interactor = vtkRenderWindowInteractor()
rw_interactor.SetRenderWindow(render_window)
rw_interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

FIELD = {"actor": None,
         "dataset": None,
         "c_data": None,
         "data1d": {'states': [], 'wells': [], 'tables': []},
         "model": None}

VTK_VIEW_SETTINGS = {
    "interactive_ratio": 1,
    "interactive_quality": 90,
}
