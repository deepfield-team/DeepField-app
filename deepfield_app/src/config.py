"App configs."
from enum import Enum
from trame.app import get_server

from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderer
)
import vtkmodules.vtkRenderingOpenGL2  # noqa


server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

state.trame__title = "DeepField"

renderer = vtkRenderer()
renderer.SetBackground(1, 1, 1)

class ActorNames(Enum):
    WELLS = 'wells_actor'
    WELL_LABELS = 'well_labels_actor'

class DatasetNames(Enum):
    WELLS = 'wells_dataset'

FIELD = {"actor": None,
         "dataset": None,
         "c_data": None,
         "data1d": {'states': [], 'wells': [], 'tables': []},
         "model": None,
         "model_copy": None}
