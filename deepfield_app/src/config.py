"App configs."
from enum import Enum
from types import SimpleNamespace
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

actor_names = SimpleNamespace(
    wells='wells_actor',
    well_labels='well_labels_actor',
    main='main_actor',
    faults='faults_actor',
    fault_labels='fault_labels_actor',
    fault_links='fault_links_actor'
)

dataset_names = SimpleNamespace(
    wells='wells_dataset'
)
FIELD = {"actor": None,
         "dataset": None,
         "c_data": None,
         "data1d": {'states': [], 'wells': [], 'tables': []},
         "model": None,
         "model_copy": None}
