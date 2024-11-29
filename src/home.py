import os
import sys
from glob import glob
import numpy as np
import vtk

from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
)

from trame.widgets import html, vuetify3 as vuetify

sys.path.append('../deepfield-team/DeepField')
from deepfield import Field

from .config import state, ctrl, FIELD, renderer
from .common import reset_camera

state.field_attrs = []
state.wellnames = []
state.dir_list = []
state.dimens = [0, 0, 0]
state.max_timestep = 0
state.data1d = []
state.tables = []
state.recentFiles = []
state.i_cells = []
state.j_cells = []
state.k_cells = []
state.i_slice = [0, 0]
state.j_slice = [0, 0]
state.k_slice = [0, 0]


def filter_path(path):
    "True if path is a directory or has .data or .hdf5 extension."
    if os.path.isdir(path):
        return True
    _, ext = os.path.splitext(path)
    return ext.lower() in ['.data', '.hdf5']

@state.change("user_request")
def get_path_variants(user_request, **kwargs):
    _ = kwargs
    paths = list(glob(user_request + "*"))
    state.dir_list = [p for p in paths if filter_path(p)]

def load_file(*args, **kwargs):
    _ = args, kwargs
    field = Field(state.user_request).load()

    if state.user_request not in state.recentFiles:
        state.recentFiles = state.recentFiles + [state.user_request, ]

    FIELD['model'] = field

    dataset = field.get_vtk_dataset()
    FIELD['dataset'] = dataset

    mapper = vtkDataSetMapper()
    mapper.SetInputData(dataset)

    py_ds = dsa.WrapDataObject(dataset)
    c_data = py_ds.CellData
    FIELD['c_data'] = c_data

    state.field_attrs = [k for k in c_data.keys() if k not in ['I', 'J', 'K']]
    state.activeField = state.field_attrs[0]

    res = []
    for well in field.wells:
        if 'RESULTS' in well:
            res.extend([k for k in well.RESULTS.columns if k != 'DATE'])
    res = sorted(list(set(res)))
    FIELD['data1d']['wells'] = res

    state.dimens = [int(x) for x in field.grid.dimens]

    state.i_slice = [0, state.dimens[0]]
    state.j_slice = [0, state.dimens[1]]
    state.k_slice = [0, state.dimens[2]]

    state.i_cells = ['Average'] + list(range(1, state.dimens[0]+1))
    state.j_cells = ['Average'] + list(range(1, state.dimens[1]+1))
    state.k_cells = ['Average'] + list(range(1, state.dimens[2]+1))

    if 'states' in field.components:
        attrs = field.states.attributes
        if attrs:
            state.max_timestep = field.states[attrs[0]].shape[0] - 1
        FIELD['data1d']['states'] = attrs

    if 'tables' in field.components:
        attrs = field.tables.attributes
        if attrs:
            state.tables = [t for t in attrs if field.tables[t].domain]

    state.data1d = list(np.concatenate([v for _, v in FIELD['data1d'].items()]))

    actor = vtkActor()

    bbox = field.grid.bounding_box
    ds = abs(bbox[1] - bbox[0])
    ds_max = ds.max()
    scales = ds_max / ds
    actor.SetScale(*scales)

    vtk_array = dsa.numpyTovtkDataArray(c_data[state.activeField])

    dataset.GetCellData().SetScalars(vtk_array)

    mapper.SetScalarRange(dataset.GetScalarRange())
    actor.SetMapper(mapper)

    renderer.RemoveActor(FIELD['actor'])
    renderer.AddActor(actor)
    FIELD['actor'] = actor

    reset_camera()
    ctrl.view_update()

ctrl.load_file = load_file

def make_empty_dataset():
    dataset = vtk.vtkUnstructuredGrid()

    mapper = vtkDataSetMapper()
    mapper.SetInputData(dataset)
    mapper.SetScalarRange(0, 1)

    actor = vtkActor()
    actor.SetMapper(mapper)

    renderer.AddActor(actor)
    renderer.ResetCamera()

    FIELD['actor'] = actor
    FIELD['dataset'] = dataset

def on_keydown(key_code, alt_pressed):
    # if key_code == "Tab":
        # pass
    if key_code == "Enter":
        _, ext = os.path.splitext(state.user_request)
        if ext.lower() in ['.data', '.hdf5']:
            ctrl.load_file()
            return
        path = state.dir_list[0]
        if os.path.isdir(path):
            path += "\\"
        state.user_request = path
    if alt_pressed and key_code == "Digit1":
        state.activeTab = "1d"
    if alt_pressed and key_code == "Digit2":
        state.activeTab = "2d"
    if alt_pressed and key_code == "Digit3":
        state.activeTab = "3d"
    if alt_pressed and key_code == "KeyH":
        state.activeTab = "home"
    if alt_pressed and key_code == "KeyI":
        state.activeTab = "info"

def render_home():
    with html.Div(style='position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 80vw; height: 10vh'):
        with vuetify.VContainer():
            with vuetify.VRow():
                with vuetify.VCol():
                    with vuetify.VTextField(
                        ref="searchInput",
                        v_model=("user_request", ""),
                        label="Input reservoir model path",
                        clearable=True,
                        name="searchInput",
                        keydown=(on_keydown, "[$event.code, $event.altKey]"),
                        __events=["keydown"]):
                        with vuetify.Template(v_slot_append=True,
                            properties=[("v_slot_append", "v-slot:append")],):
                            vuetify.VBtn('Load', click=ctrl.load_file)
            with vuetify.VRow(classes="pa-0 ma-0"):
                with vuetify.VCol(classes="pa-0 ma-0"):
                    with vuetify.VCard(
                        classes="overflow-auto", max_width="40vw", max_height="30vh"):
                        with vuetify.VList():
                            with vuetify.VListItem(
                                v_for="item, index in dir_list",
                                click="user_request = item"):
                                vuetify.VListItemTitle("{{item}}")

