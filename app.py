from pathlib import Path
import os
import sys
from glob import glob

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib
import vtk
from matplotlib.pyplot import get_cmap

from trame.widgets import html, plotly, vtk as vtk_widgets, trame, vuetify3 as vuetify
from trame.app import get_server
from trame.assets.remote import HttpFile
from trame.ui.vuetify3 import VAppLayout
from trame.assets.local import LocalFileManager

# VTK imports
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

sys.path.append('../deepfield-team/DeepField')
from deepfield import Field


server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

BASE = Path(__file__).parent
local_file_manager = LocalFileManager(__file__)
res_png = local_file_manager.url("res", BASE / "static" / "res.png")

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
         "model": None,
         'dimens': [1, 1, 1]}

state.field_attrs = []
state.wellnames = []
state.dir_list = []
state.dimens = [1, 1, 1]

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

def reset_camera():
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    x, y, z = camera.GetPosition()
    fx, fy, fz = camera.GetFocalPoint()
    dist = np.linalg.norm(np.array([x, y, z]) - np.array([fx, fy, fz]))
    camera.SetPosition(fx-dist/np.sqrt(3), fy-dist/np.sqrt(3), fz-dist/np.sqrt(3))
    camera.SetViewUp(0, 0, -1)
    renderer.ResetCamera()

make_empty_dataset()
reset_camera()

VTK_VIEW_SETTINGS = {
    "interactive_ratio": 1,
    "interactive_quality": 90,
}

@state.change("opacity")
def update_opacity(opacity, **kwargs):
    _ = kwargs
    if opacity is None:
        return
    FIELD['actor'].GetProperty().SetOpacity(opacity)
    ctrl.view_update()

@state.change("activeField")
def update_field(activeField, **kwargs):
    _ = kwargs
    if activeField is None:
        return
    if FIELD['c_data'] is None:
        return
    vtk_array = dsa.numpyTovtkDataArray(FIELD['c_data'][activeField])
    dataset = FIELD['dataset']
    dataset.GetCellData().SetScalars(vtk_array)
    mapper = FIELD['actor'].GetMapper()
    mapper.SetScalarRange(dataset.GetScalarRange())
    FIELD['actor'].SetMapper(mapper)
    ctrl.view_update()

@state.change("colormap")
def update_cmap(colormap, **kwargs):
    cmap = get_cmap(colormap)
    table = FIELD['actor'].GetMapper().GetLookupTable()
    colors = cmap(np.arange(0, cmap.N)) 
    table.SetNumberOfTableValues(len(colors))
    for i, val in enumerate(colors):
        table.SetTableValue(i, val[0], val[1], val[2])
    table.Build()
    ctrl.view_update()


@state.change("user_request")
def get_path_variants(user_request, **kwargs):
    state.dir_list = list(glob(user_request + "*"))

state.load_completed = False

def load_file(*args, **kwargs):
    state.load_completed = False
    field = Field(state.user_request).load()
    FIELD['model'] = field
    state.load_completed = True

    dataset = field.get_vtk_dataset()
    FIELD['dataset'] = dataset

    mapper = vtkDataSetMapper()
    mapper.SetInputData(dataset)

    py_ds = dsa.WrapDataObject(dataset)
    c_data = py_ds.CellData
    FIELD['c_data'] = c_data

    state.field_attrs = list(c_data.keys())
    state.wellnames = [node.name for node in field.wells]
    state.dimens = [int(x) for x in field.grid.dimens]

    actor = vtkActor()

    bbox = field.grid.bounding_box
    ds = abs(bbox[1] - bbox[0])
    ds_max = ds.max()
    scales = ds_max / ds
    actor.SetScale(*scales)
    
    vtk_array = dsa.numpyTovtkDataArray(c_data[list(c_data.keys())[0]])
    dataset.GetCellData().SetScalars(vtk_array)
    
    mapper.SetScalarRange(dataset.GetScalarRange())
    actor.SetMapper(mapper)

    renderer.RemoveActor(FIELD['actor'])
    renderer.AddActor(actor)
    FIELD['actor'] = actor

    reset_camera()
    ctrl.view_update()

ctrl.load_file = load_file

def render_home():
    with html.Div(style='position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 80vw; height: 10vh'):
        with vuetify.VContainer():
            with vuetify.VRow():
                with vuetify.VCol():
                    vuetify.VTextField(
                                v_model=("user_request", ""),
                                label="Input reservoir model path",
                                clearable=True,
                                name="searchInput"
                            )
                with vuetify.VCol(cols=1):
                    vuetify.VBtn('Load', click=ctrl.load_file)
            with vuetify.VRow(classes="pa-0 ma-0"):
                with vuetify.VCol(classes="pa-0 ma-0"):
                    with vuetify.VCard(classes="overflow-auto", max_width="40vw", max_height="30vh"):
                        with vuetify.VList():
                            with vuetify.VListItem(
                                v_for='item, index in dir_list',
                                click="user_request = item"
                                ):
                                vuetify.VListItemTitle('{{item}}')

def render_info():
    with vuetify.VCard(style="margin: 10px"):
        vuetify.VCardTitle("Description of the reservoir model")
        vuetify.VCardText('Dimensions: ' + '{{dimens}}')
        
def render_3d():
    with vuetify.VContainer(fluid=True, classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(dense=True, style="height: 100%;"):
            with vuetify.VCol(
                classes="pa-0",
                style="border-right: 1px solid #ccc; position: relative;"
                ):
                view = vtk_widgets.VtkRemoteView(
                    render_window,
                    **VTK_VIEW_SETTINGS
                    )
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    with vuetify.VBottomNavigation(grow=True, style='left: 50%; transform: translateX(-50%); width: 40vw; bottom: 5vh; opacity: 0.75'):
        with vuetify.VBtn(icon=True):
            vuetify.VIcon("mdi-magnify")
        with vuetify.VBtn(icon=True):
            vuetify.VIcon("mdi-crop")
        with vuetify.VBtn(icon=True):
            vuetify.VIcon("mdi-chart-bar")
        with vuetify.VBtn(icon=True):
            vuetify.VIcon("mdi-magic-staff")
        with vuetify.VBtn(icon=True):
            vuetify.VIcon("mdi-play")

CHART_STYLE = {
    # "display_mode_bar": ("true",),
    "mode_bar_buttons_to_remove": (
        "chart_buttons",
        [
            "resetScale2d",
            "zoomIn2d",
            "zoomOut2d",
            "toggleSpikelines",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
        ],
    ),
    "display_logo": ("false",),
}

def create_slice(arr, width, height, colormap):
    fig = px.imshow(arr, aspect="auto", color_continuous_scale=colormap.lower())
    fig.update_layout(height=height,
                      width=width,
                      showlegend=False,
                      margin={'t': 30, 'r': 30, 'l': 30, 'b': 0},)
    return fig

def get_attr_from_field(attr):
    comp, attr = attr.split('_')
    return FIELD['model']._components[comp.lower()][attr]

@state.change("figure_xsize", "activeField", "xslice", "colormap")
def update_xslice(figure_xsize, activeField, xslice, colormap, **kwargs):
    _ = kwargs
    figure_size = figure_xsize
    if figure_size is None:
        return
    if activeField is None:
        return
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("weight", 300)
    arr = get_attr_from_field(activeField)[xslice-1].T
    ctrl.update_xslice(create_slice(arr, width, height, colormap))

@state.change("figure_ysize", "activeField", "yslice", "colormap")
def update_yslice(figure_ysize, activeField, yslice, colormap, **kwargs):
    _ = kwargs
    figure_size = figure_ysize
    if figure_size is None:
        return
    if activeField is None:
        return
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 300)
    arr = get_attr_from_field(activeField)[:, yslice-1].T
    ctrl.update_yslice(create_slice(arr, width, height, colormap))

@state.change("figure_zsize", "activeField", "zslice", "colormap")
def update_zslice(figure_zsize, activeField, zslice, colormap, **kwargs):
    _ = kwargs
    figure_size = figure_zsize
    if figure_size is None:
        return
    if activeField is None:
        return
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 300)
    arr = get_attr_from_field(activeField)[:, :, zslice-1]
    ctrl.update_zslice(create_slice(arr, width, height, colormap))

def render_2d():
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="width:90%; height: 80%; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0'):
                vuetify.VSlider(
                    min=1,
                    max=("dimens[0]",),
                    step=1,
                    v_model=('xslice', 1),
                    label="x", 
                    classes="mt-5 mr-5 ml-5",
                    hide_details=False,
                    dense=False
                    )
                with trame.SizeObserver("figure_xsize"):
                    ctrl.update_xslice = plotly.Figure(**CHART_STYLE).update
            with vuetify.VCol(classes='pa-0'):
                vuetify.VSlider(
                    min=1,
                    max=("dimens[1]",),
                    step=1,
                    v_model=('yslice', 1),
                    label="y", 
                    classes="mt-5 mr-5 ml-5",
                    hide_details=False,
                    dense=False
                    )
                with trame.SizeObserver("figure_ysize"):
                    ctrl.update_yslice = plotly.Figure(**CHART_STYLE).update
            with vuetify.VCol(classes='pa-0'):
                vuetify.VSlider(
                    min=1,
                    max=("dimens[2]",),
                    step=1,
                    v_model=('zslice', 1),
                    label="z", 
                    classes="mt-5 mr-5 ml-5",
                    hide_details=False,
                    dense=False
                    )
                with trame.SizeObserver("figure_zsize"):
                    ctrl.update_zslice = plotly.Figure(**CHART_STYLE).update

def create_fig(well, width, height):
    fig = make_subplots(rows=3,
                        cols=1,
                        subplot_titles=("OIL", "WATER", "BHP"),
                        vertical_spacing = 0.15)
    fig.update_layout(height=height,
                      width=width,
                      showlegend=False,
                      margin={'t': 30, 'r': 80, 'l': 100, 'b': 0},)

    if well not in state.wellnames:
        return fig

    df = FIELD['model'].wells[well].RESULTS

    if 'WOPR' not in df:
        df['WOPR'] = None
    fig.append_trace(go.Scatter(
        x=df.DATE.values,
        y=df.WOPR,
        line=dict(color='black', width=2)
    ), row=1, col=1)

    if 'WWPR' not in df:
        df['WWPR'] = None
    fig.append_trace(go.Scatter(
        x=df.DATE.values,
        y=df.WWPR,
        line=dict(color='royalblue', width=2)
    ), row=2, col=1)

    if 'BHP' not in df:
        df['BHP'] = None
    fig.append_trace(go.Scatter(
        x=df.DATE.values,
        y=df.BHP,
        line=dict(color='green', width=2)
    ), row=3, col=1)

    return fig

@state.change("figure_size", "well")
def update_size(figure_size, well, **kwargs):
    _ = kwargs
    if figure_size is None:
        return
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 100)
    ctrl.update_size(create_fig(well, width, height))

def render_1d():
    vuetify.VSelect(
        v_model=("well", state.wellnames[0] if state.wellnames else None),
        items=("wellnames",)
    )
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="width:90%; height: 80%; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0'):
                with trame.SizeObserver("figure_size"):
                    ctrl.update_size = plotly.Figure(**CHART_STYLE).update


ctrl.on_server_ready.add(ctrl.view_update)

with VAppLayout(server) as layout:
    with layout.root:
        with vuetify.VAppBar(app=True, clipped_left=True):
            vuetify.VAppBarNavIcon(click='drawer =! drawer')

            vuetify.VToolbarTitle("DeepField")
            with vuetify.VTabs(v_model=('activeTab', 'home'), style='left: 50%; transform: translateX(-50%);'):
                vuetify.VTab('Home', value="home")
                vuetify.VTab('3d', value="3d")
                vuetify.VTab('2d', value="2d")
                vuetify.VTab('1d', value="1d")
                vuetify.VTab('Info', value="info")

            with vuetify.VBtn(icon=True):
                vuetify.VIcon("mdi-settings")
            with vuetify.VBtn(icon=True):
                vuetify.VIcon("mdi-lightbulb-multiple-outline")
            with vuetify.VBtn(icon=True):
                vuetify.VIcon("mdi-dots-vertical")

        with vuetify.VMain():
            with html.Div(v_if="activeTab === 'home'", classes="fill-height"):
                render_home()
            with html.Div(v_if="activeTab === '3d'", classes="fill-height"):
                render_3d()
            with html.Div(v_if="activeTab === '2d'", classes="fill-height"):
                render_2d()
            with html.Div(v_if="activeTab === '1d'", classes="fill-height"):
                render_1d()
            with html.Div(v_if="activeTab === 'info'"):
                render_info()

        with vuetify.VNavigationDrawer(
            app=True,
            clipped=True,
            stateless=True,
            v_model=("drawer", False),
            width=200):
            vuetify.VSlider(
                min=0,
                max=1,
                step=0.1,
                v_model=('opacity', 1),
                label="Opacity", 
                classes="mt-8 mr-3",
                hide_details=False,
                dense=False,
                thumb_label=True
                )
            vuetify.VSelect(
                label="Colormap",
                v_model=("colormap", 'jet'),
                items=("colormaps",
                    ["gray", "jet", "hsv", "Spectral", "twilight", "viridis"],
                ),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
            vuetify.VSelect(
                v_model=('activeField', state.field_attrs[0] if state.field_attrs else None),
                label='Select field',
                items=('field_attrs', )
                )

if __name__ == "__main__":
    server.start()
