from pathlib import Path

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib
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

# -----------------------------------------------------------------------------
# Get a server to work with
# -----------------------------------------------------------------------------

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

BASE = Path(__file__).parent
local_file_manager = LocalFileManager(__file__)
res_png = local_file_manager.url("res", BASE / "static" / "res.png")

dataset_file = HttpFile("./data/norne.vtu", "", __file__)

reader = vtkXMLUnstructuredGridReader()
reader.SetFileName(dataset_file.path)
reader.Update()
dataset = reader.GetOutput()

renderer = vtkRenderer()
renderer.SetBackground(1, 1, 1)
render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)

rw_interactor = vtkRenderWindowInteractor()
rw_interactor.SetRenderWindow(render_window)
rw_interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()


mapper = vtkDataSetMapper()
mapper.SetInputConnection(reader.GetOutputPort())
# mapper.SetScalarRange(0, 1)

py_ds = dsa.WrapDataObject(dataset)
c_data = py_ds.CellData

FIELDS = list(c_data.keys())

actor = vtkActor()
actor.SetScale(1, 1, 10)

mapper.SetScalarRange(0, 1)
actor.SetMapper(mapper)


@state.change("opacity")
def update_opacity(opacity, **kwargs):
    _ = kwargs
    if opacity is None:
        return
    actor.GetProperty().SetOpacity(opacity)
    ctrl.view_update()

@state.change("activeField")
def update_field(activeField, **kwargs):
    _ = kwargs
    if activeField is None:
        return
    vtk_array = dsa.numpyTovtkDataArray(c_data[activeField])
    dataset.GetCellData().SetScalars(vtk_array)
    mapper.SetScalarRange(dataset.GetScalarRange())
    actor.SetMapper(mapper)
    ctrl.view_update()

@state.change("colormap")
def update_cmap(colormap, **kwargs):
    cmap = get_cmap(colormap)
    table = actor.GetMapper().GetLookupTable()
    colors = cmap(np.arange(0,cmap.N)) 
    table.SetNumberOfTableValues(len(colors))
    for i, val in enumerate(colors):
        table.SetTableValue(i, val[0], val[1], val[2])
    table.Build()
    ctrl.view_update()

renderer.AddActor(actor)

camera = renderer.GetActiveCamera()
x, y, z = camera.GetPosition()
fx, fy, fz = camera.GetFocalPoint()
dist = np.linalg.norm(np.array([x, y, z]) - np.array([fx, fy, fz]))
camera.SetPosition(fx, fy-dist, fz)
camera.SetViewUp(0, 0, -1)
renderer.ResetCamera()

VTK_VIEW_SETTINGS = {
    "interactive_ratio": 1,
    "interactive_quality": 60,
}

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

def render_home():
    with vuetify.VContainer(style="width: 100%;height: 80vh;display: flex;justify-content: center;align-items: center;"):
        vuetify.VFileInput(
            v_model=("file_input", False),
            label='Input reservoir model file',
            style="width: 50%;",
            directory=True,
            classes='rounded-xl elevation-12')

@state.change("file_input")
def load_file(file_input, **kwargs):
    _ = kwargs
    if not file_input:
        return

def render_info():
    with vuetify.VCard(style="margin: 10px"):
        vuetify.VCardTitle("Description of the reservoir model")
        vuetify.VCardText('Dimensions: {}'.format(cube_shape))
        
def render_3d():
    with vuetify.VContainer(fluid=True, classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(dense=True, style="height: 100%;"):
            with vuetify.VCol(
                classes="pa-0",
                style="border-right: 1px solid #ccc; position: relative;",
            ):
                view = vtk_widgets.VtkRemoteView(
                    render_window,
                    **VTK_VIEW_SETTINGS,
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

cube = np.load('data/rock.npz')
cube_shape = cube['poro'].shape

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

def matplotlib_to_plotly(cmap, pl_entries, rdigits=2):
    scale = np.linspace(0, 1, pl_entries)
    colors = (cmap(scale)[:, :3]*255).astype(np.uint8)
    pl_colorscale = [[round(s, rdigits), f'rgb{tuple(color)}'] for s, color in zip(scale, colors)]
    return pl_colorscale

def create_slice(arr, width, height, colormap):
    fig = px.imshow(arr, aspect="auto", color_continuous_scale=colormap.lower())
    fig.update_layout(height=height,
                      width=width,
                      showlegend=False,
                      margin={'t': 30, 'r': 30, 'l': 30, 'b': 0},)
    return fig

@state.change("figure_xsize", "activeField", "xslice", "colormap")
def update_xslice(figure_xsize, activeField, xslice, colormap, **kwargs):
    _ = kwargs
    figure_size = figure_xsize
    if figure_size is None:
        return
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("weight", 300)
    arr = cube[activeField.lower()][xslice-1].T
    ctrl.update_xslice(create_slice(arr, width, height, colormap))

@state.change("figure_ysize", "activeField", "yslice", "colormap")
def update_yslice(figure_ysize, activeField, yslice, colormap, **kwargs):
    _ = kwargs
    figure_size = figure_ysize
    if figure_size is None:
        return
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 300)
    arr = cube[activeField.lower()][:, yslice-1].T
    ctrl.update_yslice(create_slice(arr, width, height, colormap))

@state.change("figure_zsize", "activeField", "zslice", "colormap")
def update_zslice(figure_zsize, activeField, zslice, colormap, **kwargs):
    _ = kwargs
    figure_size = figure_zsize
    if figure_size is None:
        return
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 300)
    arr = cube[activeField.lower()][:, :, zslice-1]
    ctrl.update_zslice(create_slice(arr, width, height, colormap))

def render_2d():
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="width:90%; height: 80%; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0'):
                vuetify.VSlider(
                    min=1,
                    max=cube_shape[0],
                    step=1,
                    v_model=('xslice', cube_shape[0]//2),
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
                    max=cube_shape[1],
                    step=1,
                    v_model=('yslice', cube_shape[1]//2),
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
                    max=cube_shape[2],
                    step=1,
                    v_model=('zslice', cube_shape[2]//2),
                    label="z", 
                    classes="mt-5 mr-5 ml-5",
                    hide_details=False,
                    dense=False
                    )
                with trame.SizeObserver("figure_zsize"):
                    ctrl.update_zslice = plotly.Figure(**CHART_STYLE).update

rates = pd.read_csv('data/data_egg.csv')

def create_fig(well, width, height):
    df = rates.loc[rates.cat == well]
    fig = make_subplots(rows=3,
                        cols=1,
                        subplot_titles=("OIL", "WATER", "BHP"),
                        vertical_spacing = 0.15)

    fig.append_trace(go.Scatter(
        x=df.date,
        y=df.oil,
        line=dict(color='black', width=2)
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=df.date,
        y=df.water,
        line=dict(color='royalblue', width=2)
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=df.date,
        y=df.bhp,
        line=dict(color='green', width=2)
    ), row=3, col=1)

    fig.update_layout(height=height,
                      width=width,
                      showlegend=False,
                      margin={'t': 30, 'r': 80, 'l': 100, 'b': 0},)
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
        v_model=("well", 'PROD1'),
        items=("wellnames", ['PROD1', 'PROD2', 'PROD3'])
    )
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="width:90%; height: 80%; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0'):
                with trame.SizeObserver("figure_size"):
                    ctrl.update_size = plotly.Figure(**CHART_STYLE).update

# ==========================================
ctrl.on_server_ready.add(ctrl.view_update)

with VAppLayout(server) as layout:
    with layout.root:
        # with vuetify.VThemeProvider(theme=state.theme, with_background=True):
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
                with html.Div(v_if="activeTab === 'home'"):
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
                    classes="mt-1",
                    hide_details=False,
                    dense=False
                    )
                vuetify.VSelect(
                    label="Colormap",
                    v_model=("colormap", 'jet'),
                    items=("colormaps",
                        ["autumn", "bone", "cool", "gray", "jet", "hot", "hsv", "ocean",
                         "seismic", "Spectral", "spring", "summer", "terrain",
                         "twilight", "viridis", "winter"],
                    ),
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )
                vuetify.VSelect(
                    v_model=('activeField', FIELDS[0]),
                    label='Select field',
                    items=('fields', FIELDS)
                    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
