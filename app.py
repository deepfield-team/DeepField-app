from pathlib import Path

import plotly.express as px
import numpy as np

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

surface_filter = vtkGeometryFilter()
surface_filter.SetInputConnection(reader.GetOutputPort())
surface_filter.SetPassThroughPointIds(True)

mapper = vtkDataSetMapper()
mapper.SetInputConnection(surface_filter.GetOutputPort())
# mapper.SetScalarRange(0, 1)

py_ds = dsa.WrapDataObject(dataset)
c_data = py_ds.CellData

vtk_array = dsa.numpyTovtkDataArray(c_data['PERMX'])
dataset.GetCellData().SetScalars(vtk_array)
mapper.SetScalarRange(dataset.GetScalarRange())

actor = vtkActor()
actor.GetProperty().SetOpacity(0.5)
actor.SetMapper(mapper)

renderer.AddActor(actor)
renderer.ResetCamera()

VTK_VIEW_SETTINGS = {
    "interactive_ratio": 1,
    "interactive_quality": 80,
}

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

def render_home():
    with vuetify.VContainer(style="width: 100%;height: 80vh;display: flex;justify-content: center;align-items: center;"):
        vuetify.VFileInput(
            label='Input reservoir model file',
            style="width: 50%;",
            classes='rounded-xl elevation-12')

def render_info():
    with vuetify.VRow():
        with vuetify.VCard(
            max_width="344",
            style="margin: 10px",
            hover=True):
            vuetify.VImg(src=res_png, style="margin: 10px")
            vuetify.VCardTitle("This is home")
            vuetify.VBtn("Home", to="/", style="margin: 10px")
        with vuetify.VCard(max_width="344", style="margin: 10px"):
            vuetify.VImg(src="require('res.png')",
                         width="100")
            vuetify.VCardTitle("This is home 2")
            vuetify.VBtn("Home", to="/", style="margin: 10px")
        with vuetify.VCard(max_width="344", style="margin: 10px"):
            vuetify.VCardTitle("This is home 3")
            vuetify.VBtn("Home", to="/", style="margin: 10px", color='primary')

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

def render_2d():
    with vuetify.VCard():
        vuetify.VImg(src=res_png)
    with vuetify.VContainer(style='width:100vw; display: flex;position: fixed; bottom: 5vh;justify-content: center; align-items: center;'):
        with vuetify.VBottomNavigation(style='width:30vw;opacity: 0.75'):
            with vuetify.VBtn(icon=True):
                    vuetify.VIcon("mdi-magnify")
            with vuetify.VBtn(icon=True):
                vuetify.VIcon("mdi-lightbulb-multiple-outline")
            with vuetify.VBtn(icon=True):
                vuetify.VIcon("mdi-dots-vertical")

rates = {'x': np.arange(100),
         'PROD1': np.sin(np.arange(100)),
         'PROD2': np.cos(np.arange(100)),
         'PROD3': np.arange(100)}

def create_fig(well, width, height):
    fig = px.line(
        rates,
        x='x',
        y=well,
        width=width,
        height=height
    )
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
    with vuetify.VContainer(fluid=True, classes="fill-height"):
        with vuetify.VRow(style="height: 50%;"):
            with vuetify.VCol():
                with trame.SizeObserver("figure_size"):
                    ctrl.update_size = plotly.Figure().update

# ==========================================
ctrl.on_server_ready.add(ctrl.view_update)

with VAppLayout(server) as layout:
    with layout.root:
        with vuetify.VAppBar(app=True, clipped_left=True):
            vuetify.VAppBarNavIcon(click='drawer =! drawer')

            vuetify.VToolbarTitle("DeepField", style='overflow: visible; margin-right: 10px')
            with vuetify.VTabs(v_model=('activeTab', 'home')):
                vuetify.VTab('Home', value="home")
                vuetify.VTab('3d', value="3d")
                vuetify.VTab('2d', value="2d")
                vuetify.VTab('1d', value="1d")
                vuetify.VTab('Info', value="info")

            with vuetify.VBtn(icon=True):
                vuetify.VIcon("mdi-magnify")
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
            label="Opacity", 
            classes="mt-1",
            hide_details=False,
            dense=False
            )
        vuetify.VSelect(
            label='Select color',
            items=('color', ['Red', 'Green', 'Blue'])
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
