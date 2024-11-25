from pathlib import Path
import os
import sys
from glob import glob

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vtk
from matplotlib.pyplot import get_cmap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from trame.widgets import html, plotly, vtk as vtk_widgets, trame, vuetify3 as vuetify, matplotlib
from trame.app import get_server
from trame.assets.remote import HttpFile
from trame.ui.vuetify3 import VAppLayout

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
import deepfield

def get_figure_size(f_size):
    if f_size is None:
        return {}

    dpi = f_size.get("dpi")
    rect = f_size.get("size")
    w_inch = rect.get("width") / dpi
    h_inch = rect.get("height") / dpi

    return {
        "figsize": (w_inch, h_inch),
        "dpi": dpi,
    }

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

PLOTS = {"plot1d": None}

state.field_attrs = []
state.wellnames = []
state.dir_list = []
state.dimens = [0, 0, 0]
state.max_timestep = 0
state.need_time_slider = False
state.cumulativeRates = False
state.data1d = []
state.i_cells = []
state.j_cells = []
state.k_cells = []


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

@state.change("activeField", "activeStep")
def update_field(activeField, activeStep, **kwargs):
    _ = kwargs
    if activeField is None:
        return
    if FIELD['c_data'] is None:
        return
    if activeField.split("_")[0].lower() == 'states':
        state.need_time_slider = True
        vtk_array = dsa.numpyTovtkDataArray(FIELD['c_data'][activeField][:, activeStep])
    else:
        state.need_time_slider = False
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

    state.field_attrs = [k for k in c_data.keys() if k not in ['I', 'J', 'K']]
    res = []
    for well in field.wells:
        if 'RESULTS' in well:
            res.extend([k for k in well.RESULTS.columns if k != 'DATE'])
    res = sorted(list(set(res)))
    FIELD['data1d']['wells'] = res

    state.dimens = [int(x) for x in field.grid.dimens]
    state.i_cells = ['Average'] + list(range(1, state.dimens[0]+1))
    state.j_cells = ['Average'] + list(range(1, state.dimens[1]+1))
    state.k_cells = ['Average'] + list(range(1, state.dimens[2]+1))

    if 'states' in field.components:
        attrs = field.states.attributes
        if attrs:
            state.max_timestep = field.states[attrs[0]].shape[0] - 1
        FIELD['data1d']['states'] = attrs

    state.data1d = list(np.concatenate([v for _, v in FIELD['data1d'].items()]))

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
                vuetify.VSlider(
                    v_if='need_time_slider',
                    min=0,
                    max=("max_timestep",),
                    step=1,
                    v_model=('activeStep', 0),
                    label="Timestep",
                    classes="mt-5 mr-5 ml-5",
                    hide_details=False,
                    dense=False
                    )
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

def get_data_limits(component, attr, activeStep):
    data = getattr(component, attr)
    if isinstance(component, deepfield.field.States):
        data=data[activeStep]
    data = data[FIELD['model'].grid.actnum]
    vmax = data.max()
    vmin = data.min()
    if vmax == vmin:
        vmax = 1.01 * vmax
        vmin = 0.99 * vmin
    return vmin, vmax

def create_slice(component, att, i, j, k, t, colormap, figure_size):
    plt.close("all")
    fig, ax = plt.subplots(**figure_size)
    vmin, vmax = get_data_limits(component, att, t)
    component.show_slice(attr=att, i=i, j=j, k=k, t=t, ax=ax, cmap=colormap, vmax=vmax, vmin=vmin)
    plt.tight_layout()
    return fig

def get_attr_from_field(attr):
    comp, attr = attr.split('_')
    return FIELD['model']._components[comp.lower()][attr]

@state.change("figure_xsize", "activeField", "activeStep", "xslice", "colormap")
def update_xslice(figure_xsize, activeField, activeStep, xslice, colormap, **kwargs):
    _ = kwargs
    if activeField is None:
        return
    comp_name, attr = activeField.split('_')
    comp_name = comp_name.lower()
    component = getattr(FIELD['model'], comp_name)
    if isinstance(component, deepfield.field.Rock):
        activeStep = None

    ctrl.update_xslice(create_slice(component, attr,
                                    i=xslice,
                                    j=None,
                                    k=None, t=activeStep, colormap=colormap,
                                    figure_size=get_figure_size(figure_xsize)))

@state.change("figure_ysize", "activeField", "activeStep", "yslice", "colormap")
def update_yslice(figure_ysize, activeField, activeStep, yslice, colormap, **kwargs):
    _ = kwargs
    if activeField is None:
        return

    comp_name, attr = activeField.split('_')
    comp_name = comp_name.lower()
    component = getattr(FIELD['model'], comp_name)
    if isinstance(component, deepfield.field.Rock):
        activeStep = None
    ctrl.update_yslice(create_slice(component, attr,
                                    i=None,
                                    j=yslice,
                                    k=None, t=activeStep, colormap=colormap,
                                    figure_size=get_figure_size(figure_ysize)))

@state.change("figure_zsize", "activeField", "activeStep", "zslice", "colormap")
def update_zslice(figure_zsize, activeField, activeStep, zslice, colormap, **kwargs):
    _ = kwargs
    if activeField is None:
        return
    comp_name, attr = activeField.split('_')
    comp_name = comp_name.lower()
    component = getattr(FIELD['model'], comp_name)
    if isinstance(component, deepfield.field.Rock):
        activeStep = None
    ctrl.update_zslice(create_slice(component, attr,
                                    i=None,
                                    j=None,
                                    k=zslice, t=activeStep, colormap=colormap,
                                    figure_size=get_figure_size(figure_zsize)))


@state.change("figure_cbar_size", "activeField", "activeStep", "colormap")
def update_colorbar(figure_cbar_size, activeField, activeStep, zslice, colormap, **kwargs):
    _ = kwargs
    if activeField is None:
        return
    comp_name, attr = activeField.split('_')
    comp_name = comp_name.lower()
    component = getattr(FIELD['model'], comp_name)
    if isinstance(component, deepfield.field.Rock):
        activeStep = None
    figure, ax = plt.subplots(**get_figure_size(figure_cbar_size))
    vmin, vmax = get_data_limits(component, attr, activeStep)
    figure.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=colormap), cax=ax, orientation='horizontal')
    plt.tight_layout()
    ctrl.update_colorbar(figure)



def render_2d():
    vuetify.VSlider(
        v_if='need_time_slider',
        min=0,
        max=("max_timestep",),
        step=1,
        v_model=('activeStep', 0),
        label="Timestep",
        classes="mt-5 mr-5 ml-5",
        hide_details=False,
        dense=False
        )
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="width:90%; height: 70%; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0 fill-height'):
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
                    figure = matplotlib.Figure(plt.figure(**get_figure_size(state['figure_xsize'])),
                        style="position: absolute")
                    ctrl.update_xslice = figure.update
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
                    figure = matplotlib.Figure(plt.figure(**get_figure_size(state['figure_ysize'])),
                        style="position: absolute")
                    ctrl.update_yslice = figure.update
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
                    figure = matplotlib.Figure(plt.figure(**get_figure_size(state['figure_xsize'])),
                        style="position: absolute")
                    ctrl.update_zslice = figure.update
        with vuetify.VRow(style="width:70%; height: 10%; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0'):
                with trame.SizeObserver("figure_cbar_size"):
                    figure = matplotlib.Figure(plt.figure(**get_figure_size(state['figure_csize'])),
                        style="position: absolute")
                ctrl.update_colorbar = figure.update


state.gridData = True
state.wellData = True

@state.change("data1dToShow")
def update1dWidgets(data1dToShow, **kwargs):
    _ = kwargs
    if data1dToShow is None:
        return
    state.gridData = data1dToShow in FIELD['data1d']['states']
    state.wellData = data1dToShow in FIELD['data1d']['wells']
    if state.wellData:
        state.gridItemToShow = None
        wellnames = []
        for well in FIELD['model'].wells:
            if 'RESULTS' in well:
                if data1dToShow in well.RESULTS:
                    wellnames.append(well.name)
        state.wellnames = wellnames
    if state.gridData:
        state.wellNameToShow = None

def add_line_to_plot():
    fig = PLOTS['plot1d']
    if fig is None:
        return

    if state.data1dToShow is None:
        return

    if state.gridData:
        data = FIELD['model'].states[state.data1dToShow]
        cells = np.array([state.i_cell, state.j_cell, state.k_cell])
        avr = cells == 'Average'
        if np.any(avr):
            ids = np.where(avr)[0]
            data = data.mean(axis=tuple(ids+1))
        cells = cells[~avr].astype(int)
        if len(cells) > 0:
            data = data[:, *cells]
        dates = FIELD['model'].result_dates.strftime("%Y-%m-%d")
        name = state.data1dToShow

    if state.wellData:
        df = FIELD['model'].wells[state.wellNameToShow].RESULTS
        data = df[state.data1dToShow]
        dates = df.DATE.dt.strftime("%Y-%m-%d")
        name = state.wellNameToShow + '/' + state.data1dToShow

    fig.add_trace(go.Scatter(
        x=dates,
        y=data,
        name=name,
        line=dict(width=2)
    ), secondary_y=state.secondAxis)

    fig.update_xaxes(title_text="Date")
    ctrl.update_plot(fig)

ctrl.add_line_to_plot = add_line_to_plot

def clean_plot():
    PLOTS['plot1d'].data = []
    ctrl.update_plot(PLOTS['plot1d'])

ctrl.clean_plot = clean_plot

def remove_last_line():
    if not PLOTS['plot1d'].data:
        return
    PLOTS['plot1d'].data = PLOTS['plot1d'].data[:-1]
    ctrl.update_plot(PLOTS['plot1d'])

ctrl.remove_last_line = remove_last_line

@state.change("figure_size")
def update_plot_size(figure_size, **kwargs):
    _ = kwargs
    if figure_size is None:
        return
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 100)
    if PLOTS['plot1d'] is None:
        PLOTS['plot1d'] = make_subplots(specs=[[{"secondary_y": True}]])
        PLOTS['plot1d'].update_layout(
            showlegend=True,
            margin={'t': 30, 'r': 80, 'l': 100, 'b': 80}
            )
    PLOTS['plot1d'].update_layout(height=height, width=width)
    ctrl.update_plot(PLOTS['plot1d'])

def render_1d():
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow():
            with vuetify.VCol():
                vuetify.VSelect(
                    v_model=("data1dToShow", None),
                    items=("data1d", ),
                    label="Select data"
                    )
            with vuetify.VCol():
                vuetify.VSelect(
                    disabled=("wellData",),
                    v_model=("i_cell", 'Average'),
                    items=("i_cells", ),
                    label="I index"
                    )
            with vuetify.VCol():
                vuetify.VSelect(
                    disabled=("wellData",),
                    v_model=("j_cell", 'Average'),
                    items=("j_cells", ),
                    label="J index"
                    )
            with vuetify.VCol():
                vuetify.VSelect(
                    disabled=("wellData",),
                    v_model=("k_cell", 'Average'),
                    items=("k_cells", ),
                    label="K index"
                    )
            with vuetify.VCol():
                vuetify.VSelect(
                    disabled=("gridData",),
                    v_model=("wellNameToShow", None),
                    items=("wellnames", ),
                    label="Select well"
                    )
            with vuetify.VCol():
                vuetify.VSwitch(
                    v_model=("secondAxis", False),
                    color="primary",
                    label="Second Axis",
                    hide_details=True)
            with vuetify.VCol():
                vuetify.VBtn('Add line', click=ctrl.add_line_to_plot)
            with vuetify.VCol():
                vuetify.VBtn('Undo', click=ctrl.remove_last_line)
            with vuetify.VCol():
                vuetify.VBtn('Clean', click=ctrl.clean_plot)
        with vuetify.VRow(style="width:90vw; height: 60vh; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0'):
                with trame.SizeObserver("figure_size"):
                    ctrl.update_plot = plotly.Figure(**CHART_STYLE).update

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
