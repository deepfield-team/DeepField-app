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
state.tables = []
state.domains = []
state.domainMin = 0
state.domainMax = 0
state.domainStep = None
state.needDomain = False
state.recentFiles = []
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
    activeStep = int(activeStep)
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

def filter_path(path):
    "True if path is a directory or has .data or .hdf5 extension."
    if os.path.isdir(path):
        return True
    _, ext = os.path.splitext(path)
    return ext.lower() in ['.data', '.hdf5']

@state.change("user_request")
def get_path_variants(user_request, **kwargs):
    paths = list(glob(user_request + "*"))
    state.dir_list = [p for p in paths if filter_path(p)]

def load_file(*args, **kwargs):
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

def render_home():
    with html.Div(style='position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 80vw; height: 10vh'):
        with vuetify.VContainer():
            with vuetify.VRow():
                with vuetify.VCol():
                    with vuetify.VTextField(
                                v_model=("user_request", ""),
                                label="Input reservoir model path",
                                clearable=True,
                                name="searchInput"
                            ):
                            with vuetify.Template(v_slot_append=True,
                                properties=[("v_slot_append", "v-slot:append")],):
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
                with vuetify.VSlider(
                    v_if='need_time_slider',
                    min=0,
                    max=("max_timestep",),
                    step=1,
                    v_model=('activeStep', 0),
                    label="Timestep", 
                    classes="mt-5 mr-5 ml-5",
                    hide_details=False,
                    dense=False
                    ):
                    with vuetify.Template(v_slot_append=True,
                        properties=[("v_slot_append", "v-slot:append")],):
                        vuetify.VTextField(
                            v_model="activeStep",
                            density="compact",
                            style="width: 80px",
                            type="number",
                            variant="outlined",
                            hide_details=True)
                view = vtk_widgets.VtkRemoteView(
                    render_window,
                    **VTK_VIEW_SETTINGS
                    )
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

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

@state.change("figure_xsize", "activeField", "activeStep", "xslice", "colormap")
def update_xslice(figure_xsize, activeField, activeStep, xslice, colormap, **kwargs):
    _ = kwargs
    figure_size = figure_xsize
    if figure_size is None:
        return
    if activeField is None:
        return
    activeStep = int(activeStep)
    xslice = int(xslice)
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("weight", 300)
    data = get_attr_from_field(activeField)
    if activeField.split('_')[0].lower() == 'states':
        arr = data[activeStep, xslice-1].T
    else:
        arr = data[xslice-1].T
    ctrl.update_xslice(create_slice(arr, width, height, colormap))

@state.change("figure_ysize", "activeField", "activeStep", "yslice", "colormap")
def update_yslice(figure_ysize, activeField, activeStep, yslice, colormap, **kwargs):
    _ = kwargs
    figure_size = figure_ysize
    if figure_size is None:
        return
    if activeField is None:
        return
    activeStep = int(activeStep)
    yslice = int(yslice)
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 300)
    data = get_attr_from_field(activeField)
    if activeField.split('_')[0].lower() == 'states':
        arr = data[activeStep, :, yslice-1].T
    else:
        arr = data[:, yslice-1].T
    ctrl.update_yslice(create_slice(arr, width, height, colormap))

@state.change("figure_zsize", "activeField", "activeStep", "zslice", "colormap")
def update_zslice(figure_zsize, activeField, activeStep, zslice, colormap, **kwargs):
    _ = kwargs
    figure_size = figure_zsize
    if figure_size is None:
        return
    if activeField is None:
        return
    activeStep = int(activeStep)
    zslice = int(zslice)
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 300)
    data = get_attr_from_field(activeField)
    if activeField.split('_')[0].lower() == 'states':
        arr = data[activeStep, :, :, zslice-1].T
    else:
        arr = data[:, :, zslice-1].T
    ctrl.update_zslice(create_slice(arr, width, height, colormap))

def render_2d():
    with vuetify.VSlider(
        v_if='need_time_slider',
        min=0,
        max=("max_timestep",),
        step=1,
        v_model=('activeStep', 0),
        label="Timestep", 
        classes="mt-5 mr-5 ml-5",
        hide_details=False,
        dense=False
        ):
        with vuetify.Template(v_slot_append=True,
            properties=[("v_slot_append", "v-slot:append")],):
            vuetify.VTextField(
                v_model="activeStep",
                density="compact",
                style="width: 80px",
                type="number",
                variant="outlined",
                hide_details=True)
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="width:90%; height: 80%; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0'):
                with vuetify.VSlider(
                    min=1,
                    max=("dimens[0]",),
                    step=1,
                    v_model=('xslice', 1),
                    label="x", 
                    classes="mt-5 mr-5 ml-5",
                    hide_details=False,
                    dense=False
                    ):
                    with vuetify.Template(v_slot_append=True,
                            properties=[("v_slot_append", "v-slot:append")],):
                            vuetify.VTextField(
                                v_model="xslice",
                                density="compact",
                                style="width: 80px",
                                type="number",
                                variant="outlined",
                                hide_details=True)
                with trame.SizeObserver("figure_xsize"):
                    ctrl.update_xslice = plotly.Figure(**CHART_STYLE).update
            with vuetify.VCol(classes='pa-0'):
                with vuetify.VSlider(
                    min=1,
                    max=("dimens[1]",),
                    step=1,
                    v_model=('yslice', 1),
                    label="y", 
                    classes="mt-5 mr-5 ml-5",
                    hide_details=False,
                    dense=False
                    ):
                    with vuetify.Template(v_slot_append=True,
                        properties=[("v_slot_append", "v-slot:append")],):
                        vuetify.VTextField(
                            v_model="yslice",
                            density="compact",
                            style="width: 80px",
                            type="number",
                            variant="outlined",
                            hide_details=True)
                with trame.SizeObserver("figure_ysize"):
                    ctrl.update_yslice = plotly.Figure(**CHART_STYLE).update
            with vuetify.VCol(classes='pa-0'):
                with vuetify.VSlider(
                    min=1,
                    max=("dimens[2]",),
                    step=1,
                    v_model=('zslice', 1),
                    label="z", 
                    classes="mt-5 mr-5 ml-5",
                    hide_details=False,
                    dense=False
                    ):
                    with vuetify.Template(v_slot_append=True,
                        properties=[("v_slot_append", "v-slot:append")],):
                        vuetify.VTextField(
                            v_model="zslice",
                            density="compact",
                            style="width: 80px",
                            type="number",
                            variant="outlined",
                            hide_details=True)
                with trame.SizeObserver("figure_zsize"):
                    ctrl.update_zslice = plotly.Figure(**CHART_STYLE).update


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
        icells = cells[~avr].astype(int)
        if len(icells) > 0:
            data = data[:, *icells]
        dates = FIELD['model'].result_dates.strftime("%Y-%m-%d")
        if np.any(avr):
            cells[avr] = ":"
        name = '{} ({}, {}, {})'.format(state.data1dToShow, *cells)

    if state.wellData:
        if state.wellNameToShow is None:
            return
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

@state.change("tableToShow")
def updateTableWidgets(tableToShow, **kwargs):
    if tableToShow is None:
        return
    table = FIELD['model'].tables[tableToShow]
    if len(table.domain) == 1:
        # state.domains = []
        state.needDomain = False
    else:
        state.domainMin = table.index.get_level_values(0).min()
        state.domainMax = table.index.get_level_values(0).max()
        state.domainStep = (state.domainMax - state.domainMin) / 100
        # state.domains = list(sorted(set(table.index.get_level_values(0))))
        state.needDomain = True

def plot_1d_table(fig, table):
    colors = px.colors.qualitative.Plotly
    x = table.index.values
    layout = {}

    for i, col in enumerate(table.columns):
        c = colors[i%len(colors)]
        fig.add_trace(go.Scatter(
            x=x,
            y=table[col].values,
            yaxis=None if i==0 else 'y'+str(i+1),
            name=col,
            line=dict(width=2, color=c)
            ))
        key = 'yaxis' if i==0 else 'yaxis' + str(i+1)
        layout[key] = dict(
            title=dict(text=col, font=dict(color=c)),
            side="right" if i > 0 else None,
            anchor="free" if i > 0 else None,
            overlaying="y" if i > 0 else None,
            autoshift=True if i > 0 else None,
            tickfont=dict(color=c)
            )

    fig.update_layout(
        xaxis=dict(domain=[0, 1.1-0.1*len(table.columns)]),
        **layout)
    fig.update_xaxes(title_text=table.domain[0])
    return fig

def plot_table(tableToShow, domainToShow, height, width):
    fig = go.Figure()
    fig.update_layout(
        height=height,
        width=width,
        showlegend=False,
        margin={'t': 30, 'r': 80, 'l': 100, 'b': 80}
        )

    if tableToShow is None:
        return fig

    table = FIELD['model'].tables[tableToShow]
    domain = list(table.domain)

    if len(domain) == 1:
        fig = plot_1d_table(fig, table)
    elif len(domain) == 2:
        if domainToShow is None:
            return fig
        cropped_table = table.loc[table.index.get_level_values(0) == domainToShow]
        cropped_table = cropped_table.droplevel(0)
        cropped_table.domain = [domain[1]]
        fig = plot_1d_table(fig, cropped_table)

    return fig

@state.change("figure_size", "tableToShow", "domainToShow")
def update_tplot_size(figure_size, tableToShow, domainToShow, **kwargs):
    _ = kwargs
    if figure_size is None:
        return
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 100)
    ctrl.update_tplot(plot_table(tableToShow, domainToShow, height, width))

def render_1d():
    with vuetify.VContainer(fluid=True, style='align-items: center', classes="fill-height pa-0 ma-0"):
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

        with vuetify.VRow(style="width:90vw;"):
            with vuetify.VCol():
                vuetify.VSelect(
                    v_model=("tableToShow", None),
                    items=("tables", ),
                    label="Select data"
                    )
            with vuetify.VCol():
                vuetify.VSlider(
                    disabled=("!needDomain",),
                    v_model=("domainToShow", None),
                    min=("domainMin",),
                    max=("domainMax",),
                    step=("domainStep",),
                    hide_details=False,
                    dense=False
                    )
        with vuetify.VRow(style="width:90vw; height: 60vh; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0'):
                with trame.SizeObserver("figure_size"):
                    ctrl.update_tplot = plotly.Figure(**CHART_STYLE).update

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
            vuetify.VSelect(
                v_model=('user_request',),
                label='Recent files',
                items=('recentFiles', ),
                v_if="activeTab === 'home'"
                )
            vuetify.VBtn(
                'Clean history',
                click='recentFiles = []',
                v_if="activeTab === 'home'"
                )
            vuetify.VSelect(
                v_model=('activeField', state.field_attrs[0] if state.field_attrs else None),
                label='Select data',
                items=('field_attrs', ),
                v_if="(activeTab === '3d') | (activeTab === '2d')"
                )
            vuetify.VCheckbox(
                label='Threshold selector',
                v_if="activeTab === '3d'"
                )
            vuetify.VCheckbox(
                label='Slice range selector',
                v_if="activeTab === '3d'"
                )
            vuetify.VCheckbox(
                label='Show wells',
                v_if="activeTab === '3d'"
                )
            vuetify.VCheckbox(
                label='Show faults',
                v_if="activeTab === '3d'"
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
                v_if="(activeTab === '3d') | (activeTab === '2d')"
            )
            vuetify.VSlider(
                min=0,
                max=1,
                step=0.1,
                v_model=('opacity', 1),
                label="Opacity", 
                classes="mt-8 mr-3",
                hide_details=False,
                dense=False,
                thumb_label=True,
                v_if="activeTab === '3d'"
                )

if __name__ == "__main__":
    server.start()
