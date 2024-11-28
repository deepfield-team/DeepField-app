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
state.tables = []
state.domains = []
state.domainMin = 0
state.domainMax = 0
state.domainStep = None
state.domainToShow = None
state.needDomain = False
state.domainName = None
state.recentFiles = []
state.i_cells = []
state.j_cells = []
state.k_cells = []
state.i_slice = [0, 0]
state.j_slice = [0, 0]
state.k_slice = [0, 0]


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

@state.change("i_slice", "j_slice", "k_slice")
def update_threshold_slices(i_slice, j_slice, k_slice, **kwargs):
    _ = kwargs
    if not FIELD['c_data']:
        return

    dataset = FIELD['dataset']
    vtk_array_i = dsa.numpyTovtkDataArray(FIELD['c_data']["I"])
    vtk_array_j = dsa.numpyTovtkDataArray(FIELD['c_data']["J"])
    vtk_array_k = dsa.numpyTovtkDataArray(FIELD['c_data']["K"])
    dataset.GetCellData().SetScalars(vtk_array_i)
    dataset.GetCellData().SetScalars(vtk_array_j)
    dataset.GetCellData().SetScalars(vtk_array_k)

    threshold = vtk.vtkThreshold()
    threshold.SetInputData(dataset)
    threshold.SetUpperThreshold(i_slice[1])
    threshold.SetLowerThreshold(i_slice[0])
    threshold.SetInputArrayToProcess(0, 0, 0, 1, "I")

    threshold_j = vtk.vtkThreshold()
    threshold_j.SetInputData(dataset)
    threshold_j.SetInputConnection(threshold.GetOutputPort())
    threshold_j.SetUpperThreshold(j_slice[1])
    threshold_j.SetLowerThreshold(j_slice[0])
    threshold_j.SetInputArrayToProcess(0, 0, 0, 1, "J")

    threshold_k = vtk.vtkThreshold()
    threshold_k.SetInputData(dataset)
    threshold_k.SetInputConnection(threshold_j.GetOutputPort())
    threshold_k.SetUpperThreshold(k_slice[1])
    threshold_k.SetLowerThreshold(k_slice[0])
    threshold_k.SetInputArrayToProcess(0, 0, 0, 1, "K")

    mapper = vtk.vtkDataSetMapper()                                                
    mapper.SetInputConnection(threshold_k.GetOutputPort())

    FIELD['actor'].SetMapper(mapper)
    update_field(state.activeField, state.activeStep, view_update=False)
    update_cmap(state.colormap)
    return

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

def render_home():
    with html.Div(style='position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 80vw; height: 10vh'):
        with vuetify.VContainer():
            with vuetify.VRow():
                with vuetify.VCol():
                    with vuetify.VTextField(
                                v_model=("user_request", ""),
                                label="Input reservoir model path",
                                clearable=True,
                                name="searchInput",
                                tabindex="1"
                            ):
                            with vuetify.Template(v_slot_append=True,
                                properties=[("v_slot_append", "v-slot:append")],):
                                vuetify.VBtn('Load', tabindex="0", click=ctrl.load_file)
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

def create_slice(component, att, i, j, k, t, i_line, j_line, k_line,
                 colormap, figure_size, vmin, vmax):
    fig, ax = plt.subplots(**figure_size)
    component.show_slice(attr=att, i=i, j=j, k=k, t=t,
                         i_line=i_line, j_line=j_line, k_line=k_line,
                         ax=ax, cmap=colormap, vmax=vmax, vmin=vmin)
    fig.tight_layout()
    return fig

def create_cbar(colormap, figure_size, vmin, vmax):
    fig, ax = plt.subplots(**get_figure_size(figure_size))
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                cmap=colormap),
                 cax=ax, orientation='horizontal')
    fig.tight_layout()
    return fig

@state.change("figure_size", "figure_cbar_size",
              "activeField", "activeStep",
              "xslice", "yslice", "zslice", "colormap")
def update_slices(figure_size, figure_cbar_size,
    activeField, activeStep, xslice, yslice, zslice, colormap, **kwargs):
    _ = kwargs
    if activeField is None:
        return
    
    comp_name, attr = activeField.split('_')
    comp_name = comp_name.lower()
    component = getattr(FIELD['model'], comp_name)

    activeStep = int(activeStep) if comp_name == 'states' else None
    xslice, yslice, zslice = int(xslice), int(yslice), int(zslice)
    vmin, vmax = get_data_limits(component, attr, activeStep)
    figsize = get_figure_size(figure_size)

    plt.close("all")
    ctrl.update_xslice(create_slice(component, attr,
                                    i=xslice,
                                    j=None,
                                    k=None,
                                    t=activeStep,
                                    j_line=yslice,
                                    i_line=None,
                                    k_line=zslice,
                                    colormap=colormap,
                                    figure_size=figsize,
                                    vmin=vmin,
                                    vmax=vmax))
        
    ctrl.update_yslice(create_slice(component, attr,
                                    i=None,
                                    j=yslice,
                                    k=None,
                                    t=activeStep,
                                    i_line=xslice,
                                    j_line=None,
                                    k_line=zslice,
                                    colormap=colormap,
                                    figure_size=figsize,
                                    vmin=vmin,
                                    vmax=vmax))
    
    ctrl.update_zslice(create_slice(component, attr,
                                    i=None,
                                    j=None,
                                    k=zslice,
                                    t=activeStep,
                                    i_line=xslice,
                                    j_line=yslice,
                                    k_line=None,
                                    colormap=colormap,
                                    figure_size=figsize,
                                    vmin=vmin,
                                    vmax=vmax))

    ctrl.update_colorbar(create_cbar(colormap=colormap,
                                     figure_size=figure_cbar_size,
                                     vmin=vmin,
                                     vmax=vmax))

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
                with trame.SizeObserver("figure_size"):
                    figure = matplotlib.Figure(plt.figure(**get_figure_size(state['figure_size'])),
                        style="position: absolute")
                    ctrl.update_xslice = figure.update
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
                with trame.SizeObserver("figure_size"):
                    figure = matplotlib.Figure(plt.figure(**get_figure_size(state['figure_size'])),
                        style="position: absolute")
                    ctrl.update_yslice = figure.update
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
                with trame.SizeObserver("figure_size"):
                    figure = matplotlib.Figure(plt.figure(**get_figure_size(state['figure_size'])),
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

@state.change("tableToShow", "tableXAxis")
def updateTableWidgets(tableToShow, tableXAxis, **kwargs):
    if tableToShow is None:
        return
    table = FIELD['model'].tables[tableToShow]
    if len(table.domain) == 1:
        state.needDomain = False
    elif len(table.domain) == 2:
        state.domains = list(table.index.names)
        if tableXAxis is not None:
            i = (state.domains.index(tableXAxis) + 1)%2
            state.domainName = state.domains[i]
            state.domainMin = table.index.get_level_values(i).min()
            state.domainMax = table.index.get_level_values(i).max()
            state.domainStep = (state.domainMax - state.domainMin) / 100
            state.domainToShow = (state.domainMin + state.domainMax) / 2
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
    fig.update_xaxes(title_text=table.index.name)
    return fig

def plot_table(tableToShow, tableXAxis, domainToShow, height, width):
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
        if tableXAxis is None:
            return fig
        if domainToShow is None:
            return fig

        new_table = pd.DataFrame(columns=table.columns)
        vals = [list(set(table.index.get_level_values(0))),
                list(set(table.index.get_level_values(1)))]

        i = list(table.index.names).index(tableXAxis)
        x = np.linspace(min(vals[i])*1.001, max(vals[i])*0.999, 100)

        inp = np.zeros((len(x), 2))
        inp[:, i] = x
        inp[:, (i+1)%2] = domainToShow

        new_table = pd.DataFrame(table(inp),
                                 columns=table.columns,
                                 index=x)
        new_table.index.name = tableXAxis
        fig = plot_1d_table(fig, new_table)

    return fig

@state.change("figure_size_1d", "tableToShow", "tableXAxis", "domainToShow")
def update_tplot_size(figure_size_1d, tableToShow, tableXAxis, domainToShow, **kwargs):
    _ = kwargs
    if figure_size_1d is None:
        return
    bounds = figure_size_1d.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 100)
    ctrl.update_tplot(plot_table(tableToShow, tableXAxis, domainToShow, height, width))

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
                with trame.SizeObserver("figure_size_1d"):
                    ctrl.update_plot = plotly.Figure(**CHART_STYLE).update

        with vuetify.VRow(style="width:90vw;"):
            with vuetify.VCol():
                vuetify.VSelect(
                    v_model=("tableToShow", None),
                    items=("tables", ),
                    label="Select data"
                    )
            with vuetify.VCol():
                vuetify.VSelect(
                    v_model=("tableXAxis", None),
                    items=("domains", ),
                    label="Select x-axis",
                    disabled=("!needDomain",),
                    )
            with vuetify.VCol():
                with vuetify.VSlider(
                    label=('domainName',),
                    disabled=("!needDomain",),
                    v_model=("domainToShow", ),
                    min=("domainMin",),
                    max=("domainMax",),
                    step=("domainStep",),
                    hide_details=False,
                    dense=False
                    ):
                    with vuetify.Template(v_slot_append=True,
                        properties=[("v_slot_append", "v-slot:append")],):
                        vuetify.VTextField(
                            v_model=("domainToShow",),
                            density="compact",
                            style="width: 80px",
                            type="number",
                            variant="outlined",
                            hide_details=True)
        with vuetify.VRow(style="width:90vw; height: 60vh; margin 0;", classes='pa-0'):
            with vuetify.VCol(classes='pa-0'):
                with trame.SizeObserver("figure_size_1d"):
                    ctrl.update_tplot = plotly.Figure(**CHART_STYLE).update

ctrl.on_server_ready.add(ctrl.view_update)

with VAppLayout(server) as layout:
    with layout.root:
        with vuetify.VAppBar(app=True, clipped_left=True):
            vuetify.VAppBarNavIcon(click='drawer =! drawer')

            vuetify.VToolbarTitle("DeepField")
            with vuetify.VTabs(v_model=('activeTab', 'home'), style='flex: 2'):
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
                v_if="activeTab === '3d'",
                style='height: 8vh'
                )
            vuetify.VCheckbox(
                label='Slice range selector',
                v_if="activeTab === '3d'",
                classes='pa-0 ma-0',
                style='height: 8vh'
                )
            vuetify.VCheckbox(
                label='Show wells',
                v_if="activeTab === '3d'",
                style='height: 8vh'
                )
            vuetify.VCheckbox(
                label='Show faults',
                v_if="activeTab === '3d'",
                style='height: 8vh'
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
            vuetify.VRangeSlider(
                min=1,
                max=("dimens[0]",),
                step=1,                
                v_model=("i_slice", ),
                label="I",
                thumb_label = True,
                hide_details=False,
                classes="mt-8 mr-3",
                )
            vuetify.VDivider(vertical=True, classes="mx-2")
            vuetify.VRangeSlider(
                min=1,
                max=("dimens[1]",),
                step=1,                
                v_model=("j_slice", ),
                label="J",
                thumb_label = True,
                hide_details=False,
                classes="mt-8 mr-3",
                )
            vuetify.VDivider(vertical=True, classes="mx-2")
            vuetify.VRangeSlider(
                min=1,
                max=("dimens[2]",),
                step=1,                
                v_model=("k_slice", ),
                label="K",
                thumb_label = True,
                hide_details=False,
                classes="mt-8 mr-3",
                )

if __name__ == "__main__":
    server.start()
