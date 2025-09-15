"Home page."
import os
from glob import glob
import numpy as np
import pandas as pd
from anytree import PreOrderIter
import vtk
from vtk.util.numpy_support import numpy_to_vtk # pylint: disable=no-name-in-module, import-error

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
)

from vtkmodules.util import numpy_support

from scipy import ndimage

from trame.widgets import html, vuetify3 as vuetify

from deepfield import Field

from .config import state, ctrl, FIELD, renderer, dataset_names, actor_names
from .common import reset_camera, set_active_scalars
from .view_3d import rw_style, render_window

import asyncio
from trame.app import asynchronous


USER_DIR = os.path.expanduser("~")

state.user_request = USER_DIR
state.user_click_request = None
state.dirList = []
state.pathIndex = None
state.updateDirList = True
state.initialDirState = True
state.showDirList = False
state.recentFiles = []
state.loading = False
state.showHistory = False
state.emptyHistory = True
state.errMessage = ''
state.loadFailed = False
state.modelID = 0

state.field_attrs = []
state.activeField = None
state.wellnames = []
state.dimens = [0, 0, 0]
state.max_timestep = 0
state.stateDate = None
state.startDate = None
state.lastDate = None
state.data1d = []
state.tables = []
state.i_cells = []
state.j_cells = []
state.k_cells = []
state.i_slice = [0, 0]
state.i_slice_0 = state.i_slice_1 = 0
state.j_slice = [0, 0]
state.j_slice_0 = state.j_slice_1 = 0
state.k_slice = [0, 0]
state.k_slice_0 = state.k_slice_1 = 0
state.field_slice = [0, 0]
state.field_slice_0 = state.field_slice_1 = 0
state.field_slice_min = 0
state.field_slice_max = 0
state.n_field_steps = 100
state.field_slice_step = 0
state.show_well_blocks = False
state.total_cells = 0
state.active_cells = 0
state.units = 0
state.pore_volume = 0
state.num_wells = 0
state.fluids = []
state.total_oil_production = 0
state.total_wat_production = 0
state.total_gas_production = 0
state.components_attrs = {
    'grid': [],
    'rock': [],
    'states': [],
    'tables': [],
    'wells': [],
    'faults': [],
    'aquifers': [],
    }
state.units1 = ''
state.units2 = ''
state.units3 = ''
state.units4 = ''
state.units5 = ''

state.pore_vol = 0
state.oil_vol = 0
state.wat_vol = 0
state.gas_vol = 0

def filter_path(path):
    "True if path is a directory or has .data or .hdf5 extension."
    if os.path.isdir(path):
        return True
    _, ext = os.path.splitext(path)
    return ext.lower() in ['.data', '.hdf5']

@state.change("user_click_request")
def handle_user_click_request(user_click_request, **kwargs):
    if user_click_request is None:
        return
    state.updateDirList = True
    _, ext = os.path.splitext(user_click_request)
    if ext.lower() in ['.data', '.hdf5']:
        state.user_request = user_click_request
        return
    if os.path.isdir(user_click_request):
        user_click_request += os.sep
    state.user_request = user_click_request
    state.pathIndex = None

@state.change("user_request")
def get_path_variants(user_request, **kwargs):
    "Collect and filter paths."
    _ = kwargs
    state.loading = False
    state.showHistory = False
    state.showDirList = not state.initialDirState
    paths = list(glob(user_request + "*")) if user_request is not None else []
    if state.updateDirList:
        state.dirList = [p for p in paths if filter_path(p)]

@asynchronous.task
async def load_file_async():
    with state:
        state.loading = True
        state.showHistory = False
        state.showDirList = False

    field = Field(state.user_request)

    try:
        await asyncio.to_thread(field.load)
    except Exception as err:
        with state:
            state.errMessage = str(err)
            state.loadFailed = True
            state.loading = False
        return

    FIELD['model'] = field
    FIELD["model_copy"] = None

    with state:
        try:
            process_field(field)
        except Exception as err:
            state.errMessage = str(err)
            state.loadFailed = True
            state.loading = False
            return

        if state.user_request not in state.recentFiles:
            state.recentFiles = state.recentFiles + [state.user_request]
            state.emptyHistory = False
        state.loading = False
        state.errMessage = ''
        state.loadFailed = False
        state.modelID += 1

ctrl.load_file_async = load_file_async


def process_field(field):
    "Prepare field data for visualization."

    for name in actor_names.__dict__.values():
        if name in FIELD:
            renderer.RemoveActor(FIELD[name])

    render_window.Render()
    reset_camera()
    ctrl.view_update()

    vtk_grid = field.grid.vtk_grid

    ind_i, ind_j, ind_k = np.unravel_index(field.grid.actnum_ids, field.grid.dimens)
    for name, val in zip(('I', 'J', 'K'), (ind_i, ind_j, ind_k)):
        array = numpy_to_vtk(val)
        array.SetName(name)
        vtk_grid.GetCellData().AddArray(array)

    well_dist = get_well_blocks(field)
    array = numpy_to_vtk(well_dist.ravel()[field.grid.actnum_ids])
    array.SetName('WELL_BLOCKS')
    vtk_grid.GetCellData().AddArray(array)

    FIELD['grid'] = vtk_grid

    state.dimens = [int(x) for x in field.grid.dimens]

    rock_attrs = ['ROCK_'+attr.upper() for attr in FIELD['model'].rock.attributes]
    state_attrs = ['STATES_'+attr.upper() for attr in FIELD['model'].states.attributes]

    state.field_attrs = rock_attrs + state_attrs
    state.activeField = ('ROCK_PERMZ' if 'ROCK_PERMZ' in state.field_attrs
        else state.field_attrs[0])

    attrs = []
    for well in field.wells:
        if 'RESULTS' in well:
            attrs.extend([k for k in well.RESULTS.columns if k != 'DATE'])
    attrs = sorted(list(set(attrs)))
    state.wellsAttrs = attrs

    attrs = list(field.states.attributes)
    state.max_timestep = field.states[attrs[0]].shape[0] - 1 if attrs else 0
    state.statesAttrs = attrs

    attrs = field.tables.attributes
    state.tables = [t for t in attrs if field.tables[t].domain]

    state.data1d = state.statesAttrs + state.wellsAttrs

    get_field_info(field)

    bbox = field.grid.bounding_box
    ds = abs(bbox[3:] - bbox[:3])
    ds_max = ds.max()
    scales = ds_max / ds
    FIELD['scales'] = scales

    add_scalars()

    prepare_slices()

    add_wells(field)

    add_faults(field)
    
    render_window.Render()
    reset_camera()
    ctrl.view_update()
    ctrl.default_view()

ctrl.load_file = load_file_async

def prepare_slices():
    "Get slice data and slice ranges."
    state.i_slice_0, state.i_slice_1 = 1, state.dimens[0]
    state.i_slice = [state.i_slice_0, state.i_slice_1]

    state.j_slice_0, state.j_slice_1 = 1, state.dimens[1]
    state.j_slice = [state.j_slice_0, state.j_slice_1]

    state.k_slice_0, state.k_slice_1 = 1, state.dimens[2]
    state.k_slice = [state.k_slice_0, state.k_slice_1]

    state.i_cells = ['Average'] + list(range(1, state.dimens[0]+1))
    state.j_cells = ['Average'] + list(range(1, state.dimens[1]+1))
    state.k_cells = ['Average'] + list(range(1, state.dimens[2]+1))

    state.xslice = 1
    state.yslice = 1
    state.zslice = 1

    state.i_cells = ['Average'] + list(range(1, state.dimens[0]+1))
    state.j_cells = ['Average'] + list(range(1, state.dimens[1]+1))
    state.k_cells = ['Average'] + list(range(1, state.dimens[2]+1))

def get_field_info(field):
    "Collect field info."
    state.total_cells = int(np.prod(field.grid.dimens))
    state.active_cells = len(field.grid.actnum_ids)

    actnum = field.grid.actnum

    soil = field.states.SOIL[0][actnum] if 'SOIL' in field.states else 0
    swat = field.states.SWAT[0][actnum] if 'SWAT' in field.states else 0
    sgas = field.states.SGAS[0][actnum] if 'SGAS' in field.states else 0

    c_vols = field.grid.cell_volumes

    p_vols = field.rock.poro[actnum] * c_vols

    state.pore_vol = np.round(p_vols.sum(), 2)
    state.oil_vol = np.round((p_vols * soil).sum(), 2)
    state.wat_vol = np.round((p_vols * swat).sum(), 2)
    state.gag_vol = np.round((p_vols * sgas).sum(), 2)

    state.fluids = list(field.meta['FLUIDS'])

    rates = field.wells.total_rates.fillna(0)
    rates = rates if len(rates) else {}
    state.total_oil_production = np.round(rates['WOPR'].sum(), 2) if 'WOPR' in rates else 0
    state.total_wat_production = np.round(rates['WWPR'].sum(), 2) if 'WWPR' in rates else 0
    state.total_gas_production = np.round(rates['WGPR'].sum(), 2) if 'WGPR' in rates else 0

    state.units1 = field.meta['HUNITS'][0]
    state.units2 = field.meta['HUNITS'][1]
    state.units3 = field.meta['HUNITS'][2]
    state.units4 = field.meta['HUNITS'][3]
    state.units5 = field.meta['HUNITS'][4]

    state.num_wells = len(field.wells.names)

    state.components_attrs = {}
    for name, comp in field.items():
        if name in ['wells', 'faults']:
            attrs = []
            for node in PreOrderIter(comp.root):
                attrs.extend(list(node.attributes))
            attrs = list(set(attrs))
        else:
            attrs = list(comp.attributes)
        state.components_attrs[name] = attrs

    FIELD['dates'] = field.result_dates if len(field.result_dates) else np.array([pd.to_datetime(field.meta['START'])])

    state.stateDate = FIELD['dates'][0].strftime('%Y-%m-%d')
    state.startDate = FIELD['dates'][0].strftime('%Y-%m-%d')
    state.lastDate = FIELD['dates'][-1].strftime('%Y-%m-%d')

def add_scalars():
    "Add actor for scalars."
    actor = vtkActor()
    actor.SetScale(*FIELD['scales'])

    set_active_scalars(update_range=True)

    vtk_grid = FIELD['grid']

    gf = vtk.vtkGeometryFilter()
    gf.SetInputData(vtk_grid)
    gf.Update()
    outer = gf.GetOutput()

    mapper = vtkDataSetMapper()
    mapper.SetInputData(outer)
    mapper.SetScalarRange(vtk_grid.GetScalarRange())
    actor.SetMapper(mapper)

    renderer.AddActor(actor)
    FIELD[actor_names.main] = actor

def get_well_blocks(field):
    "Get mask for well blocks."
    mask = np.full(field.grid.dimens, 0)
    field.wells.get_blocks()
    for well in field.wells:
        mask[*well.blocks.T] = 1
    return mask

def add_wells(field):
    "Add actor for wells."
    namedColors = vtk.vtkNamedColors()
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    points_links = vtk.vtkPoints()
    cells_links = vtk.vtkCellArray()

    grid = field.grid
    z_min = grid.bounding_box[2]
    dz = grid.bounding_box[-1] - z_min
    z_min = z_min - 0.1*dz

    field.wells.drop_incomplete(logger=field._logger, required=['WELLTRACK'])

    n_wells = len(field.wells.names)
    labeled_points = vtk.vtkPoints()
    labels = vtk.vtkStringArray()
    labels.SetNumberOfValues(n_wells)
    labels.SetName("labels")

    well_colors = vtk.vtkUnsignedCharArray()
    well_colors.SetNumberOfComponents(3)

    colors = vtk.vtkNamedColors()

    for i, well in enumerate(field.wells):
        labels.SetValue(i, well.name)

        welltrack = well.welltrack[:, :3]

        first_point = welltrack[0, :3].copy()
        first_point[-1] = z_min

        labeled_points.InsertNextPoint(first_point*FIELD['scales'])

        point_ids = []
        for row in welltrack:
            point_ids.append(points.InsertNextPoint(row[:3]))

        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(len(point_ids))
        for j, id in enumerate(point_ids):
            polyLine.GetPointIds().SetId(j, id)
        cells.InsertNextCell(polyLine)
        well_colors.InsertNextTypedTuple(namedColors.GetColor3ub("Red"))

        point_ids = []
        for row in [first_point, welltrack[0, :3]]:
            point_ids.append(points_links.InsertNextPoint(row))

        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(len(point_ids))
        for j, id in enumerate(point_ids):
            polyLine.GetPointIds().SetId(j, id)
        cells_links.InsertNextCell(polyLine)

    label_polyData = vtk.vtkPolyData()
    label_polyData.SetPoints(labeled_points)
    label_polyData.GetPointData().AddArray(labels)
    label_mapper = vtk.vtkLabeledDataMapper()
    label_mapper.SetInputData(label_polyData)
    label_mapper.SetFieldDataName('labels')
    label_mapper.SetLabelModeToLabelFieldData()
    label_actor = vtk.vtkActor2D()
    label_actor.SetMapper(label_mapper)

    renderer.AddActor(label_actor)
    FIELD[actor_names.well_labels] = label_actor

    FIELD[dataset_names.wells] = vtk.vtkPolyData()
    FIELD[dataset_names.wells].SetPoints(points)
    FIELD[dataset_names.wells].SetLines(cells)
    FIELD[dataset_names.wells].GetCellData().SetScalars(well_colors)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(FIELD[dataset_names.wells])
    wells_actor = vtk.vtkActor()
    wells_actor.SetScale(*FIELD['scales'])
    wells_actor.SetMapper(mapper)
    wells_actor.GetProperty().SetLineWidth(3)

    renderer.AddActor(wells_actor)
    FIELD[actor_names.wells] = wells_actor

    well_links_poly = vtk.vtkPolyData()
    well_links_poly.SetPoints(points_links)
    well_links_poly.SetLines(cells_links)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(well_links_poly)
    well_links_actor = vtk.vtkActor()
    well_links_actor.SetScale(*FIELD['scales'])
    well_links_actor.SetMapper(mapper)
    well_links_actor.GetProperty().SetLineWidth(2)
    well_links_actor.GetProperty().SetColor(colors.GetColor3d('Green'))

    renderer.AddActor(well_links_actor)
    FIELD[actor_names.well_links] = well_links_actor

def add_faults(field):
    "Add actor for faults."
    field.faults.get_blocks()
    n_segments = len(field.faults.names)

    labels = vtk.vtkStringArray()
    labels.SetNumberOfValues(n_segments)
    labels.SetName("labels")

    grid = field.grid
    z_min = grid.bounding_box[2]
    dz = grid.bounding_box[-1] - z_min
    z_min = z_min - 0.05*dz

    points = vtk.vtkPoints()
    polygons = vtk.vtkCellArray()

    labeled_points = vtk.vtkPoints()
    links_points = vtk.vtkPoints()
    links_points_ids = []
    link_cells = vtk.vtkCellArray()

    size = 0
    for i, fault in enumerate(field.faults):
        blocks = fault.blocks
        xyz = fault.faces_verts
        active = field.grid.actnum[blocks[:, 0], blocks[:, 1], blocks[:, 2]]
        xyz = xyz[active].reshape(-1, 3)
        if len(xyz) == 0:
            continue

        for p in xyz:
            points.InsertNextPoint(*p)

        labeled_points_id = labeled_points.InsertNextPoint(np.array([*xyz[0, :2], z_min])*FIELD['scales'])
        labels.SetValue(labeled_points_id, fault.name)
        links_points_ids.append(links_points.InsertNextPoint(np.array([*xyz[0, :2], z_min])))
        links_points_ids.append(links_points.InsertNextPoint(*xyz[0]))

        ids = np.arange(size, size+len(xyz))
        faces1 = np.stack([ids[::4], ids[1::4], ids[3::4]]).T
        faces2 = np.stack([ids[::4], ids[2::4], ids[3::4]]).T
        faces = np.vstack([faces1, faces2])

        for f in faces:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(3)
            for j, id in enumerate(f):
                polygon.GetPointIds().SetId(j, id)
            polygons.InsertNextCell(polygon)

        size += len(xyz)

        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(2)
        for j, id in enumerate(links_points_ids[-2:]):
            polyLine.GetPointIds().SetId(j, id)
        link_cells.InsertNextCell(polyLine)

    colors = vtk.vtkNamedColors()

    link_polyData = vtk.vtkPolyData()
    link_polyData.SetPoints(links_points)
    link_polyData.SetLines(link_cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(link_polyData)
    fault_links_actor = vtk.vtkActor()
    fault_links_actor.SetScale(*FIELD['scales'])
    fault_links_actor.SetMapper(mapper)
    (fault_links_actor.GetProperty().SetColor(colors.GetColor3d('Purple')))

    FIELD[actor_names.fault_links] = fault_links_actor
    renderer.AddActor(fault_links_actor)

    label_polyData = vtk.vtkPolyData()
    label_polyData.SetPoints(labeled_points)
    label_polyData.GetPointData().AddArray(labels)
    label_mapper = vtk.vtkLabeledDataMapper()
    label_mapper.SetInputData(label_polyData)
    label_mapper.SetFieldDataName('labels')
    label_mapper.SetLabelModeToLabelFieldData()
    label_actor = vtk.vtkActor2D()
    label_actor.SetMapper(label_mapper)
    label_actor.GetProperty().SetColor(colors.GetColor3d('Purple'))

    FIELD[actor_names.fault_labels] = label_actor
    renderer.AddActor(label_actor)

    polygon_polyData = vtk.vtkPolyData()
    polygon_polyData.SetPoints(points)
    polygon_polyData.SetPolys(polygons)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polygon_polyData)

    actor_faults = vtk.vtkActor()
    actor_faults.SetScale(*FIELD['scales'])
    actor_faults.SetMapper(mapper)
    actor_faults.GetProperty().SetColor(colors.GetColor3d('Purple'))

    renderer.AddActor(actor_faults)
    FIELD[actor_names.faults] = actor_faults

def make_empty_grid():
    "Init variables."
    grid = vtk.vtkUnstructuredGrid()

    mapper = vtkDataSetMapper()
    mapper.SetInputData(grid)
    mapper.SetScalarRange(0, 1)

    actor = vtkActor()
    actor.SetMapper(mapper)

    renderer.AddActor(actor)
    renderer.ResetCamera()

    FIELD[actor_names.main] = actor
    FIELD['grid'] = grid

def on_keydown(key_code):
    "Autocomplete path input."
    state.initialDirState = False
    state.showDirList = True
    if key_code == 'ArrowDown':
        if state.pathIndex is None:
            state.pathIndex = 0
        else:
            state.pathIndex = (state.pathIndex + 1) % len(state.dirList)
        state.updateDirList = False
        state.user_request = state.dirList[state.pathIndex]
        return
    if key_code == 'ArrowUp':
        if state.pathIndex is None:
            state.pathIndex = -1
        else:
            state.pathIndex = (state.pathIndex - 1) % len(state.dirList)
        state.updateDirList = False
        state.user_request = state.dirList[state.pathIndex]
        return
    if key_code == "Enter":
        state.updateDirList = True
        _, ext = os.path.splitext(state.user_request)
        if ext.lower() in ['.data', '.hdf5']:
            ctrl.load_file_async()
            return
        if state.pathIndex is None:
            state.pathIndex = 0
        if state.pathIndex < len(state.dirList):
            path = state.dirList[state.pathIndex]
            if os.path.isdir(path):
                path += os.sep
            state.user_request = path
            state.pathIndex = None
        return
    state.updateDirList = True

def render_home():
    "Home page layout."
    with html.Div(
        style='position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 80vw; height: 10vh'
    ):
        with vuetify.VContainer():
            with vuetify.VRow():
                with vuetify.VCol():
                    with vuetify.VTextField(
                        ref="searchInput",
                        v_model=("user_request",),
                        label="Input reservoir model path",
                        clearable=True,
                        name="searchInput",
                        autofocus=True,
                        keydown=(on_keydown, "[$event.code]"),
                        __events=["keydown"]
                    ):
                        with vuetify.Template(
                            v_slot_append=True,
                            properties=[("v_slot_append", "v-slot:append")]
                        ):
                            with vuetify.VBtn(
                                'Load',
                                click=ctrl.load_file_async,
                                disabled=("loading",)
                            ):
                                vuetify.VTooltip(
                                    text='Start reading data',
                                    activator="parent",
                                    location="top"
                                )
                        with vuetify.Template(
                            v_slot_prepend=True,
                            properties=[("v_slot_prepend", "v-slot:prepend")]
                        ):
                            with vuetify.VBtn(
                                icon=True,
                                click='showHistory = !showHistory',
                                flat=True,
                                active=('showHistory',),
                                style="background-color:transparent; backface-visibility:visible;"
                            ):
                                vuetify.VIcon("mdi-history")
                                vuetify.VTooltip(
                                    text='Show recent files',
                                    activator="parent",
                                    location="top"
                                )
                        with vuetify.Template(
                            v_slot_loader=True,
                            properties=[("v_slot_loader", "v-slot:loader")]
                        ):
                            with vuetify.VCard(
                                v_if='!loading',
                                classes="overflow-auto",
                                max_width="100%",
                                max_height="30vh"
                            ):
                                with vuetify.VList(v_if='showDirList'):
                                    with vuetify.VListItem(
                                        v_for="item, index in dirList",
                                        click="user_click_request = item"
                                    ):
                                        vuetify.VListItemTitle("{{item}}")
                                with vuetify.VList(v_if='showHistory'):
                                    with vuetify.VListItem(
                                        v_for="item, index in recentFiles",
                                        click="user_click_request = item"
                                    ):
                                        vuetify.VListItemTitle("{{item}}")
            with vuetify.VRow(classes="pa-0 ma-0"):
                with vuetify.VCol(classes="pa-0 ma-0 text-center"):
                    vuetify.VProgressCircular(
                        v_if='loading',
                        color="primary",
                        indeterminate=True,
                        size="60",
                        width="7"
                    )
                    with vuetify.VCard(v_if='loading', variant='text'):
                        vuetify.VCardText('Loading data, please wait')
                    with vuetify.VCard(
                        v_if='!showDirList & !showHistory & !loading & !loadFailed & (modelID > 0)',
                        variant='text'
                    ):
                        vuetify.VIcon('mdi-check-bold', color="success")
                        vuetify.VCardText('Loading completed')
                    with vuetify.VCard(
                        v_if='!showDirList & !showHistory & !loading & loadFailed',
                        variant='text'
                    ):
                        vuetify.VIcon('mdi-close-thick', color="error")
                        vuetify.VCardText('Loading failed: ' + '{{errMessage}}')
                    with vuetify.VCard(
                        v_if='showHistory & emptyHistory',
                        variant='text'
                    ):
                        vuetify.VCardText('History is empty.')
