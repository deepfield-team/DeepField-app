import os
import sys
from glob import glob
import numpy as np
from anytree import PreOrderIter
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
from .view_3d import update_field_slices_params

state.dirList = []
state.pathIndex = None
state.updateDirList = True
state.recentFiles = []
state.loading = False
state.loadComplete = False
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
state.j_slice = [0, 0]
state.k_slice = [0, 0]
state.field_slice = [0, 0]
state.field_slice_min = 0
state.field_slice_max = 0
state.n_field_steps = 100
state.field_slice_step = 0
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

@state.change("user_request")
def get_path_variants(user_request, **kwargs):
    _ = kwargs
    state.loading = False
    state.loadComplete = False
    state.showHistory = False
    paths = list(glob(user_request + "*")) if user_request is not None else []
    if state.updateDirList:
        state.dirList = [p for p in paths if filter_path(p)]

@state.change("loading")
def load_file(loading, **kwargs):
    _ = kwargs

    if not loading:
        return

    try:
        field = Field(state.user_request).load()
    except Exception as err:
        state.errMessage = str(err)
        state.loading = False
        state.loadFailed = True
        state.loadComplete = True
        return

    if state.user_request not in state.recentFiles:
        state.recentFiles = state.recentFiles + [state.user_request, ]
        state.emptyHistory = False

    FIELD['model'] = field
    FIELD['model_copy'] = None

    process_field(field)

    state.loading = False
    state.loadComplete = True
    state.errMessage = ''
    state.loadFailed = False
    state.modelID = state.modelID + 1

def process_field(field):
    "Prepare field data for visualization."
    dataset = field.get_vtk_dataset()
    FIELD['dataset'] = dataset

    py_ds = dsa.WrapDataObject(dataset)
    c_data = py_ds.CellData
    FIELD['c_data'] = c_data

    state.dimens = [int(x) for x in field.grid.dimens]

    state.field_attrs = [k for k in c_data.keys() if k not in ['I', 'J', 'K']]
    state.activeField = ('ROCK_PERMZ' if 'ROCK_PERMZ' in state.field_attrs
        else state.field_attrs[0])

    attrs = []
    for well in field.wells:
        if 'RESULTS' in well:
            attrs.extend([k for k in well.RESULTS.columns if k != 'DATE'])
    attrs = sorted(list(set(attrs)))
    state.wellsAttrs = attrs

    attrs = list(field.states.attributes)
    state.max_timestep = field.states[attrs[0]].shape[0] - 1 if attrs else []
    state.statesAttrs = attrs

    attrs = field.tables.attributes
    state.tables = [t for t in attrs if field.tables[t].domain]

    state.data1d = state.statesAttrs + state.wellsAttrs

    prepare_slices(dataset)

    get_field_info(field)

    bbox = field.grid.bounding_box
    ds = abs(bbox[1] - bbox[0])
    ds_max = ds.max()
    scales = ds_max / ds

    add_scalars(field, scales)

    add_wells(field, scales)

    add_faults(field, scales)

    reset_camera()
    ctrl.view_update()
    ctrl.default_view()

ctrl.load_file = load_file

def prepare_slices(dataset):
    state.i_slice = [1, state.dimens[0]]
    state.j_slice = [1, state.dimens[1]]
    state.k_slice = [1, state.dimens[2]]

    vtk_array_i = dsa.numpyTovtkDataArray(FIELD['c_data']["I"])
    vtk_array_j = dsa.numpyTovtkDataArray(FIELD['c_data']["J"])
    vtk_array_k = dsa.numpyTovtkDataArray(FIELD['c_data']["K"])

    dataset.GetCellData().SetScalars(vtk_array_i)
    dataset.GetCellData().SetScalars(vtk_array_j)
    dataset.GetCellData().SetScalars(vtk_array_k)

    update_field_slices_params(state.activeField)

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
    state.active_cells = int(np.sum(field.grid.actnum))

    soil = field.states.SOIL[0] if 'SOIL' in field.states else 0
    swat = field.states.SWAT[0] if 'SWAT' in field.states else 0
    sgas = field.states.SGAS[0] if 'SGAS' in field.states else 0

    c_vols = field.grid.cell_volumes

    p_vols = field.grid.actnum * field.rock.poro * c_vols
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

    FIELD['dates'] = field.result_dates

    state.stateDate = FIELD['dates'][0].strftime('%Y-%m-%d')
    state.startDate = FIELD['dates'][0].strftime('%Y-%m-%d')
    state.lastDate = FIELD['dates'][-1].strftime('%Y-%m-%d')

def add_scalars(field, scales):
    actor = vtkActor()
    actor.SetScale(*scales)

    vtk_array = dsa.numpyTovtkDataArray(FIELD['c_data'][state.activeField])

    dataset = FIELD['dataset']
    dataset.GetCellData().SetScalars(vtk_array)

    mapper = vtkDataSetMapper()
    mapper.SetInputData(dataset)
    mapper.SetScalarRange(dataset.GetScalarRange())
    actor.SetMapper(mapper)

    for name in ['actor', 'wells_actor', 'actor_faults', 'well_labels_actor',
                 'faults_links_actor', 'faults_label_actor']:
        if name in FIELD:
             renderer.RemoveActor(FIELD[name])

    renderer.AddActor(actor)
    FIELD['actor'] = actor

def add_wells(field, scales):
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    grid = field.grid
    z = grid.xyz[grid.actnum][..., 2]
    z_min = z.min()
    dz = z.max() - z_min
    z_min = z_min - 0.1*dz

    field.wells._get_first_entering_point()

    n_wells = len(field.wells.names)
    labeled_points = vtk.vtkPoints()
    labels = vtk.vtkStringArray()
    labels.SetNumberOfValues(n_wells)
    labels.SetName("labels")

    for i, well in enumerate(field.wells):
        labels.SetValue(i, well.name)

        wtrack_idx, first_intersection = well._first_entering_point
        welltrack = well.welltrack[:, :3]

        if first_intersection is not None:
            welltrack_tmp = np.concatenate([np.array([[first_intersection[0], first_intersection[1], z_min]]),
                                        np.asarray(first_intersection).reshape(1, -1),
                                        well.welltrack[wtrack_idx + 1:, :3]])
        else:
            welltrack_tmp = np.concatenate([np.array([[welltrack[0, 0], welltrack[0, 1], z_min]]), welltrack[:]])

        point_ids = []
        labeled_points.InsertNextPoint(welltrack_tmp[0, :3]*scales)
        for line in welltrack_tmp:
            point_ids.append(points.InsertNextPoint(line[:3]))

        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(len(point_ids))
        for i, id in enumerate(point_ids):
            polyLine.GetPointIds().SetId(i, id)
        cells.InsertNextCell(polyLine)

    label_PolyData = vtk.vtkPolyData()
    label_PolyData.SetPoints(labeled_points)
    label_PolyData.GetPointData().AddArray(labels)
    label_mapper = vtk.vtkLabeledDataMapper()
    label_mapper.SetInputData(label_PolyData)
    label_mapper.SetFieldDataName('labels')
    label_mapper.SetLabelModeToLabelFieldData()
    label_actor = vtk.vtkActor2D()
    label_actor.SetMapper(label_mapper)

    renderer.AddActor(label_actor)
    FIELD['well_labels_actor'] = label_actor

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)
    wells_actor = vtk.vtkActor()
    wells_actor.SetScale(*scales)
    wells_actor.SetMapper(mapper)
    wells_actor.GetProperty().SetLineWidth(3)

    colors = vtk.vtkNamedColors()

    (wells_actor.GetProperty()
        .SetColor(colors
            .GetColor3d('White' if state.theme == 'dark' else 'Black')))

    renderer.AddActor(wells_actor)
    FIELD['wells_actor'] = wells_actor

def add_faults(field, scales):
    field.faults.get_blocks()
    n_segments = len(field.faults.names)

    labels = vtk.vtkStringArray()
    labels.SetNumberOfValues(n_segments)
    labels.SetName("labels")

    grid = field.grid
    z = grid.xyz[grid.actnum][..., 2]
    z_min = z.min()
    dz = z.max() - z_min
    z_min = z_min - 0.05*dz

    points = vtk.vtkPoints()
    polygons = vtk.vtkCellArray()

    labeled_points = vtk.vtkPoints()
    links_points = vtk.vtkPoints()
    links_points_ids = []
    link_cells = vtk.vtkCellArray()

    size = 0
    for i, segment in enumerate(field.faults):
        blocks = segment.blocks
        xyz = segment.faces_verts
        active = field.grid.actnum[blocks[:, 0], blocks[:, 1], blocks[:, 2]]
        xyz = xyz[active].reshape(-1, 3)
        if len(xyz) == 0:
            continue

        for p in xyz:
            points.InsertNextPoint(*p)

        labels.SetValue(i, segment.name)
        labeled_points.InsertNextPoint(np.array([*xyz[0, :2], z_min])*scales)
        links_points_ids.append(links_points.InsertNextPoint(np.array([*xyz[0, :2], z_min])))
        links_points_ids.append(links_points.InsertNextPoint(*xyz[0]))

        ids = np.arange(size, size+len(xyz))
        faces1 = np.stack([ids[::4], ids[1::4], ids[3::4]]).T
        faces2 = np.stack([ids[::4], ids[2::4], ids[3::4]]).T
        faces = np.vstack([faces1, faces2])

        for f in faces:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(3)
            for i, id in enumerate(f):
                polygon.GetPointIds().SetId(i, id)
            polygons.InsertNextCell(polygon)

        size += len(xyz)

        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(2)
        for i, id in enumerate(links_points_ids[-2:]):
            polyLine.GetPointIds().SetId(i, id)
        link_cells.InsertNextCell(polyLine)

    colors = vtk.vtkNamedColors()

    link_polyData = vtk.vtkPolyData()
    link_polyData.SetPoints(links_points)
    link_polyData.SetLines(link_cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(link_polyData)
    fault_links_actor = vtk.vtkActor()
    fault_links_actor.SetScale(*scales)
    fault_links_actor.SetMapper(mapper)
    (fault_links_actor.GetProperty().SetColor(colors.GetColor3d('Red')))

    FIELD['faults_links_actor'] = fault_links_actor
    renderer.AddActor(fault_links_actor)

    label_PolyData = vtk.vtkPolyData()
    label_PolyData.SetPoints(labeled_points)
    label_PolyData.GetPointData().AddArray(labels)
    label_mapper = vtk.vtkLabeledDataMapper()
    label_mapper.SetInputData(label_PolyData)
    label_mapper.SetFieldDataName('labels')
    label_mapper.SetLabelModeToLabelFieldData()
    label_actor = vtk.vtkActor2D()
    label_actor.SetMapper(label_mapper)
    label_actor.GetProperty().SetColor(colors.GetColor3d('Red'))

    FIELD['faults_label_actor'] = label_actor
    renderer.AddActor(label_actor)

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(polygons)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polygonPolyData)

    actor_faults = vtk.vtkActor()
    actor_faults.SetScale(*scales)
    actor_faults.SetMapper(mapper)
    actor_faults.GetProperty().SetColor(colors.GetColor3d('Red'))

    renderer.AddActor(actor_faults)
    FIELD['actor_faults'] = actor_faults

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

def on_keydown(key_code):
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
            # do smth to start loading
            return
        if state.pathIndex is None:
            state.pathIndex = 0
        if state.pathIndex < len(state.dirList):
            path = state.dirList[state.pathIndex]
            if os.path.isdir(path):
                path += os.sep
            state.user_request = path
            state.pathIndex = None
    state.updateDirList = True


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
                        autofocus=True,
                        keydown=(on_keydown, "[$event.code]"),
                        __events=["keydown"]):
                        with vuetify.Template(v_slot_append=True,
                            properties=[("v_slot_append", "v-slot:append")],):
                            with vuetify.VBtn('Load', click='loading = true'):
                                vuetify.VTooltip(
                                    text='Start reading data',
                                    activator="parent",
                                    location="top")
                        with vuetify.Template(v_slot_prepend=True,
                            properties=[("v_slot_prepend", "v-slot:prepend")],):
                            with vuetify.VBtn(icon=True,
                                click='showHistory = !showHistory',
                                flat=True,
                                active=('showHistory',),
                                style="background-color:transparent;\
                                       backface-visibility:visible;"):
                                vuetify.VIcon("mdi-history")
                                vuetify.VTooltip(
                                    text='Show recent files',
                                    activator="parent",
                                    location="top")
                        with vuetify.Template(v_slot_loader=True,
                            properties=[("v_slot_loader", "v-slot:loader")],):
                            with vuetify.VCard(
                                v_if='(!loading & !loadComplete) | showHistory',
                                classes="overflow-auto",
                                max_width="100%",
                                max_height="30vh"):
                                with vuetify.VList(v_if='!loading & !showHistory'):
                                    with vuetify.VListItem(
                                        v_for="item, index in dirList",
                                        click="user_request = item"):
                                        vuetify.VListItemTitle("{{item}}")
                                with vuetify.VList(v_if='showHistory'):
                                    with vuetify.VListItem(
                                        v_for="item, index in recentFiles",
                                        click="user_request = item"):
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
                    with vuetify.VCard(v_if='loadComplete & !showHistory & !loading & !loadFailed',
                        variant='text'):
                        vuetify.VIcon('mdi-check-bold', color="success")
                        vuetify.VCardText('Loading completed')
                    with vuetify.VCard(v_if='loadComplete & !showHistory & !loading & loadFailed',
                        variant='text'):
                        vuetify.VIcon('mdi-close-thick', color="error")
                        vuetify.VCardText('Loading failed: ' + '{{errMessage}}')
                    with vuetify.VCard(v_if='showHistory & emptyHistory', variant='text'):
                        vuetify.VCardText('History is empty')