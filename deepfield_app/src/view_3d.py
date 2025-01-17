"3D view page."
import numpy as np
import pandas as pd
from matplotlib.pyplot import get_cmap
import vtk

from trame.widgets import html, vtk as vtk_widgets, vuetify3 as vuetify

from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderWindowInteractor

from .config import dataset_names, state, ctrl, FIELD, renderer, actor_names

VTK_VIEW_SETTINGS = {
    "interactive_ratio": 1,
    "interactive_quality": 90,
}

state.colormaps = sorted(["cividis", "inferno", "jet",
    "hot", "hsv", "magma", "plasma", "rainbow",
    "Spectral", 'turbo', "twilight", "viridis",
    'YlGn', 'YlGnBu', 'RdGy', 'RdYlBu', 'BuGn',
    "gray", 'Blues', 'Greens', 'Oranges', 'Reds'], key=str.casefold)
state.colormap = 'jet'

render_window = vtkRenderWindow()
render_window.AddRenderer(renderer)

rw_interactor = vtkRenderWindowInteractor()
rw_interactor.SetRenderWindow(render_window)
rw_interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

scalarWidget = vtk.vtkScalarBarWidget()
scalarWidget.SetInteractor(rw_interactor)
scalarBar = scalarWidget.GetScalarBarActor()
scalarBar.UnconstrainedFontSizeOn()
scalarBar.GetLabelTextProperty().BoldOff()
scalarBar.GetLabelTextProperty().ItalicOff()
scalarBar.GetTitleTextProperty().BoldOff()
scalarBar.GetTitleTextProperty().ItalicOff()
scalarBar.GetLabelTextProperty().SetFontSize(14)
scalarBar.SetVerticalTitleSeparation(2)
scalarBar.SetBarRatio(scalarBar.GetBarRatio() * 1.5)
scalarBar.SetMaximumWidthInPixels(50)


@state.change("theme")
def change_vtk_bgr(theme, **kwargs):
    "Change background in vtk."
    _ = kwargs
    if theme == 'light':
        renderer.SetBackground(1, 1, 1)
        scalarBar.GetLabelTextProperty().SetColor(0, 0, 0)
        scalarBar.GetTitleTextProperty().SetColor(0, 0, 0)
    else:
        renderer.SetBackground(0, 0, 0)
        scalarBar.GetLabelTextProperty().SetColor(1, 1, 1)
        scalarBar.GetTitleTextProperty().SetColor(1, 1, 1)
    ctrl.view_update()

@state.change("activeField", "activeStep", "modelID")
def update_field(activeField, activeStep, **kwargs):
    "Update field in vtk."
    _ = kwargs

    if (activeField is None) or (FIELD['c_data'] is None):
        return

    activeStep = int(activeStep) if activeStep else 0
    if activeField.split("_")[0].lower() == 'states':
        data = FIELD['c_data'][activeField]
        if data.ndim == 2:
            data = data[:, activeStep]
        state.need_time_slider = True
        vtk_array = dsa.numpyTovtkDataArray(data)
    else:
        state.need_time_slider = False
        vtk_array = dsa.numpyTovtkDataArray(FIELD['c_data'][activeField])
    state.stateDate = FIELD['dates'][activeStep].strftime('%Y-%m-%d')
    dataset = FIELD['dataset']
    dataset.GetCellData().SetScalars(vtk_array)

    mapper = FIELD[actor_names.main].GetMapper()
    mapper.SetScalarRange(dataset.GetScalarRange())
    FIELD[actor_names.main].SetMapper(mapper)
    scalarBar.SetTitle(activeField.split('_')[1])

    update_wells_status(activeStep)

    update_field_slices_params(activeField)
    update_threshold_slices(state.i_slice, state.j_slice, state.k_slice, state.field_slice)

def update_wells_status(activeStep):
    "Get wells status."
    active_step = int(activeStep)
    named_colors = vtk.vtkNamedColors()
    field = FIELD['model']

    well_colors = vtk.vtkUnsignedCharArray()
    well_colors.SetNumberOfComponents(3)
    for well in field.wells:
        if 'RESULTS' in well.attributes:
            for col in ('WOPR', 'WWPR', 'WGPR'):
                if col in well.results.columns and well.results.loc[active_step, col] > 0:
                    well_colors.InsertNextTypedTuple(named_colors.GetColor3ub("Green"))
                    break
            else:
                if 'WWIR' in well.results.columns and well.results.loc[active_step, 'WWIR'] > 0:
                    well_colors.InsertNextTypedTuple(named_colors.GetColor3ub("Blue"))
                    continue
                well_colors.InsertNextTypedTuple(named_colors.GetColor3ub("RED"))
        else:
            well_colors.InsertNextTypedTuple(named_colors.GetColor3ub("RED"))

    FIELD[dataset_names.wells].GetCellData().SetScalars(well_colors)

@state.change('stateDate')
def update_date(stateDate, **kwargs):
    "Synchronize date and activeStep."
    _ = kwargs
    if stateDate is None:
        return
    if 'dates' in FIELD:
        stateDate = pd.to_datetime(stateDate)
        if FIELD['dates'][-1] < stateDate:
            i = state.max_timestep
        else:
            i = np.argmax(FIELD['dates'] >= stateDate)
        state.activeStep = int(i)

@state.change("colormap")
def update_cmap(colormap, **kwargs):
    "Update colormap."
    _ = kwargs
    if state.showScalars:
        cmap = get_cmap(colormap)
        table = FIELD[actor_names.main].GetMapper().GetLookupTable()
        colors = cmap(np.arange(0, cmap.N))
        table.SetNumberOfTableValues(len(colors))
        for i, val in enumerate(colors):
            table.SetTableValue(i, val[0], val[1], val[2])
        table.Build()
        scalarWidget.GetScalarBarActor().SetLookupTable(table)
        scalarWidget.On()
    else:
        FIELD[actor_names.main].GetMapper().ScalarVisibilityOff()
    ctrl.view_update()

def make_threshold(slices, attr, input_threshold=None, ijk=False, component=None):
    "Set threshold filter limits."
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(FIELD['dataset'])
    if input_threshold:
        threshold.SetInputConnection(input_threshold.GetOutputPort())
    if ijk:
        slices = [int(slices[0]), int(slices[1])]
        if slices[0] == slices[1]:
            threshold.SetUpperThreshold(slices[1]-0.5)
            threshold.SetLowerThreshold(slices[0]-1)
        else:
            threshold.SetUpperThreshold(slices[1]-1)
            threshold.SetLowerThreshold(slices[0]-1)
    else:
        slices = [float(slices[0]), float(slices[1])]
        threshold.SetUpperThreshold(slices[1])
        threshold.SetLowerThreshold(slices[0])
    threshold.SetInputArrayToProcess(0, 0, 0, 1, attr)
    if component:
        threshold.SetSelectedComponent(component)
    return threshold

@state.change("i_slice_0", "i_slice_1")
def update_i_slice(i_slice_0, i_slice_1, **kwargs):
    "Update slice ranges."
    _ = kwargs
    if (not i_slice_0) or (not i_slice_1):
        return
    state.i_slice = [i_slice_0, i_slice_1]

@state.change("j_slice_0", "j_slice_1")
def update_j_slice(j_slice_0, j_slice_1, **kwargs):
    "Update slice ranges."
    _ = kwargs
    if (not j_slice_0) or (not j_slice_1):
        return
    state.j_slice = [j_slice_0, j_slice_1]

@state.change("k_slice_0", "k_slice_1")
def update_k_slice(k_slice_0, k_slice_1, **kwargs):
    "Update slice ranges."
    _ = kwargs
    if (not k_slice_0) or (not k_slice_1):
        return
    state.k_slice = [k_slice_0, k_slice_1]

@state.change("field_slice_0", "field_slice_1")
def update_field_slice(field_slice_0, field_slice_1, **kwargs):
    "Update slice ranges."
    _ = kwargs
    if (not field_slice_0) or (not field_slice_1):
        return
    state.field_slice = [field_slice_0, field_slice_1]

@state.change("i_slice", "j_slice", "k_slice", "field_slice")
def update_threshold_slices(i_slice, j_slice, k_slice, field_slice, **kwargs):
    "Filter scalars based on index and values."
    _ = kwargs
    if not FIELD['c_data']:
        return

    state.i_slice_0, state.i_slice_1 = state.i_slice
    state.j_slice_0, state.j_slice_1 = state.j_slice
    state.k_slice_0, state.k_slice_1 = state.k_slice
    state.field_slice_0, state.field_slice_1 = state.field_slice

    threshold_i = make_threshold(i_slice, "I", ijk=True)
    threshold_j = make_threshold(j_slice, "J", input_threshold=threshold_i, ijk=True)
    threshold_k = make_threshold(k_slice, "K", input_threshold=threshold_j, ijk=True)
    threshold_field = make_threshold(field_slice, state.activeField,
        input_threshold=threshold_k,
        component=int(state.activeStep) if state.activeStep else 0)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(threshold_field.GetOutputPort())
    mapper.SetScalarRange(FIELD['dataset'].GetScalarRange())
    FIELD[actor_names.main].SetMapper(mapper)
    update_cmap(state.colormap)

def update_field_slices_params(activeField):
    "Init filter limits."
    if activeField is None:
        return
    state.field_slice_min = float(FIELD['c_data'][activeField].min())
    state.field_slice_max = float(FIELD['c_data'][activeField].max())
    state.field_slice_0 = state.field_slice_min
    state.field_slice_1 = state.field_slice_max
    state.field_slice = [state.field_slice_min, state.field_slice_max]
    state.field_slice_step = (state.field_slice_max - state.field_slice_min) / state.n_field_steps

@state.change("opacity")
def update_opacity(opacity, **kwargs):
    "Update opacity."
    _ = kwargs
    if opacity is None:
        return
    FIELD[actor_names.main].GetProperty().SetOpacity(opacity)
    ctrl.view_update()

@state.change("showScalars")
def change_field_visibility(showScalars, **kwargs):
    "Set visibility of scalars."
    _ = kwargs
    if showScalars is None:
        return
    if showScalars:
        FIELD[actor_names.main].GetProperty().SetRepresentationToSurface()
        FIELD[actor_names.main].SetVisibility(True)
        FIELD[actor_names.main].GetMapper().ScalarVisibilityOn()
        scalarBar.SetVisibility(True)
    else:
        if state.showWireframe:
            FIELD[actor_names.main].GetProperty().SetRepresentationToWireframe()
            FIELD[actor_names.main].GetMapper().ScalarVisibilityOff()
            scalarBar.SetVisibility(False)
        else:
            FIELD[actor_names.main].SetVisibility(False)
            scalarBar.SetVisibility(False)
    ctrl.view_update()

@state.change("showWireframe")
def change_wireframe_visibility(showWireframe, **kwargs):
    "Set visibility of wireframe."
    _ = kwargs
    if showWireframe is None:
        return
    if state.showScalars:
        return
    if showWireframe:
        FIELD[actor_names.main].GetProperty().SetRepresentationToWireframe()
        FIELD[actor_names.main].GetMapper().ScalarVisibilityOff()
        FIELD[actor_names.main].SetVisibility(True)
    else:
        FIELD[actor_names.main].SetVisibility(False)
    ctrl.view_update()

@state.change("showWells")
def change_wells_visibility(showWells, **kwargs):
    "Set visibility of wells."
    _ = kwargs
    for name in (actor_names.wells, actor_names.well_labels):
        if name in FIELD:
            FIELD[name].SetVisibility(showWells)
    ctrl.view_update()

@state.change("showFaults")
def change_faults_visibility(showFaults, **kwargs):
    "Set visibility of faults."
    _ = kwargs
    for name in [actor_names.faults, actor_names.fault_links, actor_names.fault_labels]:
        if name in FIELD:
            FIELD[name].SetVisibility(showFaults)
    ctrl.view_update()

def default_view():
    "Reset 3d view setting to initial state."
    state.i_slice = [1, state.dimens[0]]
    state.j_slice = [1, state.dimens[1]]
    state.k_slice = [1, state.dimens[2]]
    state.field_slice = [state.field_slice_min, state.field_slice_max]
    state.opacity = 1
    state.showScalars = True
    state.showWireframe = True
    state.showWells = True
    state.showFaults = True
    state.activeStep = 0
    ctrl.view_reset_camera()

ctrl.default_view = default_view

def render_3d():
    "3D view layout."
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="height: 100%; width: 100%", classes='pa-0 ma-0'):
            with vuetify.VCol(classes="pa-0"):
                view = vtk_widgets.VtkRemoteView(
                    render_window,
                    **VTK_VIEW_SETTINGS
                    )
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    with html.Div(
        style='position: fixed; width: 100%; bottom: 0; padding-left: 10vw; padding-right: 10vw;'):
        with vuetify.VTextField(
              v_model=("stateDate",),
              label="Select a date",
              hide_details=True,
              density='compact',
              type="date"):
            with vuetify.Template(v_slot_append=True,
                properties=[("v_slot_append", "v-slot:append")],):
                with vuetify.VSlider(
                    min=0,
                    max=("max_timestep",),
                    step=1,
                    v_model=('activeStep',),
                    label="Timestep",
                    hide_details=True,
                    style='width: 60vw'
                    ):
                    with vuetify.Template(v_slot_append=True,
                        properties=[("v_slot_append", "v-slot:append")],):
                        vuetify.VTextField(
                            v_model="activeStep",
                            density="compact",
                            style="width: 80px",
                            type="number",
                            variant="outlined",
                            bg_color=('bgColor',),
                            hide_details=True)

    with vuetify.VCard(
        color=('sideBarColor',),
        flat=True,
        style='position: fixed; left: 0; top: calc(50% + 48px); transform: translateY(calc(0px - 50% - 24px));'):
        with vuetify.VContainer(fluid=True,
            style='align-items: start; justify-content: left;',
            classes='pa-0 ma-0'):
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VTooltip(
                            text='Select field to show',
                            activator="parent",
                            location="end")
                        vuetify.VIcon("mdi-database-export-outline")
                        with vuetify.VMenu(activator="parent",
                            location="right",
                            close_on_content_click=False):
                            with vuetify.VCard(classes="overflow-auto", max_height="50vh"):
                                with vuetify.VList():
                                    with vuetify.VListItem(
                                        v_for="item, index in field_attrs",
                                        active=("item === activeField",),
                                        click="activeField = item"):
                                        vuetify.VListItemTitle("{{item}}")
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VTooltip(
                            text='Change colormap',
                            activator="parent",
                            location="end")
                        vuetify.VIcon("mdi-format-color-fill")
                        with vuetify.VMenu(activator="parent",
                            location="right",
                            close_on_content_click=False):
                            with vuetify.VCard(classes="overflow-auto", max_height="50vh"):
                                with vuetify.VList():
                                    with vuetify.VListItem(
                                        v_for="(item, index) in colormaps",
                                        click="colormap = item",
                                        active=("item === colormap",)
                                        ):
                                        vuetify.VListItemTitle("{{item}}")
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VTooltip(
                            text='Set opacity',
                            activator="parent",
                            location="end")
                        vuetify.VIcon("mdi-circle-opacity")
                        with vuetify.VMenu(activator="parent",
                            location="right",
                            close_on_content_click=False):
                            with html.Div(style='width: 20vw'):
                                with vuetify.VSlider(
                                    min=0,
                                    max=1,
                                    step=0.1,
                                    v_model=('opacity', 1),
                                    hide_details=True,
                                    ):
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="opacity",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VTooltip(
                            text='Filter grid based on cell index I',
                            activator="parent",
                            location="end")
                        vuetify.VIcon("mdi-alpha-i")
                        with vuetify.VMenu(activator="parent",
                            location="right",
                            close_on_content_click=False):
                            with html.Div(style='width: 30vw'):
                                with vuetify.VRangeSlider(
                                    min=1,
                                    max=("dimens[0]",),
                                    step=1,
                                    v_model=("i_slice",),
                                    hide_details=True
                                    ):
                                    with vuetify.Template(v_slot_prepend=True,
                                        properties=[("v_slot_prepend", "v-slot:prepend")],):
                                        vuetify.VTextField(
                                            v_model="i_slice_0",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="i_slice_1",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VTooltip(
                            text='Filter grid based on cell index J',
                            activator="parent",
                            location="end")
                        vuetify.VIcon("mdi-alpha-j")
                        with vuetify.VMenu(activator="parent",
                            location="right",
                            close_on_content_click=False):
                            with html.Div(style='width: 30vw'):
                                with vuetify.VRangeSlider(
                                    min=1,
                                    max=("dimens[1]",),
                                    step=1,
                                    v_model=("j_slice",),
                                    hide_details=True
                                    ):
                                    with vuetify.Template(v_slot_prepend=True,
                                        properties=[("v_slot_prepend", "v-slot:prepend")],):
                                        vuetify.VTextField(
                                            v_model="j_slice_0",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="j_slice_1",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VTooltip(
                            text='Filter grid based on cell index K',
                            activator="parent",
                            location="end")
                        vuetify.VIcon("mdi-alpha-k")
                        with vuetify.VMenu(activator="parent",
                            location="right",
                            close_on_content_click=False):
                            with html.Div(style='width: 30vw'):
                                with vuetify.VRangeSlider(
                                    min=1,
                                    max=("dimens[2]",),
                                    step=1,
                                    v_model=("k_slice",),
                                    hide_details=True
                                    ):
                                    with vuetify.Template(v_slot_prepend=True,
                                        properties=[("v_slot_prepend", "v-slot:prepend")],):
                                        vuetify.VTextField(
                                            v_model="k_slice_0",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="k_slice_1",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VTooltip(
                            text='Filter grid based of scalar values',
                            activator="parent",
                            location="end")
                        vuetify.VIcon("mdi-filter")
                        with vuetify.VMenu(activator="parent",
                            location="right",
                            close_on_content_click=False):
                            with html.Div(style='width: 30vw'):
                                with vuetify.VRangeSlider(
                                    min=("field_slice_min",),
                                    max=("field_slice_max",),
                                    step=("field_slice_step",),
                                    v_model=("field_slice",),
                                    hide_details=True
                                    ):
                                    with vuetify.Template(v_slot_prepend=True,
                                        properties=[("v_slot_prepend", "v-slot:prepend")],):
                                        vuetify.VTextField(
                                            v_model="field_slice_0",
                                            density="compact",
                                            style="width: 80px;",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="field_slice_1",
                                            density="compact",
                                            style="width: 80px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VTooltip(
                            text='Change visibility of the objects',
                            activator="parent",
                            location="end")
                        vuetify.VIcon("mdi-layers-outline")
                        with vuetify.VMenu(activator="parent",
                            location="right",
                            close_on_content_click=False):
                            with vuetify.VCard(classes="pr-2"):
                                vuetify.VCheckbox(label='Scalars',
                                    v_model=('showScalars', True),
                                    hide_details=True,
                                    density='compact')
                                vuetify.VCheckbox(label='Wireframe',
                                    v_model=('showWireframe', True),
                                    hide_details=True,
                                    density='compact')
                                vuetify.VCheckbox(label='Wells',
                                    v_model=('showWells', True),
                                    hide_details=True,
                                    density='compact')
                                vuetify.VCheckbox(label='Faults',
                                    v_model=('showFaults', True),
                                    hide_details=True,
                                    density='compact')
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;",
                        click=ctrl.default_view):
                        vuetify.VTooltip(
                            text='Reset view settings to default values',
                            activator="parent",
                            location="end")
                        vuetify.VIcon("mdi-fit-to-page-outline")

ctrl.on_server_ready.add(ctrl.view_update)
