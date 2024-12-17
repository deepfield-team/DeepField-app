import numpy as np
from matplotlib.pyplot import get_cmap
import vtk

from trame.widgets import html, vtk as vtk_widgets, vuetify3 as vuetify

from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderWindowInteractor

from .config import state, ctrl, FIELD, renderer

VTK_VIEW_SETTINGS = {
    "interactive_ratio": 1,
    "interactive_quality": 90,
}

state.colormaps = ["jet", "gray", "hsv", "Spectral", "twilight", "viridis"]
state.colormap = state.colormaps[0]

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
    if theme == 'light':
        renderer.SetBackground(1, 1, 1)
        scalarBar.GetLabelTextProperty().SetColor(0, 0, 0)
        scalarBar.GetTitleTextProperty().SetColor(0, 0, 0)
    else:
        renderer.SetBackground(0, 0, 0)
        scalarBar.GetLabelTextProperty().SetColor(1, 1, 1)
        scalarBar.GetTitleTextProperty().SetColor(1, 1, 1)
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
    scalarBar.SetTitle(activeField.split('_')[1])

    ctrl.view_update()

@state.change("colormap")
def update_cmap(colormap, **kwargs):
    if state.showScalars:
        cmap = get_cmap(colormap)
        table = FIELD['actor'].GetMapper().GetLookupTable()
        colors = cmap(np.arange(0, cmap.N))
        table.SetNumberOfTableValues(len(colors))
        for i, val in enumerate(colors):
            table.SetTableValue(i, val[0], val[1], val[2])
        table.Build()
        scalarWidget.GetScalarBarActor().SetLookupTable(table)
        scalarWidget.On()
    else:
        FIELD['actor'].GetMapper().ScalarVisibilityOff()
    ctrl.view_update()

def make_threshold(slices, attr, input_threshold=None, ijk=False):
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(FIELD['dataset'])
    if input_threshold:
        threshold.SetInputConnection(input_threshold.GetOutputPort())
    if ijk:
        if slices[0] == slices[1]:
            threshold.SetUpperThreshold(slices[1]-0.5)
            threshold.SetLowerThreshold(slices[0]-1)
        else:
            threshold.SetUpperThreshold(slices[1]-1)
            threshold.SetLowerThreshold(slices[0]-1)
    else:
        threshold.SetUpperThreshold(slices[1])
        threshold.SetLowerThreshold(slices[0])
    threshold.SetInputArrayToProcess(0, 0, 0, 1, attr)
    return threshold

@state.change("i_slice", "j_slice", "k_slice", "field_slice")
def update_threshold_slices(i_slice, j_slice, k_slice, field_slice, **kwargs):
    _ = kwargs
    if not FIELD['c_data']:
        return

    threshold_i = make_threshold(i_slice, "I", ijk=True)
    threshold_j = make_threshold(j_slice, "J", input_threshold=threshold_i, ijk=True)
    threshold_k = make_threshold(k_slice, "K", input_threshold=threshold_j, ijk=True)
    threshold_field = make_threshold(field_slice, state.activeField, input_threshold=threshold_k)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(threshold_field.GetOutputPort())
    mapper.SetScalarRange(FIELD['dataset'].GetScalarRange())
    FIELD['actor'].SetMapper(mapper)
    update_cmap(state.colormap)

@state.change("activeField", "activeStep")
def update_field_slices_params(activeField, **kwargs):
    _ = kwargs
    if activeField:
        state.field_slice_min = float(FIELD['c_data'][activeField].min())
        state.field_slice_max = float(FIELD['c_data'][activeField].max())
        state.field_slice = [state.field_slice_min, state.field_slice_max]
        state.field_slice_step = (state.field_slice_max - state.field_slice_min) / state.n_field_steps

@state.change("opacity")
def update_opacity(opacity, **kwargs):
    _ = kwargs
    if opacity is None:
        return
    FIELD['actor'].GetProperty().SetOpacity(opacity)
    ctrl.view_update()

@state.change("showScalars")
def change_field_visibility(showScalars, **kwargs):
    _ = kwargs
    if showScalars is None:
        return
    if showScalars:
        FIELD['actor'].GetProperty().SetRepresentationToSurface()
        FIELD['actor'].SetVisibility(True)
        FIELD['actor'].GetMapper().ScalarVisibilityOn()
        scalarBar.SetVisibility(True)
    else:
        if state.showWireframe:
            FIELD['actor'].GetProperty().SetRepresentationToWireframe()
            FIELD['actor'].GetMapper().ScalarVisibilityOff()
            scalarBar.SetVisibility(False)
        else:
            FIELD['actor'].SetVisibility(False)
            scalarBar.SetVisibility(False)
    ctrl.view_update()

@state.change("showWireframe")
def change_field_visibility(showWireframe, **kwargs):
    _ = kwargs
    if showWireframe is None:
        return
    if state.showScalars:
        return
    if showWireframe:
        FIELD['actor'].GetProperty().SetRepresentationToWireframe()
        FIELD['actor'].GetMapper().ScalarVisibilityOff()
        FIELD['actor'].SetVisibility(True)
    else:
        FIELD['actor'].SetVisibility(False)
    ctrl.view_update()

@state.change("showWells")
def change_wells_visibility(showWells, **kwargs):
    if 'actor_wells' in FIELD:
        FIELD['actor_wells'].SetVisibility(showWells)
        ctrl.view_update()

@state.change("showFaults")
def change_wells_visibility(showFaults, **kwargs):
    if 'actor_faults' in FIELD:
        FIELD['actor_faults'].SetVisibility(showFaults)
        ctrl.view_update()

def default_view():
    state.i_slice = [1, state.dimens[0]]
    state.j_slice = [1, state.dimens[1]]
    state.k_slice = [1, state.dimens[2]]
    state.field_slice = [state.field_slice_min, state.field_slice_max]
    state.opacity = 1
    state.showScalars = True
    state.showWireframe = True
    state.showWells = True
    state.showFaults = True
    ctrl.view_reset_camera()

ctrl.default_view = default_view

def render_3d():
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="height: 100%; width: 100%", classes='pa-0 ma-0'):
            with vuetify.VCol(classes="pa-0"):
                view = vtk_widgets.VtkRemoteView(
                    render_window,
                    **VTK_VIEW_SETTINGS
                    )
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    with html.Div(v_if='need_time_slider', style='position: fixed; width: 100%; bottom: 0;'):
        with vuetify.VSlider(
            min=0,
            max=("max_timestep",),
            step=1,
            v_model=('activeStep',),
            label="Timestep",
            hide_details=True,
            classes='pr-2 pl-2 pb-1'
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
                                            v_model="i_slice[0]",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="i_slice[1]",
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
                                            v_model="j_slice[0]",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="j_slice[1]",
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
                                            v_model="k_slice[0]",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="k_slice[1]",
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
                                            v_model="field_slice[0]",
                                            density="compact",
                                            style="width: 80px;",
                                            type="number",
                                            variant="outlined",
                                            bg_color=('bgColor',),
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="field_slice[1]",
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
                        vuetify.VIcon("mdi-fit-to-page-outline")

ctrl.on_server_ready.add(ctrl.view_update)
