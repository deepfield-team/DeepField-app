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


@state.change("theme")
def change_vtk_bgr(theme, **kwargs):
    if theme == 'light':
        renderer.SetBackground(1, 1, 1)
    else:
        renderer.SetBackground(0, 0, 0)
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
    #scalarBar = vtk.vtkScalarBarActor()
    #scalarBar.SetOrientationToVertical()
    #scalarBar.SetLookupTable(table)
    #renderer.AddActor2D(scalarBar)
    ctrl.view_update()

def make_threshold(slices, attr, input_threshold = None):
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(FIELD['dataset'])
    if input_threshold:
        threshold.SetInputConnection(input_threshold.GetOutputPort())       
    threshold.SetUpperThreshold(slices[1])
    threshold.SetLowerThreshold(slices[0])
    threshold.SetInputArrayToProcess(0, 0, 0, 1, attr)
    return threshold

@state.change("i_slice", "j_slice", "k_slice", "field_slice")
def update_threshold_slices(i_slice, j_slice, k_slice, field_slice, **kwargs):
    _ = kwargs
    if not FIELD['c_data']:
        return

    threshold_i = make_threshold(i_slice, "I")
    threshold_j = make_threshold(j_slice, "J", input_threshold=threshold_i)
    threshold_k = make_threshold(k_slice, "K", input_threshold=threshold_j)
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
        state.field_slice_min = FIELD['c_data'][activeField].min()
        state.field_slice_max = FIELD['c_data'][activeField].max()
        state.field_slice = [state.field_slice_min, state.field_slice_max]
        state.field_slice_step = state.field_slice_max / state.n_field_steps

@state.change("opacity")
def update_opacity(opacity, **kwargs):
    _ = kwargs
    if opacity is None:
        return
    FIELD['actor'].GetProperty().SetOpacity(opacity)
    ctrl.view_update()


def fit_view():
    state.i_slice = [1, state.dimens[0]]
    state.j_slice = [1, state.dimens[1]]
    state.k_slice = [1, state.dimens[2]]
    update_field_slices_params(state.activeField)
    state.opacity = 1
    ctrl.view_reset_camera()

ctrl.fit_view = fit_view

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
            hide_details=True
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

    with vuetify.VCard(
        color=('sideBarColor',),
        flat=True,
        style='position: fixed; left: 0; top: 20vh;'):
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
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="i_slice[1]",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
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
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="j_slice[1]",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
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
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="k_slice[1]",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            hide_details=True)
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VIcon("mdi-pac-man")
                        with vuetify.VMenu(activator="parent",
                            location="right",
                            close_on_content_click=False):
                            with html.Div(style='width: 30vw'):
                                with vuetify.VRangeSlider(
                                    min=("field_slice_min",),
                                    max=("field_slice_max",),
                                    step="field_slice_step",
                                    v_model=("field_slice",),
                                    hide_details=True
                                    ):
                                    with vuetify.Template(v_slot_prepend=True,
                                        properties=[("v_slot_prepend", "v-slot:prepend")],):
                                        vuetify.VTextField(
                                            v_model="field_slice[0]",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            hide_details=True)
                                    with vuetify.Template(v_slot_append=True,
                                        properties=[("v_slot_append", "v-slot:append")],):
                                        vuetify.VTextField(
                                            v_model="field_slice[1]",
                                            density="compact",
                                            style="width: 70px",
                                            type="number",
                                            variant="outlined",
                                            hide_details=True)
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,flat=True,
                        style="background-color:transparent;\
                               backface-visibility:visible;",
                        click=ctrl.fit_view):
                        vuetify.VIcon("mdi-fit-to-page-outline")

ctrl.on_server_ready.add(ctrl.view_update)
