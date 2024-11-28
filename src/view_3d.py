import numpy as np
from matplotlib.pyplot import get_cmap
import vtk

from trame.widgets import vtk as vtk_widgets, vuetify3 as vuetify

from vtkmodules.numpy_interface import dataset_adapter as dsa

from .config import state, ctrl, FIELD, render_window, VTK_VIEW_SETTINGS


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

@state.change("opacity")
def update_opacity(opacity, **kwargs):
    _ = kwargs
    if opacity is None:
        return
    FIELD['actor'].GetProperty().SetOpacity(opacity)
    ctrl.view_update()

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
                    v_model=('activeStep',),
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

ctrl.on_server_ready.add(ctrl.view_update)
