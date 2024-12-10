import time
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from deepfield.field.plot_utils import get_slice_trisurf
from deepfield.field import States
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from trame.widgets import html, trame, vuetify3 as vuetify, matplotlib, plotly

from .config import state, ctrl, FIELD


state.activeSlice = 'k'

FIELD['slices'] = {}

PLOT_2D = {'fig': None}

CHART_STYLE_2D = {
    # "display_mode_bar": ("true",),
    'scrollZoom': True,
    "mode_bar_buttons_to_remove": (
        "chart_buttons",
        [   "orbitRotation",
            "resetScale2d",
            "zoomIn2d",
            "zoomOut2d",
            "toggleSpikelines",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "tableRotation",
            "resetCameraDefault3d",
            "resetCameraLastSave3d"
        ],
    ),
    "display_logo": ("false",),
}


def triangle_centroids(x, y, triangles):
    points = np.stack((x, y), axis=1)

    median_start = points[triangles[:, 0]]

    median_end = (points[triangles[:, 1]] + points[triangles[:, 2]])/2

    centroids = median_end*2/3 + median_start*1/3
    return centroids

@state.change("plotlyTheme")
def change_plotly_theme_2d(plotlyTheme, **kwargs):
    fig = PLOT_2D['fig']
    if fig is not None:
        fig.update_layout(template=plotlyTheme)
        ctrl.update_slice(fig)

def get_attr_from_field(attr):
    comp, attr = attr.split('_')
    return FIELD['model']._components[comp.lower()][attr]

def get_figure_size(f_size):
    if f_size is None:
        return {}

    dpi = f_size.get("dpi")
    rect = f_size.get("size")
    w_inch = rect.get("width") / dpi
    h_inch = rect.get("height") / dpi

    return {"figsize": (w_inch, h_inch), "dpi": dpi}

def get_data_limits(component, attr, activeStep):
    data = getattr(component, attr)
    if data.ndim == 4:
        data = data[activeStep]

    data = data[FIELD['model'].grid.actnum]
    vmax = data.max()
    vmin = data.min()
    if vmax == vmin:
        vmax = 1.01 * vmax
        vmin = 0.99 * vmin
    return vmin, vmax


def create_slice(component, att, i, j, k, t, range_x, range_y,
                 xaxis_name, yaxis_name, width,  height, colormap):
    x, y, triangles, data = get_slice_trisurf(component, att, i, j, k, t)
    if triangles is None:
        x = np.zeros(0)
        y = np.zeros(0)
        triangles = np.zeros((0,3))
        data = np.zeros(0)
    z = np.zeros(x.shape)
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            intensity=data,
            customdata=triangle_centroids(x, y, triangles),
            intensitymode='cell',
            colorscale=colormap.lower(),
            hovertemplate=f"{xaxis_name}: %{{customdata[0]:.2f}}<br>{yaxis_name}: %{{customdata[1]:.2f}}<extra></extra>",
            i=triangles[:, 0].ravel(),
            j=triangles[:, 1].ravel(),
            k=triangles[:, 2].ravel(),
            showscale=True,
            flatshading=True,
            colorbar=dict(len=0.75)
            ),
        ],
        layout=go.Layout(
            template=state.plotlyTheme,
            width=width,
            height=height,
            scene={
                'xaxis': dict(visible=False, range=range_x),
                'zaxis': dict(visible=False),
                'yaxis': dict(visible=False, range=range_y),
                'aspectmode':'manual',
                'aspectratio': dict(x=1, y=1, z=0.01),
                'camera': dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=1.5)),
                'dragmode': 'zoom',
            },
            margin={'t': 0, 'r': 110, 'l': 0, 'b': 0},
            )
        )
    PLOT_2D['fig'] = fig
    return fig

@state.change("figure_size", "activeSlice",
              "activeField", "activeStep",
              "xslice", "yslice", "zslice", "colormap")
def update_slices(figure_size, activeSlice,
    activeField, activeStep, xslice, yslice, zslice, colormap, **kwargs):

    figure_size = figure_size
    _ = kwargs
    if activeField is None:
        return
    if figure_size is None:
        return
    activeStep = int(activeStep)
    grid = FIELD['model'].grid
    xyz = grid.xyz
    bounds = figure_size.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 300)
    comp_name, attr = activeField.split('_')
    comp_name = comp_name.lower()
    component = getattr(FIELD['model'], comp_name)
    if activeSlice == 'i':
        range_x = (xyz[grid.actnum][..., 1].min(), xyz[grid.actnum][..., 1].max())
        range_y = (xyz[grid.actnum][..., 2].min(), xyz[grid.actnum][..., 2].max())
        ctrl.update_slice(
            create_slice(component=component,
                        att=attr, i=int(xslice)-1,
                        j=None,
                        k=None,
                        t=int(activeStep) if isinstance(component, States) else None,
                        range_x=range_x,
                        range_y=range_y,
                        xaxis_name='y',
                        yaxis_name='z',
                        width=width,
                        height=height,
                        colormap=colormap))

    if activeSlice == 'j':
        range_x = (xyz[grid.actnum][..., 0].min(), xyz[grid.actnum][..., 0].max())
        range_y = (xyz[grid.actnum][..., 2].min(), xyz[grid.actnum][..., 2].max())
        ctrl.update_slice(
            create_slice(component=component,
                        att=attr,
                        i=None,
                        j=int(yslice)-1,
                        k=None,
                        t=int(activeStep) if isinstance(component, States) else None,
                        range_x=range_x,
                        range_y=range_y,
                        xaxis_name='x',
                        yaxis_name='z',
                        width=width,
                        height=height,
                        colormap=colormap))

    if activeSlice == 'k':
        range_x = (xyz[grid.actnum][..., 0].min(), xyz[grid.actnum][..., 0].max())
        range_y = (xyz[grid.actnum][..., 1].min(), xyz[grid.actnum][..., 1].max())
        ctrl.update_slice(
            create_slice(
                component=component,
                att=attr,
                i=None,
                j=None,
                k=int(zslice)-1,
                t=int(activeStep) if isinstance(component, States) else None,
                range_x=range_x,
                range_y=range_y,
                xaxis_name='x',
                yaxis_name='y',
                width=width,
                height=height,
                colormap=colormap))

def render_2d():
    with vuetify.VContainer(fluid=True,
        style='align-items: start',
        classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="width:100%; height: 95%", classes='pa-0 ma-0'):
            with vuetify.VCol(classes='pa-0'):
                with vuetify.VSlider(
                	v_if="activeSlice === 'i'",
                    min=1,
                    max=("dimens[0]",),
                    step=1,
                    v_model=('xslice', 1),
                    label='Slice I',
                    classes='pt-3 pr-2 pl-2',
                    hide_details=True
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
                with vuetify.VSlider(
                	v_if="activeSlice === 'j'",
                    min=1,
                    max=("dimens[1]",),
                    step=1,
                    v_model=('yslice', 1),
                    label='Slice J',
                    classes='pt-3 pr-2 pl-2',
                    hide_details=True
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
                with vuetify.VSlider(
                	v_if="activeSlice === 'k'",
                    min=1,
                    max=("dimens[2]",),
                    step=1,
                    v_model=('zslice', 1),
                    label='Slice K',
                    classes='pt-3 pr-2 pl-2',
                    hide_details=True
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
                    ctrl.update_slice = plotly.Figure(**CHART_STYLE_2D).update

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
        style='position: fixed; left: 0; top: 30vh;'):
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
                    with vuetify.VBtn(icon=True, flat=True,
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
                    with vuetify.VBtn(icon=True,
                        flat=True,
                        click="activeSlice = 'i'",
                        active=("activeSlice === 'i'",),
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VIcon("mdi-alpha-i")
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,
                        flat=True,
                        click="activeSlice = 'j'",
                        active=("activeSlice === 'j'",),
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VIcon("mdi-alpha-j")
            with vuetify.VRow(classes='pa-0 ma-0'):
                with vuetify.VCol(classes='pa-0 ma-0'):
                    with vuetify.VBtn(icon=True,
                        flat=True,
                        click="activeSlice = 'k'",
                        active=("activeSlice === 'k'",),
                        style="background-color:transparent;\
                               backface-visibility:visible;"):
                        vuetify.VIcon("mdi-alpha-k")
