import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from trame.widgets import html, trame, vuetify3 as vuetify, matplotlib

from .config import state, ctrl, FIELD

state.activeSlice = 'k'


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

def create_slice(component, att, i, j, k, t, i_line, j_line, k_line,
                 colormap, figure_size, vmin, vmax):
    fig, ax = plt.subplots(**figure_size)
    component.show_slice(attr=att, i=i, j=j, k=k, t=t,
                         i_line=i_line, j_line=j_line, k_line=k_line,
                         ax=ax, cmap=colormap, vmax=vmax, vmin=vmin)
    fig.tight_layout()
    return fig

@state.change("figure_size", "activeSlice",
              "activeField", "activeStep",
              "xslice", "yslice", "zslice", "colormap")
def update_slices(figure_size, activeSlice,
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
    if activeSlice == 'i':
    	fig = create_slice(component, attr,
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
            vmax=vmax)

    elif activeSlice == 'j':
    	fig = create_slice(component, attr,
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
            vmax=vmax)
    
    elif activeSlice == 'k':
        fig = create_slice(component, attr,
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
            vmax=vmax)
    else:
    	return

    ctrl.update_slice(fig)

def render_2d():
    with vuetify.VContainer(fluid=True, style='align-items: start', classes="fill-height pa-0 ma-0"):
        with vuetify.VRow(style="width:100%; height: 95%; margin 0;", classes='pa-0'):
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
                            v_model="xslice",
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
                    figure = matplotlib.Figure(plt.figure(**get_figure_size(state['figure_size'])),
                        style="position: absolute")
                    ctrl.update_slice = figure.update

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

    with vuetify.VCard(color="grey-lighten-4", flat=True,
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
