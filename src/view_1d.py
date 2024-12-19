import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from trame.widgets import trame, plotly, vuetify3 as vuetify

from .config import state, ctrl, FIELD

PLOTS = {"plot1d": None,
         "plot_pvt": None}

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

state.domains = []
state.domainMin = 0
state.domainMax = 0
state.domainStep = None
state.domainToShow = None
state.needDomain = False
state.domainName = None
state.gridData = True
state.wellData = True


@state.change('modelID')
def reset_plots(modelID, **kwargs):
    _ = kwargs
    clean_plot()
    clean_pvt_plot()

@state.change("plotlyTheme")
def change_plotly_theme(plotlyTheme, **kwargs):
    fig = PLOTS['plot1d']
    if fig is not None:
        fig.layout.template = plotlyTheme
        ctrl.update_plot(fig)
    fig = PLOTS['plot_pvt']
    if fig is not None:
        fig.layout.template = plotlyTheme
        ctrl.update_tplot(fig)

@state.change("data1dToShow")
def update1dWidgets(data1dToShow, **kwargs):
    _ = kwargs
    if data1dToShow is None:
        return
    state.gridData = data1dToShow in state.statesAttrs
    state.wellData = data1dToShow in state.wellsAttrs
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
    if PLOTS['plot1d'] is None:
        return
    PLOTS['plot1d'].data = []
    ctrl.update_plot(PLOTS['plot1d'])

ctrl.clean_plot = clean_plot

def remove_last_line():
    if not PLOTS['plot1d'].data:
        return
    PLOTS['plot1d'].data = PLOTS['plot1d'].data[:-1]
    ctrl.update_plot(PLOTS['plot1d'])

ctrl.remove_last_line = remove_last_line

@state.change("figure_size_1d")
def update_plot_size(figure_size_1d, **kwargs):
    _ = kwargs
    if figure_size_1d is None:
        return
    bounds = figure_size_1d.get("size", {})
    width = bounds.get("width", 300)
    height = bounds.get("height", 100)
    if PLOTS['plot1d'] is None:
        PLOTS['plot1d'] = make_subplots(specs=[[{"secondary_y": True}]])
        PLOTS['plot1d'].update_layout(
            showlegend=True,
            margin={'t': 30, 'r': 10, 'l': 80, 'b': 30},
            legend={'x': 1.01,},
            template=state.plotlyTheme,
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
        xaxis=dict(domain=[0, 1.02-0.02*len(table.columns)]),
        **layout)
    fig.update_xaxes(title_text=table.index.name)
    return fig

def plot_table(tableToShow, tableXAxis, domainToShow, height, width):
    fig = go.Figure()
    fig.update_layout(
        height=height,
        width=width,
        showlegend=False,
        template=state.plotlyTheme,
        margin={'t': 30, 'r': 50, 'l': 80, 'b': 30},
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
    PLOTS['plot_pvt'] = plot_table(tableToShow, tableXAxis, domainToShow, height, width)
    ctrl.update_tplot(PLOTS['plot_pvt'])

def clean_pvt_plot():
    if PLOTS['plot_pvt'] is None:
        return
    PLOTS['plot_pvt'].data = []
    ctrl.update_tplot(PLOTS['plot_pvt'])

def render_ts():
    with vuetify.VContainer(fluid=True, style='align-items: top', classes="pa-0 ma-0"):
        with vuetify.VRow(classes='pa-0 ma-0'):
            with vuetify.VCol(classes='pa-0 ma-0'):
                vuetify.VSelect(
                    v_model=("data1dToShow", None),
                    items=("data1d", ),
                    label="Select data"
                    )
            with vuetify.VCol(classes='pa-0 ma-0'):
                vuetify.VSelect(
                    disabled=("wellData",),
                    v_model=("i_cell", 'Average'),
                    items=("i_cells", ),
                    label="I index"
                    )
            with vuetify.VCol(classes='pa-0 ma-0'):
                vuetify.VSelect(
                    disabled=("wellData",),
                    v_model=("j_cell", 'Average'),
                    items=("j_cells", ),
                    label="J index"
                    )
            with vuetify.VCol(classes='pa-0 ma-0'):
                vuetify.VSelect(
                    disabled=("wellData",),
                    v_model=("k_cell", 'Average'),
                    items=("k_cells", ),
                    label="K index"
                    )
            with vuetify.VCol(classes='pa-0 ma-0'):
                vuetify.VSelect(
                    disabled=("gridData",),
                    v_model=("wellNameToShow", None),
                    items=("wellnames", ),
                    label="Select well"
                    )
            with vuetify.VCol(classes='pa-0 ma-0'):
                vuetify.VSwitch(
                    v_model=("secondAxis", False),
                    color="primary",
                    label="Second Axis",
                    hide_details=True)
            with vuetify.VCol(classes='pa-0 ma-0', style='flex-grow: 0'):
                vuetify.VBtn('Add line', click=ctrl.add_line_to_plot, classes='mt-2')
            with vuetify.VCol(classes='pa-0 mt-0', style='flex-grow: 0'):
                vuetify.VBtn('Undo', click=ctrl.remove_last_line, classes='mt-2')
            with vuetify.VCol(classes='pa-0 ma-0', style='flex-grow: 0'):
                vuetify.VBtn('Clean', click=ctrl.clean_plot, classes='mt-2')

        with vuetify.VRow(style="width: 100%; height: 75vh", classes='pa-0 ma-0'):
            with vuetify.VCol(classes='pa-0'):
                with trame.SizeObserver("figure_size_1d"):
                    ctrl.update_plot = plotly.Figure(**CHART_STYLE).update
                    update_plot_size(state.figure_size_1d)

def render_pvt():
    with vuetify.VContainer(fluid=True, style='align-items: top', classes="pa-0 ma-0"):
        with vuetify.VRow(classes='pa-0 ma-0'):
            with vuetify.VCol(classes='pa-0 ma-0'):
                vuetify.VSelect(
                    v_model=("tableToShow", None),
                    items=("tables", ),
                    label="Select data"
                    )
            with vuetify.VCol(classes='pa-0 ma-0'):
                vuetify.VSelect(
                    v_model=("tableXAxis", None),
                    items=("domains", ),
                    label="Select x-axis",
                    disabled=("!needDomain",),
                    )
            with vuetify.VCol(classes='pa-0 ma-0'):
                with vuetify.VSlider(
                    label=('domainName',),
                    disabled=("!needDomain",),
                    v_model=("domainToShow", ),
                    min=("domainMin",),
                    max=("domainMax",),
                    step=("domainStep",),
                    hide_details=False,
                    classes='mt-2',
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
        with vuetify.VRow(style="width: 100%; height: 75vh", classes='pa-0 ma-0'):
            with vuetify.VCol(classes='pa-0'):
                with trame.SizeObserver("figure_size_1d"):
                    ctrl.update_tplot = plotly.Figure(**CHART_STYLE).update
                    update_tplot_size(state.figure_size_1d, None, None, None)
