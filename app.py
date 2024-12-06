from trame.widgets import html, client, vuetify3 as vuetify
from trame.ui.vuetify3 import VAppLayout

from src.home import render_home, make_empty_dataset
from src.view_3d import render_3d
from src.view_2d import render_2d
from src.view_1d import render_ts, render_pvt
from src.common import reset_camera
from src.info import render_info
from src.script import render_script
from src.config import server, state, ctrl


state.theme = 'light'
state.sideBarColor = "grey-lighten-4"
state.plotlyTheme = 'plotly'

def change_theme(*args, **kwargs):
    if state.theme == 'light':
        state.theme = 'dark'
        state.sideBarColor = "grey-darken-4"
        state.plotlyTheme = 'plotly_dark'
    else:
        state.theme = 'light'
        state.sideBarColor = "grey-lighten-4"
        state.plotlyTheme = 'plotly'
ctrl.change_theme = change_theme


make_empty_dataset()
reset_camera()
with VAppLayout(server, theme=('theme',)) as layout:
    style = client.Style("body { background-color: white }")
    ctrl.update_style = style.update
    with layout.root:
        with vuetify.VAppBar(app=True, clipped_left=True, density="compact"):
            vuetify.VToolbarTitle("DeepField")
            vuetify.VSpacer()
            with vuetify.VTabs(v_model=('activeTab', 'home')):
                vuetify.VTab('Home', value="home")
                vuetify.VTab('3d view', value="3d")
                vuetify.VTab('2d view', value="2d")
                vuetify.VTab('Timeseries', value="ts")
                vuetify.VTab('PVT/RP', value="pvt")
                vuetify.VTab('Info', value="info")
                vuetify.VTab('Script', value="script")
            
            vuetify.VSpacer()
            with vuetify.VBtn(icon=True, click=ctrl.change_theme):
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
            with html.Div(v_if="activeTab === 'ts'", classes="fill-height"):
                render_ts()
            with html.Div(v_if="activeTab === 'pvt'", classes="fill-height"):
                render_pvt()
            with html.Div(v_if="activeTab === 'info'"):
                render_info()
            with html.Div(v_if="activeTab === 'script'"):
                render_script()


if __name__ == "__main__":
    server.start()
