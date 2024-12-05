from trame.widgets import html, vuetify3 as vuetify
from trame.ui.vuetify3 import VAppLayout

from src.home import render_home, make_empty_dataset
from src.view_3d import render_3d
from src.view_2d import render_2d
from src.view_1d import render_ts, render_pvt
from src.common import reset_camera
from src.info import render_info
from src.config import server, state


make_empty_dataset()
reset_camera()

with VAppLayout(server) as layout:
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
            with vuetify.VBtn(icon=True):
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
                pass


if __name__ == "__main__":
    server.start()
