from trame.widgets import html, vuetify3 as vuetify
from trame.ui.vuetify3 import VAppLayout

from src.home import render_home, make_empty_dataset
from src.view_3d import render_3d
from src.view_2d import render_2d
from src.view_1d import render_1d
from src.common import reset_camera
from src.info import render_info
from src.config import server, state


make_empty_dataset()
reset_camera()

with VAppLayout(server) as layout:
    with layout.root:
        with vuetify.VAppBar(app=True, clipped_left=True):
            vuetify.VAppBarNavIcon(click='drawer =! drawer')

            vuetify.VToolbarTitle("DeepField")
            with vuetify.VTabs(v_model=('activeTab', 'home'), style='flex: 2'):
                vuetify.VTab('Home', value="home")
                vuetify.VTab('3d', value="3d")
                vuetify.VTab('2d', value="2d")
                vuetify.VTab('1d', value="1d")
                vuetify.VTab('Info', value="info")

            with vuetify.VBtn(icon=True):
                vuetify.VIcon("mdi-settings")
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
            with html.Div(v_if="activeTab === '1d'", classes="fill-height"):
                render_1d()
            with html.Div(v_if="activeTab === 'info'"):
                render_info()

        with vuetify.VNavigationDrawer(
            app=True,
            clipped=True,
            stateless=True,
            v_model=("drawer", False),
            width=200):
            vuetify.VSelect(
                v_model=('user_request',),
                label='Recent files',
                items=('recentFiles', ),
                v_if="activeTab === 'home'"
                )
            vuetify.VBtn(
                'Clean history',
                click='recentFiles = []',
                v_if="activeTab === 'home'"
                )
            vuetify.VSelect(
                v_model=('activeField', state.field_attrs[0] if state.field_attrs else None),
                label='Select data',
                items=('field_attrs', ),
                v_if="(activeTab === '3d') | (activeTab === '2d')"
                )
            vuetify.VCheckbox(
                label='Threshold selector',
                v_if="activeTab === '3d'",
                style='height: 8vh'
                )
            vuetify.VCheckbox(
                label='Slice range selector',
                v_model='show_slice',
                v_if="activeTab === '3d'",
                classes='pa-0 ma-0',
                style='height: 8vh'
                )
            vuetify.VCheckbox(
                label='Show wells',
                v_if="activeTab === '3d'",
                style='height: 8vh'
                )
            vuetify.VCheckbox(
                label='Show faults',
                v_if="activeTab === '3d'",
                style='height: 8vh'
                )
            vuetify.VSelect(
                label="Colormap",
                v_model=("colormap", 'jet'),
                items=("colormaps",
                    ["gray", "jet", "hsv", "Spectral", "twilight", "viridis"],
                ),
                hide_details=True,
                dense=True,
                outlined=True,
                v_if="(activeTab === '3d') | (activeTab === '2d')"
            )
            vuetify.VSlider(
                min=0,
                max=1,
                step=0.1,
                v_model=('opacity', 1),
                label="Opacity",
                classes="mt-8 mr-3",
                hide_details=False,
                dense=False,
                thumb_label=True,
                v_if="activeTab === '3d'"
                )


if __name__ == "__main__":
    server.start()
