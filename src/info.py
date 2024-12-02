from trame.widgets import vuetify3 as vuetify


def render_info():
    with vuetify.VCard(style="margin: 10px"):
        vuetify.VCardTitle("Description of the reservoir model")
        vuetify.VCardText('Dimensions: ' + '{{dimens}}')
