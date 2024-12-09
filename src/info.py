from trame.widgets import vuetify3 as vuetify


def render_info():
    with vuetify.VCard(style="margin: 10px"):
        vuetify.VCardTitle("Description of the reservoir model")
        vuetify.VCardText('Dimensions: ' + '{{dimens}}')
        vuetify.VCardTitle("Components in the model")
        vuetify.VCardText('Components: ' + '{{fluids}}')
        vuetify.VCardTitle("Number of total cells")
        vuetify.VCardText('Total: ' + '{{total_cells}}')
        vuetify.VCardTitle("Number of active cells")
        vuetify.VCardText('Active cells: ' + '{{active_cells}}')
        vuetify.VCardTitle("Pore volume")
        vuetify.VCardText('{{pore_volume}}' + ' ' + '{{units3}}')
        vuetify.VCardTitle("Number of total timesteps")
        vuetify.VCardText('num of timesteps: ' + '{{domainMax}}')
        vuetify.VCardTitle("Number of wells")
        vuetify.VCardText('Number of wells: ' + '{{number_of_wells}}')

        vuetify.VCardTitle("Components and attributes")
        
        vuetify.VCardTitle("Attributes of grid")
        vuetify.VCardText("{{components_attrs['grid']}}")
        
        vuetify.VCardTitle("Attributes of rock")
        vuetify.VCardText("{{components_attrs['rock']}}")

        vuetify.VCardTitle("Attributes of states")
        vuetify.VCardText("{{components_attrs['states']}}")

        vuetify.VCardTitle("Attributes of tables")
        vuetify.VCardText("{{components_attrs['tables']}}")

        vuetify.VCardTitle("Attributes of wells")
        vuetify.VCardText("{{components_attrs['wells']}}")

        vuetify.VCardTitle("Attributes of faults")
        vuetify.VCardText("{{components_attrs['faults']}}")

        vuetify.VCardTitle("Attributes of aquifers")
        vuetify.VCardText("{{components_attrs['aquifers']}}")