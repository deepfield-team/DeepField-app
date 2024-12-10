from trame.widgets import vuetify3 as vuetify

def render_info():
    with vuetify.VCard(style="margin: 10px"):
        vuetify.VCardTitle("Description of the reservoir model", style="font-size: 32px; font-weight: bold;")
        vuetify.VCardText('Dimensions: ' + '{{dimens}}', style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardText('Components in model: ' + '{{fluids}}', style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardText('Total cells: ' + '{{total_cells}}', style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardText('Active cells: ' + '{{active_cells}}', style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardText('Pore volume: ' + '{{pore_volume}}' + ' ' + '{{units3}}', style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardText('Number of total timesteps: ' + '{{max_timestep}}', style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardText('Number of wells: ' + '{{number_of_wells}}', style="font-size: 20px; margin-top: 10px;")
        
        vuetify.VCardTitle("Components and attributes", style="font-size: 28px; margin-top: 10px; font-weight: bold;")
        vuetify.VCardTitle("Attributes of grid: " + "{{components_attrs['grid']}}", style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardTitle("Attributes of rock: " + "{{components_attrs['rock']}}", style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardTitle("Attributes of states: " + "{{components_attrs['states']}}", style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardTitle("Attributes of tables: " + "{{components_attrs['tables']}}", style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardTitle("Attributes of wells: " + "{{components_attrs['wells']}}", style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardTitle("Attributes of faults: " + "{{components_attrs['faults']}}", style="font-size: 20px; margin-top: 10px;")
        vuetify.VCardTitle("Attributes of aquifers: " + "{{components_attrs['aquifers']}}", style="font-size: 20px; margin-top: 10px;")

