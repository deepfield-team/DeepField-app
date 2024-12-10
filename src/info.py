from trame.widgets import vuetify3 as vuetify

def render_info():
    common_text_style = "font-size: 20px; margin-top: 10px;"
    title_style = "font-size: 32px; font-weight: bold;"
    subtitle_style = "font-size: 28px; margin-top: 10px; font-weight: bold;"

    with vuetify.VCard(style="margin: 10px"):
        vuetify.VCardTitle("Description of the reservoir model", style=title_style)
        vuetify.VCardText('Dimensions: ' + '{{dimens}}', style=common_text_style)
        vuetify.VCardText('Components in model: ' + '{{fluids}}', style=common_text_style)
        vuetify.VCardText('Total cells: ' + '{{total_cells}}', style=common_text_style)
        vuetify.VCardText('Active cells: ' + '{{active_cells}}', style=common_text_style)
        vuetify.VCardText('Pore volume: ' + '{{pore_volume}}' + ' ' + '{{units3}}', style=common_text_style)
        vuetify.VCardText('Oil volume: ' + '{{oil_volume}}' + ' ' + '{{units3}}', style=common_text_style)
        vuetify.VCardText('Gas volume: ' + '{{oil_volume}}' + ' ' + '{{units4}}', style=common_text_style)
        vuetify.VCardText('Water volume: ' + '{{oil_volume}}' + ' ' + '{{units3}}', style=common_text_style)

        vuetify.VCardText('Number of total timesteps: ' + '{{max_timestep}}', style=common_text_style)

        vuetify.VCardText('Number of wells: ' + '{{number_of_wells}}', style=common_text_style)
        vuetify.VCardText('Total oil production: ' + '{{total_oil_production}}' + ' ' + '{{units1}}', style=common_text_style)
        vuetify.VCardText('Total gas production: ' + '{{total_gas_production}}' + ' ' + '{{units2}}', style=common_text_style)
        vuetify.VCardText('Total water production: ' + '{{total_water_production}}' + ' ' + '{{units1}}', style=common_text_style)
        
        vuetify.VCardTitle("Components and attributes", style=subtitle_style)
        vuetify.VCardTitle("Attributes of grid: " + "{{components_attrs['grid']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of rock: " + "{{components_attrs['rock']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of states: " + "{{components_attrs['states']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of tables: " + "{{components_attrs['tables']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of wells: " + "{{components_attrs['wells']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of faults: " + "{{components_attrs['faults']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of aquifers: " + "{{components_attrs['aquifers']}}", style=common_text_style)
