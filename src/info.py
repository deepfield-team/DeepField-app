from trame.widgets import vuetify3 as vuetify

def render_info():
    # Вынесем повторяющийся стиль в отдельную переменную
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
        vuetify.VCardText('Number of total timesteps: ' + '{{max_timestep}}', style=common_text_style)
        vuetify.VCardText('Number of wells: ' + '{{number_of_wells}}', style=common_text_style)

        vuetify.VCardTitle("Production rates", style=subtitle_style)
        vuetify.VCardText('Initial oil rate: ' + '{{start_oil_rate}}' + ' ' + '{{units1}}', style=common_text_style)
        vuetify.VCardText('Final oil rate: ' + '{{end_oil_rate}}' + ' ' + '{{units1}}', style=common_text_style)
        vuetify.VCardText('Initial water rate: ' + '{{start_water_rate}}' + ' ' + '{{units3}}', style=common_text_style)
        vuetify.VCardText('Final water rate: ' + '{{end_water_rate}}' + ' ' + '{{units3}}', style=common_text_style)

        
        vuetify.VCardTitle("Components and attributes", style=subtitle_style)
        vuetify.VCardTitle("Attributes of grid: " + "{{components_attrs['grid']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of rock: " + "{{components_attrs['rock']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of states: " + "{{components_attrs['states']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of tables: " + "{{components_attrs['tables']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of wells: " + "{{components_attrs['wells']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of faults: " + "{{components_attrs['faults']}}", style=common_text_style)
        vuetify.VCardTitle("Attributes of aquifers: " + "{{components_attrs['aquifers']}}", style=common_text_style)
