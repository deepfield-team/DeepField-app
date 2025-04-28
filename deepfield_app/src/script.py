"Script page."
from trame.widgets import vuetify3 as vuetify

from .config import state, FIELD
from .home import process_field


state.scriptInput = """
def f(field): #do not change this line
    return
"""
state.scriptOutput = ''
state.scriptRunning = False
state.fieldRestoring = False


@state.change("scriptRunning")
def run_script(scriptRunning, **kwargs):
    "Run script."
    _ = kwargs

    if not scriptRunning:
        return

    local_vars = {}
    exec(state.scriptInput, None, local_vars)
    success = False
    res = ''

    if FIELD['model_copy'] is None:
        FIELD['model_copy'] = FIELD['model'].copy()

    if 'f' in local_vars:
        try:
            res = local_vars['f'](FIELD['model'])
            success = True
        except Exception as err:
            res = 'Script failed: ' + str(err)
            success = False
    if success and FIELD['model'] is not None:
        try:
            process_field(FIELD['model'])
            state.modelID += 1
        except Exception as err:
            res = 'Field update failed: ' + str(err)
    state.scriptOutput = str(res)
    state.scriptRunning = False
    state.scriptFinished = True

@state.change("fieldRestoring")
def restore_field(fieldRestoring, **kwargs):
    "Restore initial field data."
    _ = kwargs

    if not fieldRestoring:
        return

    FIELD['model'] = FIELD['model_copy'].copy()
    process_field(FIELD['model'])
    state.fieldRestoring = False

def render_script():
    "Script page layout."
    vuetify.VTextarea(
    	v_model=('scriptInput',),
    	label="Type function to be executed")
    with vuetify.VBtn('Execute',
        click='scriptRunning = true',
        loading=('scriptRunning',)):
        vuetify.VTooltip(
            text='Run the script',
            activator="parent",
            location="end")

    with vuetify.VCard(style="margin-top: 10px", variant='flat'):
        vuetify.VCardTitle("Ouptput:")
        vuetify.VCardText('{{scriptOutput}}')

    with vuetify.VBtn('Restore field',
        click='fieldRestoring = true',
        loading=('fieldRestoring',)):
        vuetify.VTooltip(
            text='Discard all changes in the reservoir model',
            activator="parent",
            location="end")
