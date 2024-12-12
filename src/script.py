from trame.widgets import vuetify3 as vuetify

from .config import state, ctrl, FIELD
from .home import process_field


state.scriptInput = """
def f(field):
    return
"""
state.scriptOutput = ''
state.scriptRunning = False
state.scriptFinished = False


@state.change("scriptRunning")
def run_script(scriptRunning, **kwargs):
    _ = kwargs

    if not scriptRunning:
        return

    local_vars = {}
    exec(state.scriptInput, None, local_vars)
    success = False
    res = ''
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
        except Exception as err:
            res = 'Field update failed: ' + str(err)
    state.scriptOutput = str(res)
    state.scriptRunning = False
    state.scriptFinished = True

def render_script():
    vuetify.VTextarea(
    	v_model=('scriptInput',),
    	label="Type function to be executed")
    vuetify.VBtn('Execute',
        click='scriptRunning = true',
        loading=('scriptRunning',))

    with vuetify.VCard(style="margin-top: 10px"):
        vuetify.VCardTitle("Ouptput:")
        vuetify.VCardText('{{scriptOutput}}')
