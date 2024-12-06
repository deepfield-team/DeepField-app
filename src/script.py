from trame.widgets import vuetify3 as vuetify

from .config import state, ctrl, FIELD


state.scriptInput = """
def f(field):
    return
"""
state.scriptOutput = ''

def run_script(*args, **kwargs):
	local_vars = {}
	exec(state.scriptInput, None, local_vars)
	if 'f' in local_vars:
		res = local_vars['f'](FIELD['model'])
		state.scriptOutput = str(res)

ctrl.run_script = run_script

def render_script():
    vuetify.VTextarea(
    	v_model=('scriptInput',),
    	label="Type function to be executed")
    vuetify.VBtn('Execute', click=ctrl.run_script)

    with vuetify.VCard(style="margin-top: 10px"):
        vuetify.VCardTitle("Ouptput:")
        vuetify.VCardText('{{scriptOutput}}')
