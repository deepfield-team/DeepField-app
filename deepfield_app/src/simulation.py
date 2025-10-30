"""JutulDarcy simulation utils."""
import importlib
import pandas as pd
import numpy as np
from datetime import timedelta
from queue import Empty

from .config import state, FIELD

def jd_load(path):
    jd = importlib.import_module('jutuldarcy')

    return jd.setup_case_from_data_file(path)

def jd_simulate(path):
    jd = importlib.import_module('jutuldarcy')

    return jd.simulate_data_file(path, convert=True)

def well_states(well, dates, start_date):
    states_df = pd.concat([dates, pd.DataFrame(well)], axis=1)
    record0 = pd.DataFrame([[start_date] + [0.0]*(len(well.columns))],
                           columns=states_df.columns)
    states_df = pd.concat([record0, states_df])
    return states_df

def convert_results(case, res, output):
    jd = importlib.import_module('jutuldarcy')
    jl = importlib.import_module('juliacall').Main
    
    state0_pressure = np.array(
        jl.seval("state0 -> state0[:Reservoir][:Pressure]")(case.state0)).reshape(1, -1)

    state0_sats = np.array(
        jl.seval("state0 -> state0[:Reservoir][:Saturations]")(case.state0)).reshape(1, 2, -1)

    n_steps = len(res['STATES'])
    jd_pressure = np.array([res['STATES'][i]['Pressure'] for i in range(n_steps)])
    jd_sats = np.array([res['STATES'][i]['Saturations'] for i in range(n_steps)])

    jd_pressure = np.vstack([state0_pressure, jd_pressure])
    jd_sats = np.vstack([state0_sats, jd_sats])

    output['saturations'] = jd_sats
    output['pressure'] = jd_pressure

    n_timestamps = len(res["DAYS"])
    start_date = case.input_data["RUNSPEC"]["START"]
    timestamps = [start_date + timedelta(days=res["DAYS"][i]) for i in range(n_timestamps)]
    dates = pd.DataFrame({"DATE": timestamps})

    welldata = {}

    wellnames = res["WELLS"].keys()
    welldata = {w: {"RESULTS": well_states(pd.DataFrame(res["WELLS"][w]), dates, start_date)} for w in wellnames}

    output['wellnames'] = wellnames
    output['welldata'] = welldata

def simulate(queue, results, timeout=1):
    while True:
        try:
            task_id, path = queue.get(timeout=timeout)
            try:
                case = jd_load(path)
                res = jd_simulate(path)
                convert_results(case, res, results)

                field = FIELD['model']
                field.states.pressure = results['pressure']
                field.states.soil = results['saturations'][:, 1, :]
                field.states.swat = results['saturations'][:, 0, :]
                field.states.to_spatial()
                
                field.wells.update(results['welldata'])
                state.modelID += 1

                results[task_id] = 'Done'
            except:
                results[task_id] = 'Failed'
        except Empty:
            continue
        except Exception as err:
            break