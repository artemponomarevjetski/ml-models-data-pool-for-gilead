"""
My App 2: Hello with Bokeh plot, Jinja2 template, and Bootstrap
"""

# adapted from https://github.com/silpara/simulators/blob/master/compartmental_models/SEIRD%20Simulator%20with%20Parameter%20Estimation%20in%20Python.ipynb
# eference https://www.idmod.org/docs/hiv/model-seir.html
# ref: https://www.medrxiv.org/content/10.1101/2020.04.01.20049825v1.full.pdf

import datetime
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import requests
from lmfit import Parameters
import bokeh.io
from bokeh.plotting import figure
from bokeh.embed import components
from flask import (
    Flask, request, render_template, abort, Response, redirect, url_for
)

bokeh.io.reset_output()

class Range():
    """
    this class determines ranges for input parameters
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

def data_formatting(df, varname, statecol, countrycol):
    """
    data formatting
    """
    values = [c for c in list(df) if c not in [statecol, countrycol, 'latitude', 'Lat', 'Long', 'longitude', 'location_geom']]
    ids = [statecol, countrycol]
    df = pd.melt(df, id_vars=ids, value_vars=values).groupby([countrycol, 'variable']).sum().reset_index()
    df.columns = ['country', 'date', varname]

    return df

def ode_model(z, t, beta, sigma, gamma, mu):
    """
    Reference https://www.idmod.org/docs/hiv/model-seir.html
    """
    S, E, I, R, D = z
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def ode_solver(t, initial_conditions, params):
    """
    odinary differential equations solver
    """
    initE, initI, initR, initN, initD = initial_conditions
    beta, sigma, gamma, mu = params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
    initS = initN - (initE + initI + initR + initD)
    res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu))
    return res

def model1(df_covid_history, initE, initI, initR, initD, initN, beta, sigma, gamma, mu, days, param_fitting=False):
    """
    SEIRD
    """
    initial_conditions = [initE, initI, initR, initN, initD]
    params = Parameters()
    params.add('beta', value=beta, min=0, max=1000)
    params.add('sigma', value=sigma, min=0, max=1000)
    params.add('gamma', value=gamma, min=0, max=1000)
    params.add('mu', value=mu, min=0, max=10)
    print(params)
    params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value = [beta, sigma, gamma, mu]
    tspan = np.arange(0, days, 1)
    sol = ode_solver(tspan, initial_conditions, params)
    return sol

def model_response(df_covid_history, initE, initI, initR, initD, initN, beta, sigma, gamma, mu, days, param_fitting):
    """
    another wrapper around the actual model
    """
    sol = model1(df_covid_history, initE, initI, initR, initD, initN, beta, sigma, gamma, mu, days, param_fitting)
    S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
    y = I
    return y


app = Flask(__name__)

print("\nHERE1\n")
@app.route('/', methods=['GET', 'POST'])
@app.route('/<xdata>/<slider_val>/<option>', methods=['GET', 'POST'])
def index(xdata=None, slider_val=None, option=None):
    """
    flask index function that interacts with index.html
    """
    time_format = "%d%b%Y %H:%M"
    print('Current time: ', datetime.datetime.now().strftime(time_format))

    df_pop = pd.read_csv('input/population.csv')
    print('Population dataframe: ', df_pop)

    #    statecol = 'province_state'
    #    countrycol = 'country_region' -- what are these values for?

    statecol = 'Province/State'
    countrycol = 'Country/Region'
    # Import JHU data and format
    #df_confirmed = data_formatting(pd.read_csv('1_Data/confirmed.csv'), 'confirmed')
    #df_deaths = data_formatting(pd.read_csv('1_Data/deaths.csv'), 'deaths')
    #df_recovered = data_formatting(pd.read_csv('1_Data/recovered.csv'), 'recovered')
    c_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    df_confirmed = data_formatting(pd.read_csv(c_path), 'confirmed', statecol, countrycol)
    d_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    df_deaths = data_formatting(pd.read_csv(d_path), 'deaths', statecol, countrycol)
    r_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    df_recovered = data_formatting(pd.read_csv(r_path), 'recovered', statecol, countrycol)

    # Merge JHU data
    df_merge = pd.merge(df_confirmed, df_deaths, on=['country', 'date'], how='inner')
    df_merge = pd.merge(df_merge, df_recovered, on=['country', 'date'], how='inner')

    # Add Population
    df_merge = pd.merge(df_merge, df_pop[['country_name', 'midyear_population']],
                        left_on=['country'], right_on=['country_name'])

    print(df_merge)
    # Fix the date column
    try:
        df_merge['date'] = pd.to_datetime(df_merge['date'], format='_%m_%d_%y')
    except Exception:
        df_merge['date'] = pd.to_datetime(df_merge['date'], format='%m/%d/%y')

    # Column names
    df_merge.columns = ['Country', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'country_name', 'population']
    df_merge['Susceptible'] = df_merge['population'] - df_merge['Confirmed']
    df_merge['Infected'] = df_merge['Confirmed'] - df_merge['Deaths'] - df_merge['Recovered']
    print(df_merge)

    # different data set, kept the first choice of loading data for consistency
    response = requests.get('https://api.rootnet.in/covid19-in/stats/history')
    print('Request Success? {}'.format(response.status_code == 200))
    covid_history = response.json()['data']

    keys = ['day', 'total', 'confirmedCasesIndian', 'confirmedCasesForeign', 'confirmedButLocationUnidentified',
            'discharged', 'deaths']
    df_covid_history = pd.DataFrame([[d.get('day'),
                                      d['summary'].get('total'),
                                      d['summary'].get('confirmedCasesIndian'),
                                      d['summary'].get('confirmedCasesForeign'),
                                      d['summary'].get('confirmedButLocationUnidentified'),
                                      d['summary'].get('discharged'),
                                      d['summary'].get('deaths')]
                                     for d in covid_history],
                                    columns=keys)
    df_covid_history = df_covid_history.sort_values(by='day')
    df_covid_history['infected'] = df_covid_history['total'] - df_covid_history['discharged'] - df_covid_history['deaths']
    df_covid_history['total_recovered_or_dead'] = df_covid_history['discharged'] + df_covid_history['deaths']

    # ref: https://www.medrxiv.org/content/10.1101/2020.04.01.20049825v1.full.pdf
    # S0 = 966000000 --what is this param. for?
    initE = 10
    initI = 5
    initR = 0
    initD = 0
    initN = 200000000
    R0 = 4.0
    sigma = 0.1923076923076923
    gamma = 0.3448275862068966
    mu = 0.034
    days = 112
    param_fitting = True

    beta = R0 * gamma

    print("\nHERE2\n", 'xdata = ', xdata, 'slider = ', 50, 'radiobutton = ', option)
    if request.method == 'GET':
        print("\nHERE3\n", xdata, slider_val, option)
        starting_sv = 0
        if slider_val is None:
            starting_sv = 0
        else:
            starting_sv = slider_val
        print("\nHERE3a\n", "default_sv = ", "current slider = ", starting_sv, "current radiobutton = ", option)
        checks = ['', '', '']
        kwargs = {'title': 'Flask-bokeh-webapp', 'slider_current_value' : starting_sv, 'check1' : 'checked'}
        if xdata is not None and slider_val is not None and option is not None:
            print("\nHERE4\n")
            try:
                xdata = [float(x) for x in xdata.split(',')]
                print("\nHERE4a\n")
            except ValueError as err:
                print(xdata)
                print("\nHERE4b\n", err)
                pass
            else:
                print("\nHERE5\n", xdata, slider_val, option)
                if option[8] == '1':
                    print("\nHERE5a\n")
                    plot = figure(title='squares from input')
                    x = np.arange(days)
                    # make initE from 0 to 100 using the slider value with sv=50 corresponding to initE
                    # a=1, when sv=100; a=10/50, when sv=50
                    # 100*b+c=1
                    # 50*b+c=0.2
                    # 50b=.8
                    b = .8/50.
                    c = 1.-100.*b
                    a = b*float(slider_val)+c
                    initEslider = a*float(slider_val)*initE/10.
                    y = model_response(df_covid_history, initEslider, initI, initR, initD, initN, beta, sigma, gamma, mu, days, param_fitting)
                    #    output_file("slider.html", title="slider.py example") -- does this command output an html file into working dir?
                    plot.line(x, y, legend='COVID-19 Model 1')
                    plot_script, plot_div = components(plot)
                    kwargs.update(plot_script=plot_script, plot_div=plot_div)
                    checks[0] = 'checked'
                    kwargs.update(check1=checks[0], check2=checks[1], check3=checks[2])
                elif option[8] == '2':
                    print("\nHERE5b\n")
                    plot = figure(title='squares from input')
                    plot.line(xdata, [float(slider_val)*x*x for x in xdata], legend='y=slider_value*x^2')
                    plot_script, plot_div = components(plot)
                    kwargs.update(plot_script=plot_script, plot_div=plot_div)
                    checks[1] = 'checked'
                    kwargs.update(check1=checks[0], check2=checks[1], check3=checks[2])
                elif option[8] == '3':
                    print("\nHERE5c\n")
                    plot = figure(title='squares from input')
                    plot.line(xdata, [float(slider_val)*x*x*x for x in xdata], legend='y=slider_value*x^3')
                    plot_script, plot_div = components(plot)
                    kwargs.update(plot_script=plot_script, plot_div=plot_div)
                    checks[2] = 'checked'
                    kwargs.update(check1=checks[0], check2=checks[1], check3=checks[2])
                else:
                    print("\nHERE5d\n")
        print("\nHERE6\n")
        print(', '.join(['{}={!r}'.format(k, v) for k, v in kwargs.items()]))
        print("\nHERE6a\n")
        return render_template('index.html', **kwargs)
    elif request.method == 'POST':
        print("\nHERE7\n")
        xdata = request.form.get('xdata')
        comma_seperated = request.form.get('commaSeparatedCheck')
        if not comma_seperated:
            print("\nHERE7a\n", comma_seperated)
            xdata = ','.join(xdata.split())
        print("\nHERE8\n", xdata)
        slider_val = request.form.get('slider')
        print("\nHERE9\n", slider_val)
        option = request.form.getlist('options')
        print("\nHERE11\n", option)
        return redirect(url_for('index', xdata=xdata, slider_val=slider_val, option=option))
    abort(404)
    abort(Response('\nHERE12\n'))


if __name__ == '__main__':
    app.run(debug=True)
