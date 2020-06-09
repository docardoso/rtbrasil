# For some reason Theano is unhappy when I run the GP, need to disable future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use('agg')

import os
import requests
import gzip
import time
import pymc3 as pm
import pandas as pd
import numpy as np
import theano
import theano.tensor as tt

from datetime import date
from datetime import datetime

import multiprocessing as mp

import urllib.request
import json

STATE_CITY_SEP = ';'

# Linhas 29-98: importa dados usando a API brasil.io
# def get_csv_covid19brasilio( outputFilePath ):
#    csvURL = "https://data.brasil.io/dataset/covid19/caso.csv.gz"
#    csvFileName = outputFilePath + csvURL.split("/")[-1]
#    with open(csvFileName, "wb") as f:
#        r = requests.get(csvURL)
#        f.write(r.content)

#    print(csvFileName)
#    return csvFileName
# csv_file = get_csv_covid19brasilio('/tmp/')
# data = pd.read_csv(
#     csv_file,
#     compression='gzip',
#     parse_dates=False,
#     low_memory=False)
# data = data[(data.date.notna()) & (data.city != 'Importados/Indefinidos') & (data.confirmed > 0)]

# states = data[data.city.isna()]
# maxDate = states.date.max()
# statesToUpdate = states[(states.is_last == True) & (states.date != maxDate)]
# #print(statesToUpdate) # FOR DEBUG

# # Linhas 52-70: adiciona as informações faltantes dos estados, antes de computar as informacoes do pais.
# # Linhas 53-54: considera que existe apenas um dado (dia) faltante.
# #statesToUpdate.date = maxDate
# #states = states.append(statesToUpdate, ignore_index=True)

# # Linhas 58-70: considera que pode haver mais de um dado (dia) faltante para os estados.
# # Nesse caso, o último dado é replicado para os dias (futuros) faltantes.
# def addMissingRows ( df, statesToUpdate, maxDate ) :
#     for index, row in statesToUpdate.iterrows():

#         date1 = df[df.state == row.state].date.max()
#         date1 = pd.to_datetime(date1, format='%Y-%m-%d').date()
#         date2 = pd.to_datetime(maxDate, format='%Y-%m-%d').date()

#         while (date1 < date2):
#             date1 = date1 + pd.to_timedelta(1, unit='d')
#             newRow = row
#             newRow.date = date1
#             df = df.append(newRow, ignore_index=True)
#     return df

# #states.to_csv('country1.csv') # FOR DEBUG
# states = addMissingRows(states, statesToUpdate, maxDate)
# #states.to_csv('country2.csv') # FOR DEBUG
# states.state = 'BR' + STATE_CITY_SEP
# country = states.groupby(['date','state'])['confirmed'].sum().reset_index()

# # A linha abaixo reinicia as infos dos estados.
# states = data[data.city.isna()]

# # Para usar os estados com as infos (dias) faltantes, as quais foram utilizadas para gerar os dados do pais,
# # basta descomentar a linha abaixo.
# states = addMissingRows(states, statesToUpdate, maxDate)

# cities = data[(data.is_last == True) & (~data.city.isna())].sort_values('confirmed').groupby('state').tail(12).city_ibge_code
# cities = data[data.city_ibge_code.isin(cities)]

# states = pd.concat([states, cities])
# states.state = states.state + STATE_CITY_SEP + states.city.fillna('')

# states = states[['date', 'state', 'confirmed']]
# states = pd.concat([states, country])
# print(states) # FOR DEBUG
# states = states.rename(columns={'confirmed': 'positive'})
# states.date = pd.to_datetime(states.date)
# states = states.set_index(['state', 'date']).sort_index()
# #states.to_csv('brasil_io.csv') # FOR DEBUG
# #time.sleep(10000000) # FOR DEBUG

# Linhas 101-131: importa dados do Portal do SUS
def get_csv_covid19br( outputFilePath ):
    url = 'https://xx9p7hp1p7.execute-api.us-east-1.amazonaws.com/prod/PortalGeral'
    values = {'X-Parse-Application-Id' : 'unAFkcaNDeXajurGB7LChj8SgQYS2ptm'}
    req  = urllib.request.Request( url, None, values )
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())
        csvURL = data[ 'results' ][ 0 ][ 'arquivo' ][ 'url' ]
        csvFileName = csvURL[ csvURL.rfind('/') + 1 : ]
        print( 'csvURL:', csvURL )
        print( 'csvFileName:', csvFileName )
        urllib.request.urlretrieve( csvURL, outputFilePath + '/' + csvFileName )
        return outputFilePath + '/' + csvFileName
csv_file = get_csv_covid19br('/tmp')
data = pd.read_excel(csv_file)
data = data[data.data.notna()]

states = data[data.codmun.isna()]
states.estado = states.estado.fillna('BR')

cities = data[(data.data==data.data.max()) & (~data.municipio.isna())].sort_values('casosAcumulado').groupby('estado').tail(12).codmun
cities = data[data.codmun.isin(cities)]
states = pd.concat([states, cities])
states.estado = states.estado + STATE_CITY_SEP + states.municipio.fillna('')

#import pdb; pdb.set_trace()
states = states[['data', 'estado', 'casosAcumulado']]
print(states)
states = states.rename(columns={'data':'date', 'estado': 'state', 'casosAcumulado': 'positive'})
states.date = pd.to_datetime(states.date)
states = states.set_index(['state', 'date']).sort_index()
#states.to_csv('sus.csv')

def download_file(url, local_filename):
    """From https://stackoverflow.com/questions/16694907/"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    return local_filename


URL = "https://raw.githubusercontent.com/beoutbreakprepared/nCoV2019/master/latest_data/latestdata.tar.gz"
LINELIST_PATH = '/tmp/linelist.tar.gz'

print('Downloading file, this will take a while ~100mb')
try:
    download_file(URL, LINELIST_PATH)
    print('Done downloading.')
except:
    raise Exception('Something went wrong. Try again.')
    
# Load the patient CSV
patients = pd.read_csv(
    LINELIST_PATH,
    compression='gzip',
    parse_dates=False,
    low_memory=False)

#patients = patients[patients.country == 'Brazil']
patients = patients[['date_onset_symptoms','date_confirmation', ]]

patients.columns = ['Onset', 'Confirmed']

# There's an errant reversed date
patients = patients.replace('01.31.2020', '31.01.2020')

# Only keep if both values are present
patients = patients.dropna()

# Must have strings that look like individual dates
# "2020.03.09" is 10 chars long
is_ten_char = lambda x: x.str.len().eq(10)
patients = patients[is_ten_char(patients.Confirmed) & 
                    is_ten_char(patients.Onset)]

# Convert both to datetimes
patients.Confirmed = pd.to_datetime(
    patients.Confirmed, format='%d.%m.%Y')
patients = patients[patients.Onset != '31.04.2020']
#import pdb; pdb.set_trace()
patients.Onset = pd.to_datetime(
    patients.Onset, format='%d.%m.%Y')

# Only keep records where confirmed > onset
patients = patients[patients.Confirmed > patients.Onset]

# Calculate the delta in days between onset and confirmation
delay = (patients.Confirmed - patients.Onset).dt.days

print(delay.describe())

# Convert samples to an empirical distribution
p_delay = delay.value_counts().sort_index()
new_range = np.arange(0, p_delay.index.max()+1)
p_delay = p_delay.reindex(new_range, fill_value=0)
p_delay /= p_delay.sum()

def confirmed_to_onset(confirmed, p_delay):

    assert not confirmed.isna().any()
    
    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)

    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1],
                       periods=len(convolved))

    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)
    
    return onset


#onset = confirmed_to_onset(confirmed, p_delay)

def adjust_onset_for_right_censorship(onset, p_delay):
    cumulative_p_delay = p_delay.cumsum()
    
    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)
    
    # Add ones and flip back
    cumulative_p_delay = np.pad(
        cumulative_p_delay,
        padding_shape,
        constant_values=1)
    cumulative_p_delay = np.flip(cumulative_p_delay)
    
    # Adjusts observed onset values to expected terminal onset values
    adjusted = onset / cumulative_p_delay
    
    return adjusted, cumulative_p_delay


#adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)

class MCMCModel(object):
    
    def __init__(self, region, onset, cumulative_p_delay, window=100):
        
        # Just for identification purposes
        self.region = region
        
        # For the model, we'll only look at the last N
        self.onset = onset.iloc[-window:]
        self.cumulative_p_delay = cumulative_p_delay[-window:]
        
        # Where we store the results
        self.trace = None
        self.trace_index = self.onset.index[1:]

    def run(self, chains=1, tune=3000, draws=1000, target_accept=.95):

        with pm.Model() as model:

            # Random walk magnitude
            step_size = pm.HalfNormal('step_size', sigma=.03)

            # Theta random walk
            theta_raw_init = pm.Normal('theta_raw_init', 0.1, 0.1)
            theta_raw_steps = pm.Normal('theta_raw_steps', shape=len(self.onset)-2) * step_size
            theta_raw = tt.concatenate([[theta_raw_init], theta_raw_steps])
            theta = pm.Deterministic('theta', theta_raw.cumsum())

            # Let the serial interval be a random variable and calculate r_t
            serial_interval = pm.Gamma('serial_interval', alpha=6, beta=1.5)
            gamma = 1.0 / serial_interval
            r_t = pm.Deterministic('r_t', theta/gamma + 1)

            inferred_yesterday = self.onset.values[:-1] / self.cumulative_p_delay[:-1]
            
            expected_today = inferred_yesterday * self.cumulative_p_delay[1:] * pm.math.exp(theta)

            # Ensure cases stay above zero for poisson
            mu = pm.math.maximum(.1, expected_today)
            observed = self.onset.round().values[1:]
            cases = pm.Poisson('cases', mu=mu, observed=observed)

            try:
                self.trace = pm.sample(
                    chains=chains,
                    tune=tune,
                    draws=draws,
                    target_accept=target_accept)
            except:
                print('BAAAAAAAAAAAAAAAAA:', self.region)
                raise
            
            return self
    
    def run_gp(self):
        with pm.Model() as model:
            gp_shape = len(self.onset) - 1

            length_scale = pm.Gamma("length_scale", alpha=3, beta=.4)

            eta = .05
            cov_func = eta**2 * pm.gp.cov.ExpQuad(1, length_scale)

            gp = pm.gp.Latent(mean_func=pm.gp.mean.Constant(c=0), 
                              cov_func=cov_func)

            # Place a GP prior over the function f.
            theta = gp.prior("theta", X=np.arange(gp_shape)[:, None])

            # Let the serial interval be a random variable and calculate r_t
            serial_interval = pm.Gamma('serial_interval', alpha=6, beta=1.5)
            gamma = 1.0 / serial_interval
            r_t = pm.Deterministic('r_t', theta / gamma + 1)

            inferred_yesterday = self.onset.values[:-1] / self.cumulative_p_delay[:-1]
            expected_today = inferred_yesterday * self.cumulative_p_delay[1:] * pm.math.exp(theta)

            # Ensure cases stay above zero for poisson
            mu = pm.math.maximum(.1, expected_today)
            observed = self.onset.round().values[1:]
            cases = pm.Poisson('cases', mu=mu, observed=observed)

            self.trace = pm.sample(chains=1, tune=1000, draws=1000, target_accept=.8)
        return self
    
def df_from_model(model):
    
    r_t = model.trace['r_t']
    mean = np.mean(r_t, axis=0)
    median = np.median(r_t, axis=0)
    hpd_90 = pm.stats.hpd(r_t, credible_interval=.9)
    hpd_50 = pm.stats.hpd(r_t, credible_interval=.5)
    
    idx = pd.MultiIndex.from_product([
            [model.region],
            model.trace_index
        ], names=['region', 'date'])
        
    df = pd.DataFrame(data=np.c_[mean, median, hpd_90, hpd_50], index=idx,
                 columns=['mean', 'median', 'lower_90', 'upper_90', 'lower_50','upper_50'])
    return df

def create_and_run_model(name, state):
    state = np.maximum.accumulate(state)
    confirmed = state.positive.diff().dropna()
    onset = confirmed_to_onset(confirmed, p_delay)
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    return MCMCModel(name, onset, cumulative_p_delay).run()

state_inputs = [(state, grp.droplevel(0)) for state, grp in states.groupby('state')]

with mp.Pool() as pool:
    results = pool.starmap(create_and_run_model, state_inputs)


models = {model.region: model for model in results}

# Check to see if there were divergences
n_diverging = lambda x: x.trace['diverging'].nonzero()[0].size
divergences = pd.Series([n_diverging(m) for m in models.values()], index=models.keys())
has_divergences = divergences.gt(0)

print('Diverging states:')
print(divergences[has_divergences])

# Rerun states with divergences
def do_run(state, state_model):
    return state, state_model.run()

diverging = [(state, models[state]) for state in divergences[has_divergences].index]

with mp.Pool() as pool:
    result = pool.starmap(do_run, diverging)
    
for state, state_model in result:
    models[state] = state_model
    
results = None

for state, model in models.items():

    df = df_from_model(model)

    if results is None:
        results = df
    else:
        results = pd.concat([results, df], axis=0)
        
#import pdb; pdb.set_trace()
results = results.reset_index()
results[['state', 'city']] = results.region.str.split(STATE_CITY_SEP, expand=True)
results = results.set_index(['state', 'city', 'date']).drop('region', 1)
results.round(3).to_pickle('rt_brazil.pickle')
print('DONE')
