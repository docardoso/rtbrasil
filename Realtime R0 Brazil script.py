import pandas as pd
import numpy as np
from scipy import stats as sps
from scipy.interpolate import interp1d
import urllib.request
import json
FILTERED_REGIONS = []
'''
    'Virgin Islands',
    'American Samoa',
    'Northern Mariana Islands',
    'Guam',
    'Puerto Rico']
'''
FILTERED_REGION_CODES = []#'AS', 'GU', 'PR', 'VI', 'MP']
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)
GAMMA = 1/7
def highest_density_interval(pmf, p=.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i+1:]):
            if (high_value-value > p) and (not best or j<best[1]-best[0]):
                best = (i, i+j+1)
                break
    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pd.Series([low, high], index=[f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])
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
last_line = None  #next(i for i, x in enumerate(open(csv_file, encoding='ISO-8859-1')) if x[0] == ';') - 1
#states = pd.read_csv(csv_file, nrows=last_line, encoding = 'ISO-8859-1', delimiter=';', usecols=['data', 'estado', 'casosAcumulados'])
states = pd.read_excel(csv_file)
states = states[states.codmun.isna()][['data', 'estado', 'casosAcumulado']]
states.estado = states.estado.fillna('BR')
print(states)
states = states.rename(columns={'data':'date', 'estado': 'state', 'casosAcumulado': 'positive'})
states.date = pd.to_datetime(states.date)
#agg = states.groupby('date').sum()
#agg['state'] = 'BR'
#states = states.append(agg.reset_index(), sort=False)
states = states.set_index(['state', 'date']).squeeze().sort_index()
#import pdb; pdb.set_trace()
def prepare_cases(cases):
    new_cases = cases.diff()
    smoothed = new_cases.rolling(9,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=3).round()
    idx_start = np.searchsorted(smoothed, 1)
    try:
        idx_start = idx_start[0]
    except IndexError:
        pass
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    return original, smoothed
def get_posteriors(sr, sigma=0.15):
    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 
    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()
    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0
    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    return posteriors, log_likelihood
sigmas = np.linspace(1/20, 1, 20)  #np.array([0.15]) #
targets = ~states.index.get_level_values('state').isin(FILTERED_REGION_CODES)
states_to_process = states.loc[targets]
results = {}
for state_name, cases in states_to_process.groupby(level='state'):
    print(state_name)
    new, smoothed = prepare_cases(cases)
    result = {}
    # Holds all posteriors with every given value of sigma
    result['posteriors'] = []
    # Holds the log likelihood across all k for each value of sigma
    result['log_likelihoods'] = []
    for sigma in sigmas:
        posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
        result['posteriors'].append(posteriors)
        result['log_likelihoods'].append(log_likelihood)
    # Store all results keyed off of state name
    results[state_name] = result
print('Done.')
total_log_likelihoods = np.zeros_like(sigmas)
for state_name, result in results.items():
    total_log_likelihoods += result['log_likelihoods']
max_likelihood_index = total_log_likelihoods.argmax()
sigma = sigmas[max_likelihood_index]
print('Chosen sigma:', sigma)
final_results = None
for state_name, result in results.items():
    print(state_name)
    posteriors = result['posteriors'][max_likelihood_index]
    hdis_90 = highest_density_interval(posteriors, p=.9)
    hdis_50 = highest_density_interval(posteriors, p=.5)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
    if final_results is None:
        final_results = result
    else:
        final_results = pd.concat([final_results, result])
print('Done.')
no_lockdown = [
    #'North Dakota', 'ND',
    #'South Dakota', 'SD',
    #'Nebraska', 'NB',
    #'Iowa', 'IA',
    #'Arkansas','AR'
]
partial_lockdown = [
    #'Utah', 'UT',
    #'Wyoming', 'WY',
    #'Oklahoma', 'OK',
    #'Massachusetts', 'MA'
]
final_results.round(3).to_pickle('rt_brazil.pickle')
#filtered = final_results.index.get_level_values(0).isin(FILTERED_REGIONS)
#mr = final_results.loc[~filtered].groupby(level=0)[['ML', 'High_90', 'Low_90']].last()
#mr.sort_values('ML', inplace=True)
#mr.sort_values('High_90', inplace=True)
#show = mr[mr.High_90.le(1)].sort_values('ML')
#show = mr[mr.Low_90.ge(1.0)].sort_values('Low_90')
#print('Finished')
