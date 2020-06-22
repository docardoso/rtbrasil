import flask
import flask_cors
import pandas as pd

webapi = flask.Flask(__name__)
webapi.config['DEBUG'] = True

flask_cors.CORS(webapi)

@webapi.route('/last_update')
def last_update():
    df = pd.read_pickle('rt_brazil.pickle')
    return flask.jsonify(df.index.get_level_values('date').max().strftime('%Y-%m-%d'))

@webapi.route('/list_states')
def list_states():
    df = pd.read_pickle('rt_brazil.pickle')
    #return flask.jsonify(df.index.get_level_values('region').unique().tolist())
    return flask.jsonify(df.index.get_level_values('state').unique().tolist())

@webapi.route('/list_cities/<state>')
def list_cities(state):
    df = pd.read_pickle('rt_brazil.pickle')
    x = df.groupby('state').get_group(state)
    x = x.index.get_level_values('city').unique()
    x = x[x != ''].tolist()
    return flask.jsonify(x)

@webapi.route('/last_rt')
def last_rt():
    df = pd.read_pickle('rt_brazil.pickle')
    #data = df.groupby('region').last()
    data = df[df.index.get_level_values('city') == ''].groupby('state').last()
    return flask.jsonify([data.index.tolist(), data['mean'].tolist(), data[['lower_90', 'upper_90']].values.tolist()])

#@webapi.route('/rt_ts/<state>')
#def rt_ts(state):
#    df = pd.read_pickle('rt_brazil.pickle')
#    #x = df.groupby('region').get_group(state).droplevel(0)
#    x = df.groupby(['state', 'city']).get_group((state, '')).droplevel([0, 1])
#    y = (x.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
#    return flask.jsonify([list(zip(y, x['mean'])), list(zip(y, x.lower_90, x.upper_90))])

@webapi.route('/rt_ts/<state>/')
@webapi.route('/rt_ts/<state>/<city>')
def rt_ts(state, city=''):
    df = pd.read_pickle('rt_brazil.pickle')
    #x = df.groupby('region').get_group(state).droplevel(0)
    x = df.groupby(['state', 'city']).get_group((state, city)).droplevel([0, 1])
    y = (x.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
    return flask.jsonify([list(zip(y, x['mean'])), list(zip(y, x.lower_90, x.upper_90))])

@webapi.route('/last_cases_deaths')
def last_cases_deaths():
    df = pd.read_pickle('cases_deaths_brazil.pickle')
    data = df[df.index.get_level_values('city') == ''].groupby('state').last()
    print(data)
    return flask.jsonify([data.index.tolist(), data[['positive', 'death']].values.tolist()])

@webapi.route('/cases_deaths_ts/<state>')
@webapi.route('/cases_deaths_ts/<state>/<city>')
def cases_deaths_ts(state, city=''):
    df = pd.read_pickle('cases_deaths_brazil.pickle')
    x = df.groupby(['state', 'city']).get_group((state, city)).droplevel([0, 1])
    y = (x.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
    return flask.jsonify([list(zip(y, x.positive)), list(zip(y, x.death))])

if __name__ == '__main__':
    webapi.run(port=7777)
