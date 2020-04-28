import flask
import flask_cors
import pandas as pd

webapi = flask.Flask(__name__)
webapi.config['DEBUG'] = True

flask_cors.CORS(webapi)

@webapi.route('/last_update')
def last_update():
    df = pd.read_pickle('rt_brazil.pickle')
    return flask.jsonify(df.index[-1][1].strftime('%Y-%m-%d'))

@webapi.route('/list_states')
def list_states():
    df = pd.read_pickle('rt_brazil.pickle')
    return flask.jsonify(df.index.get_level_values('state').unique().tolist())

@webapi.route('/last_rt')
def last_rt():
    df = pd.read_pickle('rt_brazil.pickle')
    data = df.groupby('state').last()
    return flask.jsonify([data.index.tolist(), data.ML.tolist(), data[['Low_50', 'High_50']].values.tolist()])

@webapi.route('/rt_ts/<state>')
def rt_ts(state):
    df = pd.read_pickle('rt_brazil.pickle')
    x = df.groupby('state').get_group(state).droplevel(0)
    y = (x.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
    return flask.jsonify([list(zip(y, x.ML)), list(zip(y, x.Low_50, x.High_50))])

if __name__ == '__main__':
    webapi.run(port=7777)
