import plotly.express as px


def plot_sacatter(data):
    fig = px.scatter(data, x='UCS', y='COST')
    return fig