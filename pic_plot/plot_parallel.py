import plotly.graph_objects as go

def parallel(data):
    fig = go.Figure(data=
        go.Parcoords(
            line_color='blue',
            tickfont=dict(size=15),
            dimensions = list([
                dict(range = [min(data[0]),max(data[0])],
                     label = '胶凝材料用量', values = data[0].values),
                dict(range = [min(data[1]),max(data[1])],
                     label = '水泥掺量', values = data[1].values),
                dict(range = [min(data[2]),max(data[2])],
                     label = '粉煤灰掺量', values = data[2].values),
                dict(range=[min(data[3]), max(data[3])],
                     label='矿粉掺量', values=data[3].values),
                dict(range=[min(data[4]), max(data[4])],
                     label='水胶比', values=data[4].values),
                dict(range=[min(data[5]), max(data[5])],
                     label='砂率', values=data[5].values),
                dict(range=[min(data[6]), max(data[6])],
                     label='容重', values=data[6].values),
                dict(range=[min(data[7]), max(data[7])],
                     label='水泥28d强度', values=data[7].values),
                dict(range=[min(data['COST']), max(data["COST"])],
                     label='成本', values=data['COST'].values),
                dict(range=[min(data['UCS']), max(data["UCS"])],
                     #constraintrange = [54.5, 54.7], # change this range by dragging the pink line
                     label='强度', values=data['UCS'].values),
            ])
        )
    )
    return fig