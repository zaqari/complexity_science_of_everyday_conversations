import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_symetric_powerlaw_plot(
    data: list,
    data_color: list=[],
    data_names: list=[],
    power_exponent_range: list=[-2,2],
    main_plot_scale: int=10,
    divisions_per_side_of_zero: int=50,
    background_color_scale: str|list[list]='pubugn',
    mean_indicator_line_width: int=5,
    show_legend: bool=True,
):
    data_color_swatches = np.random.choice(px.colors.qualitative.Set3, size=(len(data),))

    fig = make_subplots(
        rows=len(data) + 1,
        cols=1,
        row_heights=[.8] + ([.15] * len(data)),
        shared_xaxes='all',
        # shared_yaxes='all'
    )

    for i, d in enumerate(data):
        name = str(i) + '-data'
        legend_group = str(i)
        color = data_color_swatches[i]

        if (i < len(data_names)):
            name = data_names[i] + '-data'
            legend_group = data_names[i]

        if (i < len(data_color)):
            color = data_color[i]

        fig.add_trace(
            go.Scatter(
                x=d,
                y=np.zeros_like(d),
                mode='markers',
                marker=dict(
                    symbol='line-ns',
                    size=15,
                    color=color,
                    line=dict(
                        width=1,
                        color=color
                    )
                ),
                name=name,
                legendgroup=legend_group,
                showlegend=False,
            ),
            row=i + 2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[data[i].mean().item(), data[i].mean().item(), data[i].mean().item()],
                y=[main_plot_scale, 0, -main_plot_scale],
                mode='lines',
                name=name.replace('-data', ''),
                legendgroup=legend_group,
                showlegend=show_legend,
                line=dict(
                    color=color,
                    # dash='dash'
                    width=5,
                ),
            ),
            row=1, col=1
        )

    if isinstance(background_color_scale, str):
        if background_color_scale.endswith('_r'):
            right = px.colors.sample_colorscale(
                colorscale=background_color_scale,
                samplepoints=int(divisions_per_side_of_zero),
                low=0.0,
                high=1.0
            )
            left = px.colors.sample_colorscale(
                colorscale=background_color_scale.replace('_r', ''),
                samplepoints=int(divisions_per_side_of_zero),
                low=0.0,
                high=1.0
            )
        else:
            left = px.colors.sample_colorscale(
                colorscale=background_color_scale + '_r',
                samplepoints=int(divisions_per_side_of_zero),
                low=0.0,
                high=1.0
            )
            right = px.colors.sample_colorscale(
                colorscale=background_color_scale,
                samplepoints=int(divisions_per_side_of_zero),
                low=0.0,
                high=1.0
            )
    else:
        left = px.colors.sample_colorscale(
            colorscale=background_color_scale[0],
            samplepoints=int(divisions_per_side_of_zero),
            low=0.0,
            high=1.0
        )
        right = px.colors.sample_colorscale(
            colorscale=background_color_scale[1],
            samplepoints=int(divisions_per_side_of_zero),
            low=0.0,
            high=1.0
        )

    color_transitions = left + right

    xpoints = np.linspace(start=power_exponent_range[0], stop=power_exponent_range[1], num=len(color_transitions))
    xtransitions = list(zip(xpoints[:-1], xpoints[1:])) #+ [(xpoints[-1], None)]

    for i, transition in enumerate(xtransitions):
        fig.add_vrect(
            x0=transition[0],
            x1=transition[1],
            opacity=.75,
            layer='below',
            fillcolor=color_transitions[i],
            line_width=0,
            row=1,
            col=1
        )

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(range=[power_exponent_range[0], power_exponent_range[1]])
    fig.update_yaxes(
        range=[-main_plot_scale, main_plot_scale],
        row=1, col=1
    )

    fig.update_layout(
        template='ygridoff',
    )

    return fig

def create_noise_plot(
        data: list,
        data_color: list=[],
        data_names: list=[],
        sound_range: list=[-.5,.5],
        main_plot_scale: int=10,
        color_divisions_per_side_of_zero: int=50
):
    data_color_swatches = np.random.choice(px.colors.qualitative.Set3,size=(len(data),))

    fig = make_subplots(
        rows=len(data)+1,
        cols=1,
        row_heights=[.8] + ([.15] * len(data)),
        shared_xaxes='all',
        # shared_yaxes='all'
    )

    for i,d in enumerate(data):
        name = str(i) + '-data'
        legend_group = str(i)
        color=data_color_swatches[i]

        if (i < len(data_names)):
            name = data_names[i] + '-data'
            legend_group = data_names[i]

        if (i < len(data_color)):
            color = data_color[i]

        fig.add_trace(
            go.Scatter(
                x=d,
                y=np.zeros_like(d),
                mode='markers',
                marker=dict(
                    symbol='line-ns',
                    size=15,
                    color=color,
                    line=dict(
                        width=1,
                        color=color
                    )
                ),
                name=name,
                legendgroup=legend_group,
                showlegend=False,
            ),
            row=i+2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[data[i].mean().item(), data[i].mean().item(), data[i].mean().item()],
                y=[main_plot_scale,0,-main_plot_scale],
                mode='lines',
                name=name.replace('-data', ''),
                legendgroup=legend_group,
                showlegend=True,
                line=dict(
                    color=color,
                    # dash='dash'
                    width=5,
                ),
            ),
            row=1, col=1
        )

    white_pink = px.colors.sample_colorscale(
        # colorscale=['rgb(255,255,255)', 'rgb(255, 192, 203)'],
        colorscale=['rgb(255,255,255)', 'rgb(255, 102, 128)'],
        samplepoints=int(color_divisions_per_side_of_zero),
        low=0.0,
        high=1.0
    )

    pink_brown = px.colors.sample_colorscale(
        # colorscale=['rgb(255, 192, 203)','rgb(165,42,42)'],
        colorscale=['rgb(255, 102, 128)','rgb(165,42,42)'],
        samplepoints=int(color_divisions_per_side_of_zero/2),
        low=0.0,
        high=1.0
    )

    brown_blue = px.colors.sample_colorscale(
        # colorscale=['rgb(165, 42, 42)', 'rgb(0, 0, 255)'],
        colorscale=['rgb(165, 42, 42)', 'rgb(77, 77, 255)'],
        samplepoints=int(color_divisions_per_side_of_zero/2),
        low=0.0,
        high=1.0
    )

    color_transitions = white_pink[:-1] + pink_brown + brown_blue

    xpoints = np.linspace(start=sound_range[0], stop=1, num=len(color_transitions))
    xtransitions = list(zip(xpoints[:-1],xpoints[1:])) + [(xpoints[-1],1)]

    for i,transition in enumerate(xtransitions):
        fig.add_vrect(
            x0=transition[0],
            x1=transition[1],
            opacity=.75,
            layer='below',
            fillcolor=color_transitions[i],
            line_width=0,
            row=1,
            col=1
        )

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(range=[-1,1])
    fig.update_yaxes(
        range=[main_plot_scale,-main_plot_scale],
        row=1, col=1
    )

    fig.update_layout(
        template='ygridoff',
    )

    return fig