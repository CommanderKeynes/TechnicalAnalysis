
import dash
import dash_core_components as dcc
import dash_html_components as html
from plotly.tools import mpl_to_plotly


from identify_patterns import (
    find_head_and_shoulders,
    find_inverse_head_and_shoulders,
    find_broadening_top,
    find_broadening_bottom,
    find_triangle_top,
    find_triangle_bottom,
    find_rectangle_top,
    find_rectangle_bottom,
    find_double_top,
    find_double_bottom,

    pull_prices,
    find_rolling_price_windows,
    run_kernel_regressions,
    find_extrema_from_list_of_series,
    find_series_with_5_or_more_extrema,
    plot_chart_plotly,
)


app = dash.Dash()


technical_analysis_patterns = {
    'Head and Shoulders': find_head_and_shoulders, 'Inverse Head and Shoulders': find_inverse_head_and_shoulders,
    'Broadening Top': find_broadening_top, 'Broadening Bottom': find_broadening_bottom,
    'Triangle Top': find_triangle_top, 'Triangle Bottom': find_triangle_bottom,
    'Rectangle Top': find_rectangle_top, 'Rectangle Bottom': find_rectangle_bottom,
    'Double Top': find_double_top, 'Double Bottom': find_double_bottom, }


list_of_patterns = list(technical_analysis_patterns.keys())


app.layout = html.Div([
    dcc.Input(id='ticker_input', type='text', placeholder="input ticker here", ),
    dcc.Dropdown(
        id='technical_pattern_dropdown',
        options=[{'label': i, 'value': i, } for i in list_of_patterns],
        value=list_of_patterns[0], ),
    html.Button(children='Run Technical Analysis', id='run_analysis_button', n_clicks=0, ),
    html.Div(children='', id='output_graph'),
    dcc.Loading(
        id="loading-2",
        children=[html.Div([html.Div(id="loading-output-2")])],
        type="circle",
        fullscreen=True,
    )
])


@app.callback(
    output=[dash.dependencies.Output('output_graph', 'children'),
            dash.dependencies.Output('loading-2', 'children', )],
    inputs=[dash.dependencies.Input('run_analysis_button', 'n_clicks'), ],
    state=[dash.dependencies.State('ticker_input', 'value'),
           dash.dependencies.State('technical_pattern_dropdown', 'value'), ], )
def update_output(n_clicks, ticker_input, technical_pattern_dropdown) -> list:

    if n_clicks == 0:
        return ['', '', ]

    prices = pull_prices(ticker=ticker_input, period='1y', )
    window_price_data, window_date_data = find_rolling_price_windows(hist=prices, )
    result_list = run_kernel_regressions(window_data=window_price_data, date_data=window_date_data, )

    result_list_2 = find_extrema_from_list_of_series(window_data=result_list, )

    window_data_greater_than_4_extrema = find_series_with_5_or_more_extrema(window_data=result_list_2, )

    data = technical_analysis_patterns[technical_pattern_dropdown](window_data_greater_than_4_extrema)
    matplotlib_graph = plot_chart_plotly(price_series_data=data[0], )

    return [dcc.Graph(id='matplotlib-graph', figure=matplotlib_graph, ), '', ]


if __name__ == '__main__':
    app.run_server(debug=True, )
