

import pandas as pd
import numpy as np
import pytest


from TechnicalAnalysis.identify_patterns import (
    # main,
    pull_prices,
    find_rolling_price_windows,
    rolling_window,
    # run_kernel_regressions,
    find_extrema_from_list_of_series,
    find_extrema,
    find_series_with_5_or_more_extrema,
    # plot_chart,
    plot_chart_plotly,

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
)


sequence_head_and_shoulders = [100, 90, 120, 90, 100, ]
sequence_inverse_head_and_shoulders = [100, 110, 80, 110, 100, ]
sequence_broadening_top = [100, 95, 110, 90, 120, ]
sequence_broadening_bottom = [100, 105, 90, 106, 80, ]
sequence_triangle_top = [100, 85, 95, 87, 90, ]
sequence_triangle_bottom = [100, 110, 105, 107, 106, ]
sequence_rectangle_top = [100, 75, 101, 76, 102, ]
sequence_rectangle_bottom = [0, 0, 0, 0, 0, ]
sequence_double_top = [0, 0, 0, 0, 0, ]
sequence_double_bottom = [0, 0, 0, 0, 0, ]


window_data = [
    {'extrema_y_vals': sequence_head_and_shoulders, },
    {'extrema_y_vals': sequence_inverse_head_and_shoulders, },
    {'extrema_y_vals': sequence_broadening_top, },
    {'extrema_y_vals': sequence_broadening_bottom, },
    {'extrema_y_vals': sequence_triangle_top, },
    {'extrema_y_vals': sequence_triangle_bottom, },
    {'extrema_y_vals': sequence_rectangle_top, },
]


def test_plot_chart_plotly():

    price_series_data = {
        'dates': [20200101, 20200102, 20200103, 20200104, 20200105],
        'closing_prices': sequence_broadening_bottom, }

    chart = plot_chart_plotly(price_series_data=price_series_data, )
    chart.show()


@pytest.mark.skip(reason='Test not written yet', )
def test_main() -> None:

    """
    TODO: Write test
    :return:
    """

    assert False


def test_pull_prices() -> None:

    prices = pull_prices(ticker='TSLA', period='1d')
    minimum_expected_columns = ['Close', 'Date', ]

    assert len(prices.index) > 0
    assert [i for i in minimum_expected_columns if i in list(prices.columns)] == minimum_expected_columns


def test_find_rolling_price_windows() -> None:

    test_data = pd.DataFrame({
        'Close': [i for i in range(0, 38 * 2)], 'Date': [i + 1 for i in range(0, 38 * 2, )], }, )

    window_price_data, window_date_data = find_rolling_price_windows(hist=test_data, )

    assert window_price_data.shape == (39, 38, )
    assert window_date_data.shape == (39, 38, )

    assert list(window_price_data[0]) == [i for i in range(0, 38, )]
    assert list(window_date_data[0]) == [i + 1 for i in range(0, 38, )]

    assert list(window_price_data[-1]) == [i for i in range(38, 38 * 2, )]
    assert list(window_date_data[-1]) == [i + 1 for i in range(38, 38 * 2, )]


def test_rolling_window() -> None:

    ary = np.array([1, 2, 3, 4, 5, ])

    window_array = rolling_window(a=ary, window_input=3, )

    expected_output = np.array(
        [[1, 2, 3, ],
         [2, 3, 4, ],
         [3, 4, 5, ], ])

    assert (window_array == expected_output).all()


@pytest.mark.skip(reason='Test not written yet', )
def test_run_kernel_regressions() -> None:

    """
    TODO: Write test
    :return:
    """

    assert False


def test_find_extrema_from_list_of_series() -> None:

    window_data = [
        {'kernel_regression_line': np.array([11, 12, 13, 14, 13, 12, 11, ], ),
         'dates': [20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107, ], }
    ]
    extrema = find_extrema_from_list_of_series(window_data=window_data, )

    assert len(extrema) == 1
    assert extrema[0]['extrema_x_vals'] == [3, ]
    assert extrema[0]['extrema_y_vals'] == [14, ]
    assert extrema[0]['extrema_dates'] == [20200104, ]
    assert (extrema[0]['kernel_regression_line'] == window_data[0]['kernel_regression_line']).all()
    assert extrema[0]['dates'] == window_data[0]['dates']


def test_find_extrema() -> None:

    ary = np.array([11, 12, 13, 14, 13, 12, 11, ], )

    extrema = find_extrema(regression_line=ary, )

    assert extrema[0] == 3


def test_find_series_with_5_or_more_extrema() -> None:

    window_data = [{'extrema_x_vals': [1, 2, 3, 4, 5, 6, ], }, {'extrema_x_vals': [1, 2, 3, 4, ], }]

    modified_window_data = find_series_with_5_or_more_extrema(window_data=window_data, )

    assert len(modified_window_data) == 1
    assert modified_window_data[0]['extrema_x_vals'] == [1, 2, 3, 4, 5, 6, ]


@pytest.mark.skip(reason='Test not written yet', )
def test_plot_chart() -> None:

    """
    TODO: Write test
    :return:
    """

    assert False


def test_find_head_and_shoulders() -> None:

    output_list = find_head_and_shoulders(window_data=window_data, )

    expected_return = [
        {'extrema_y_vals': sequence_head_and_shoulders, }, ]

    assert output_list == expected_return


def test_find_inverse_head_and_shoulders() -> None:

    output_list = find_inverse_head_and_shoulders(window_data=window_data, )

    expected_return = [
        {'extrema_y_vals': sequence_inverse_head_and_shoulders, }, ]

    assert output_list == expected_return


def test_find_broadening_top() -> None:

    output_list = find_broadening_top(window_data=window_data, )

    expected_return = [
        {'extrema_y_vals': sequence_broadening_top, }, ]

    assert output_list == expected_return


def test_find_broadening_bottom() -> None:

    output_list = find_broadening_bottom(window_data=window_data, )

    expected_return = [
        {'extrema_y_vals': sequence_broadening_bottom, }, ]

    assert output_list == expected_return


def test_find_triangle_top() -> None:

    output_list = find_triangle_top(window_data=window_data, )

    expected_return = [
        {'extrema_y_vals': sequence_triangle_top, }, ]

    assert output_list == expected_return


def test_find_triangle_bottom() -> None:

    output_list = find_triangle_bottom(window_data=window_data, )

    expected_return = [
        {'extrema_y_vals': sequence_triangle_bottom, }, ]

    assert output_list == expected_return


def test_find_rectangle_top() -> None:

    """
    TODO: Write positive test
    :return:
    """

    output_list = find_rectangle_top(window_data=window_data, )

    expected_return = [
        {'extrema_y_vals': sequence_rectangle_top, }, ]

    assert output_list == expected_return


def test_find_rectangle_bottom() -> None:

    """
    TODO: Write positive test
    :return:
    """

    output_list = find_rectangle_bottom(window_data=window_data, )

    expected_return = []

    assert output_list == expected_return


@pytest.mark.skip(message='Need to complete writing test')
def test_find_double_top() -> None:

    """
    TODO: Write positive test
    :return:
    """

    output_list = find_double_top(window_data=window_data, )

    expected_return = []

    assert output_list == expected_return


@pytest.mark.skip(message='Need to complete writing test')
def test_find_double_bottom() -> None:

    """
    TODO: Write positive test
    :return:
    """

    output_list = find_double_bottom(window_data=window_data, )

    expected_return = []

    assert output_list == expected_return
