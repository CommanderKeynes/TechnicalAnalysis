

import numpy as np
import yfinance as yf
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg as kr
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def main():

    prices = pull_prices(ticker='MSFT', period='1y', )
    window_price_data, window_date_data = find_rolling_price_windows(hist=prices, )
    result_list = run_kernel_regressions(window_data=window_price_data, date_data=window_date_data, )

    result_list_2 = find_extrema_from_list_of_series(window_data=result_list, )

    window_data_greater_than_4_extrema = find_series_with_5_or_more_extrema(window_data=result_list_2, )

    window_data_head_and_shoulders = find_head_and_shoulders(
        window_data=window_data_greater_than_4_extrema, )

    plot_chart(price_series_data=window_data_head_and_shoulders[0], )

    window_data_inverse_head_and_shoulders = find_inverse_head_and_shoulders(window_data=window_data_greater_than_4_extrema)

    plot_chart(price_series_data=window_data_inverse_head_and_shoulders[0], )

    window_data_broadening_top = find_broadening_top(window_data=window_data_greater_than_4_extrema)

    plot_chart(price_series_data=window_data_broadening_top[0], )

    window_data_broadening_bottom = find_broadening_bottom(window_data=window_data_greater_than_4_extrema)

    plot_chart(price_series_data=window_data_broadening_bottom[0], )

    window_data_triangle_top = find_triangle_top(
        window_data=window_data_greater_than_4_extrema)

    plot_chart(price_series_data=window_data_triangle_top[0], )

    window_data_triangle_top = find_triangle_bottom(
        window_data=window_data_greater_than_4_extrema)

    plot_chart(price_series_data=window_data_triangle_top[0], )

    window_data_triangle_top = find_rectangle_top(
        window_data=window_data_greater_than_4_extrema)

    plot_chart(price_series_data=window_data_triangle_top[0], )

    window_data_triangle_top = find_rectangle_bottom(
        window_data=window_data_greater_than_4_extrema)

    plot_chart(price_series_data=window_data_triangle_top[0], )

    window_data_double_top = find_double_top(
        window_data=result_list_2)

    plot_chart(price_series_data=window_data_double_top[0], )

    window_data_double_bottom = find_double_bottom(
        window_data=result_list_2)

    plot_chart(price_series_data=window_data_double_bottom[0], )


def pull_prices(ticker: str, period: str) -> pd.DataFrame:

    ticker = yf.Ticker(ticker=ticker, )

    hist = ticker.history(period=period, )

    hist = hist.reset_index()

    return hist


def find_rolling_price_windows(hist: pd.DataFrame):

    window = 38

    window_price_data = rolling_window(hist['Close'].values, window)
    window_date_data = rolling_window(hist['Date'].values, window)

    return window_price_data, window_date_data


def rolling_window(a: np.array, window_input: int, ) -> np.array:

    shape = a.shape[:-1] + (a.shape[-1] - window_input + 1, window_input)
    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def run_kernel_regressions(window_data: np.array, date_data: np.array, ) -> list:

    result_list = []

    for i in range(0, len(window_data)):

        temp_window_data = dict()
        temp_window_data['closing_prices'] = window_data[i]
        temp_window_data['dates'] = date_data[i]

        exog = [i for i in range(0, len(temp_window_data['closing_prices']))]

        regression_result = kr(endog=temp_window_data['closing_prices'], exog=exog, var_type='c', )
        regression_result = regression_result.fit()[0]

        temp_window_data['kernel_regression_line'] = regression_result

        result_list = result_list + [temp_window_data, ]

    return result_list


def find_extrema_from_list_of_series(window_data: list, ) -> list:

    result_list = []
    for i in range(0, len(window_data)):
        temp_window_data = window_data[i]

        temp_window_data['extrema_x_vals'] = find_extrema(
            regression_line=temp_window_data['kernel_regression_line'], )

        temp_window_data['extrema_y_vals'] = [
            temp_window_data['kernel_regression_line'][i]
            for i in temp_window_data['extrema_x_vals']]

        temp_window_data['extrema_dates'] = [
            temp_window_data['dates'][i]
            for i in temp_window_data['extrema_x_vals']]

        result_list = result_list + [temp_window_data, ]

    return result_list


def find_extrema(regression_line) -> list:
    local_maxima_coords = list(argrelextrema(regression_line, np.greater, )[0])

    local_minima_coords = list(argrelextrema(regression_line, np.less, )[0])

    local_extrema_coords = local_minima_coords + local_maxima_coords

    local_extrema_coords.sort()

    return local_extrema_coords


def find_series_with_5_or_more_extrema(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        if len(window_data[i]['extrema_x_vals']) >= 5:
            result_list = result_list + [window_data[i], ]

    return result_list


def plot_chart(price_series_data: dict) -> None:
    plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots()

    y_actual = price_series_data['closing_prices']
    x = np.linspace(0, len(y_actual), len(y_actual))
    ax.plot(price_series_data['dates'], y_actual, 'o', color='green');

    y_regressed = price_series_data['kernel_regression_line']
    ax.plot(price_series_data['dates'], y_regressed, '-o', color='blue');

    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    # TODO: Plot extrema as x's
    y_extrema = price_series_data['extrema_y_vals']
    x_extrema = price_series_data['extrema_dates']
    plt.plot(x_extrema, y_extrema, 'x', color='purple')


def find_head_and_shoulders(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']

        e1_is_local_maxima = extrema_y_vals[0] > extrema_y_vals[1]
        e3_is_greater_than_e1 = extrema_y_vals[2] > extrema_y_vals[0]
        e3_is_greater_than_e5 = extrema_y_vals[2] > extrema_y_vals[4]

        e2_e4_avg = (extrema_y_vals[1] + extrema_y_vals[3]) / 2
        e2_within_15b_of_e2_e4_avg = abs(extrema_y_vals[1] / e2_e4_avg - 1) <= 0.015
        e4_within_15b_of_e2_e4_avg = abs(extrema_y_vals[3] / e2_e4_avg - 1) <= 0.015
        e2_and_e4_within_15_basis_points_of_their_average = e4_within_15b_of_e2_e4_avg and e2_within_15b_of_e2_e4_avg

        e1_e5_avg = (extrema_y_vals[0] + extrema_y_vals[4]) / 2
        e1_within_15b_of_e1_e5_avg = abs(extrema_y_vals[0] / e1_e5_avg - 1) <= 0.015
        e5_within_15b_of_e1_e5_avg = abs(extrema_y_vals[4] / e1_e5_avg - 1) <= 0.015
        e1_and_e5_within_15_basis_points_of_their_average = e1_within_15b_of_e1_e5_avg and e5_within_15b_of_e1_e5_avg

        price_series_has_head_and_shoulders = (
                e1_is_local_maxima and e3_is_greater_than_e1 and e3_is_greater_than_e5
                and e2_and_e4_within_15_basis_points_of_their_average
                and e1_and_e5_within_15_basis_points_of_their_average)

        if price_series_has_head_and_shoulders:
            result_list = result_list + [window_data[i], ]

    return result_list


def find_inverse_head_and_shoulders(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']

        e1_is_local_minima = extrema_y_vals[0] < extrema_y_vals[1]
        e3_is_greater_than_e1 = extrema_y_vals[2] < extrema_y_vals[0]
        e3_is_greater_than_e5 = extrema_y_vals[2] < extrema_y_vals[4]

        e2_e4_avg = (extrema_y_vals[1] + extrema_y_vals[3]) / 2
        e2_within_15b_of_e2_e4_avg = abs(extrema_y_vals[1] / e2_e4_avg - 1) <= 0.015
        e4_within_15b_of_e2_e4_avg = abs(extrema_y_vals[3] / e2_e4_avg - 1) <= 0.015
        e2_and_e4_within_15_basis_points_of_their_average = e4_within_15b_of_e2_e4_avg and e2_within_15b_of_e2_e4_avg

        e1_e5_avg = (extrema_y_vals[0] + extrema_y_vals[4]) / 2
        e1_within_15b_of_e1_e5_avg = abs(extrema_y_vals[0] / e2_e4_avg - 1) <= 0.015
        e5_within_15b_of_e1_e5_avg = abs(extrema_y_vals[4] / e2_e4_avg - 1) <= 0.015
        e1_and_e5_within_15_basis_points_of_their_average = e1_within_15b_of_e1_e5_avg and e5_within_15b_of_e1_e5_avg

        price_series_has_head_and_shoulders = (
                e1_is_local_minima and e3_is_greater_than_e1 and e3_is_greater_than_e5
                and e2_and_e4_within_15_basis_points_of_their_average and e1_and_e5_within_15_basis_points_of_their_average)

        if price_series_has_head_and_shoulders:
            result_list = result_list + [window_data[i], ]

    return result_list


def find_broadening_top(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']

        e1_is_local_maxima = extrema_y_vals[0] > extrema_y_vals[1]
        cond1 = extrema_y_vals[0] < extrema_y_vals[2]
        e3_is_greater_than_e1 = extrema_y_vals[2] < extrema_y_vals[4]
        e3_is_greater_than_e5 = extrema_y_vals[1] > extrema_y_vals[3]

        price_series_has_head_and_shoulders = (
                cond1 and e1_is_local_maxima and e3_is_greater_than_e1 and e3_is_greater_than_e5)

        if price_series_has_head_and_shoulders:
            result_list = result_list + [window_data[i], ]

    return result_list


def find_broadening_bottom(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']

        e1_is_local_maxima = extrema_y_vals[0] < extrema_y_vals[1]
        cond1 = extrema_y_vals[0] > extrema_y_vals[2]
        e3_is_greater_than_e1 = extrema_y_vals[2] > extrema_y_vals[4]
        e3_is_greater_than_e5 = extrema_y_vals[1] < extrema_y_vals[3]

        price_series_has_head_and_shoulders = (
                cond1 and e1_is_local_maxima and e3_is_greater_than_e1 and e3_is_greater_than_e5)

        if price_series_has_head_and_shoulders:
            result_list = result_list + [window_data[i], ]

    return result_list


def find_triangle_top(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']

        e1_is_local_maxima = extrema_y_vals[0] > extrema_y_vals[1]
        cond1 = extrema_y_vals[0] > extrema_y_vals[2]
        e3_is_greater_than_e1 = extrema_y_vals[2] > extrema_y_vals[4]
        e3_is_greater_than_e5 = extrema_y_vals[1] < extrema_y_vals[3]

        price_series_has_head_and_shoulders = (
                cond1 and e1_is_local_maxima and e3_is_greater_than_e1 and e3_is_greater_than_e5)

        if price_series_has_head_and_shoulders:
            result_list = result_list + [window_data[i], ]

    return result_list


def find_triangle_bottom(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']

        e1_is_local_maxima = extrema_y_vals[0] < extrema_y_vals[1]
        cond1 = extrema_y_vals[0] < extrema_y_vals[2]
        e3_is_greater_than_e1 = extrema_y_vals[2] < extrema_y_vals[4]
        e3_is_greater_than_e5 = extrema_y_vals[1] > extrema_y_vals[3]

        price_series_has_head_and_shoulders = (
                cond1 and e1_is_local_maxima and e3_is_greater_than_e1 and e3_is_greater_than_e5)

        if price_series_has_head_and_shoulders:
            result_list = result_list + [window_data[i], ]

    return result_list


# TODO: No examples found in price series,
# TODO: Need to make unit test or find another price series

def find_rectangle_top(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']

        e1_is_local_maxima = extrema_y_vals[0] > extrema_y_vals[1]

        tops_list = extrema_y_vals[::2]
        bottoms_list = extrema_y_vals[1::2]

        avg_tops = sum(tops_list) / len(tops_list)
        avg_bottoms = sum(bottoms_list) / len(bottoms_list)

        avg_tops_threshold_breached = [
            i for i in tops_list if abs(i / avg_tops) >= .75]
        avg_bottoms_threshold_breached = [
            i for i in bottoms_list if abs(i / avg_bottoms) >= .75]

        any_bottoms_breached_theshold = len(avg_tops_threshold_breached) == 0
        any_tops_breached_theshold = len(avg_tops_threshold_breached) == 0

        lowest_top_greater_than_highest_bottom = max(bottoms_list) < min(tops_list)

        price_series_has_head_and_shoulders = (
                e1_is_local_maxima and any_bottoms_breached_theshold
                and any_tops_breached_theshold
                and lowest_top_greater_than_highest_bottom)

        if price_series_has_head_and_shoulders:
            result_list = result_list + [window_data[i], ]

    return result_list


# TODO: No examples found in price series,
# TODO: Need to make unit test or find another price series

def find_rectangle_bottom(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']

        e1_is_local_maxima = extrema_y_vals[0] < extrema_y_vals[1]

        bottoms_list = extrema_y_vals[::2]
        tops_list = extrema_y_vals[1::2]

        avg_tops = sum(tops_list) / len(tops_list)
        avg_bottoms = sum(bottoms_list) / len(bottoms_list)

        avg_tops_threshold_breached = [
            i for i in tops_list if abs(i / avg_tops) >= .75]
        avg_bottoms_threshold_breached = [
            i for i in bottoms_list if abs(i / avg_bottoms) >= .75]

        any_bottoms_breached_theshold = len(avg_tops_threshold_breached) == 0
        any_tops_breached_theshold = len(avg_tops_threshold_breached) == 0

        lowest_top_greater_than_highest_bottom = max(bottoms_list) < min(tops_list)

        price_series_has_head_and_shoulders = (
                e1_is_local_maxima and any_bottoms_breached_theshold
                and any_tops_breached_theshold
                and lowest_top_greater_than_highest_bottom)

        if price_series_has_head_and_shoulders:
            result_list = result_list + [window_data[i], ]

    return result_list


# TODO: No examples found in price series,
# TODO: Need to make unit test or find another price series

def find_double_top(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']
        extrema_x_vals = window_data[i]['extrema_x_vals']

        e1_is_local_maxima = extrema_y_vals[0] > extrema_y_vals[1]

        e1_x = extrema_y_vals[0]
        e1_y = extrema_x_vals[0]

        local_maxima_ys_list = extrema_y_vals[::2]
        local_maxima_xs_list = extrema_x_vals[::2]

        local_maxima_after_initial = [
            {'y': local_maxima_ys_list[i], 'x': local_maxima_xs_list[i], }
            for i in range(0, len(local_maxima_ys_list))
            if local_maxima_xs_list[i] - e1_x > 22
               and abs((local_maxima_ys_list[i] / (
                    (local_maxima_ys_list[i] + e1_y) / 2) - 1)) < 0.015
               and abs((e1_y / (
                    (local_maxima_ys_list[i] + e1_y) / 2) - 1)) < 0.015
               and max(local_maxima_ys_list[0:i]) < local_maxima_ys_list[i]
        ]

        price_series_has_double_top = len(local_maxima_after_initial) > 0

        if price_series_has_double_top:
            result_list = result_list + [window_data[i], ]

    return result_list


# TODO: No examples found in price series,
# TODO: Need to make unit test or find another price series

def find_double_bottom(window_data: list, ) -> list:
    result_list = []

    for i in range(0, len(window_data)):

        # TODO: test for head and shoulders E1 is max, E1<E3>E5
        extrema_y_vals = window_data[i]['extrema_y_vals']
        extrema_x_vals = window_data[i]['extrema_x_vals']

        e1_is_local_maxima = extrema_y_vals[0] < extrema_y_vals[1]

        e1_x = extrema_y_vals[0]
        e1_y = extrema_x_vals[0]

        local_maxima_ys_list = extrema_y_vals[::2]
        local_maxima_xs_list = extrema_x_vals[::2]

        local_maxima_after_initial = [
            {'y': local_maxima_ys_list[i], 'x': local_maxima_xs_list[i], }
            for i in range(0, len(local_maxima_ys_list))
            if local_maxima_xs_list[i] - e1_x > 22
               and abs((local_maxima_ys_list[i] / (
                    (local_maxima_ys_list[i] + e1_y) / 2) - 1)) < 0.015
               and abs((e1_y / (
                    (local_maxima_ys_list[i] + e1_y) / 2) - 1)) < 0.015
               and min(local_maxima_ys_list[0:i]) < local_maxima_ys_list[i]
        ]

        price_series_has_double_top = len(local_maxima_after_initial) > 0

        if price_series_has_double_top:
            result_list = result_list + [window_data[i], ]

    return result_list


if __name__ == '__main__':
    main()
