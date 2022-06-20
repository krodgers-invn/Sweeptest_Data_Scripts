import pandas as pd
import matplotlib.pyplot as plt
import base64
import struct
import numpy as np
from scipy.optimize import curve_fit
from sqlalchemy import create_engine
import seaborn as sns
import os
from tkinter import Tk, filedialog

__author__ = 'Stefon Shelton'

def file_dialog():
    root = Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilenames(parent=root, title='Choose a file')
    files = root.filename
    c = '/'
    index = [pos for pos, x in enumerate(files[0]) if x == c]
    path = files[0][:index[-1]+1]
    return files, path

def read_csv_or_jmp_files(files):
    #read csv or jmp files into test and measurement dataframes
    for f in files:
        print(f)
        # if input files are jmp format save as csv (must have JMP installed)
        if f[-3:] == "jmp":
            from win32com.client import Dispatch
            jmp = Dispatch("JMP.Application")
            doc = jmp.OpenDocument(f)
            doc.SaveAs(f[:-4] + ".csv")
            # convert variable to point to csv file
            f = f[:-4] + ".csv"
            isjmp = 1
        else:
            isjmp = 0

        # read test and measurement files
        if "measurement" in str(f):
            measurement = pd.read_csv(f)
        elif "test" in str(f):
            test = pd.read_csv(f)
        else:
            print('test or measurement not found')

        # delete csv if input files are jmp
        if isjmp == 1:
            # delete generated csv files
            os.remove(f)
        else:
            print('csv not removed')
    return test, measurement

def decode_qi_data(encoded_qi_data):
    # ouput is a panda dataframe with columns of test_id, time, I, Q
    try:
        qi_str = base64.b64decode(encoded_qi_data)
        returnvalue = [struct.unpack('<hh', qi_str[x:x + 4]) for x in range(0, len(qi_str), 4)]
    except:
        returnvalue =''
    # print(qi_str)
    return returnvalue

def iqdata_unwrap(decoded_qi_data):
    if len(decoded_qi_data) > 0:
        idata = np.array([i for q, i in decoded_qi_data])
        qdata = np.array([q for q, i in decoded_qi_data])
        i_q = idata + 1j*qdata
    else:
        i_q = 0
    return i_q

def gaussfitter(timetrace, magtrace):
    def gauss(x, A, mu, sigma):
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [np.max(magtrace), timetrace[np.argmax(magtrace)], .00005]
    coeff, var_matrix = curve_fit(gauss, timetrace, magtrace, p0=p0)
    betterxdata = np.linspace(np.amin(timetrace), np.amax(timetrace), 1000)
    fitdata = gauss(betterxdata, *coeff)
    return coeff, betterxdata[np.argmax(fitdata)], betterxdata, fitdata

def rangefinder(time_trace, mag_trace, time_window=[-2.0,-1.0], range_window=[-2.0,-1.0]):
    if time_window == [-2, -1]:
        approx_min_time = 2*float(range_window[0])/1000/343
        approx_max_time = 2*float(range_window[1])/1000/343
    else:
        approx_min_time = time_window[0]
        approx_max_time = time_window[1]

    index_window = np.where(np.logical_and(time_trace > approx_min_time, time_trace < approx_max_time))[0]

    try:
        peakindex = index_window[0] + np.argmax(mag_trace[index_window[0]:index_window[len(index_window)-1]+1])
        final_time_trace = time_trace[peakindex-3:peakindex+4]
        final_mag_trace = mag_trace[peakindex-3:peakindex+4]
        try:
            [A, mu, sigma], center, x, y = gaussfitter(final_time_trace, final_mag_trace)
        except:
            A = 0
            mu = 0
            sigma = 0
    except:
        A = 0
        mu = 0
        sigma = 0

    rangetime = mu-np.sqrt(-2.*sigma**2*np.log(.5))
    # if (rangetime > approx_min_time) and (rangetime < approx_max_time):
    range = 343.*rangetime*.5*1000
    intensity = A
    # else:
    #     range = 0
    #     intensity = 0

    return pd.Series({'fit_range':range, 'fit_intensity':intensity})

def gen_fixed_length_time_trace(meas_freq):
    time_trace = 8 / meas_freq * np.array(range(60))
    return time_trace

def gen_time_trace(measurement):
    time_trace = 8 / measurement['frequency'] * np.array(range(len(measurement['mag_trace'])))
    return time_trace

def gen_time_trace_itw(measurement):
    if measurement['frequency'] > 0 and not isinstance(measurement['mag_trace'], int):
        time_trace = 8 / measurement['frequency'] * np.array(range(len(measurement['mag_trace'])))
    else:
        time_trace = np.array([])
    return time_trace


def gen_var_time_trace_rx(data):
    # print(type(data))
    # print(data['rx_mag_trace'])
    # print(data['rx_res_frequency'])
    # time_trace = 8 / data['rx_res_frequency'] * np.array(range(len(data['rx_mag_trace'])))
    time_trace = 8 / data['rx_frequency'] * np.array(range(len(data['rx_mag_trace'])))
    return time_trace


def gen_var_time_trace_tx(data):
    # time_trace = 8 / data['tx_res_frequency'] * np.array(range(len(data['tx_mag_trace'])))
    time_trace = 8 / data['tx_frequency'] * np.array(range(len(data['tx_mag_trace'])))
    return time_trace


def plot_joined_iq_traces(joined, leg=False, leg_field_value='test_id', type='linear', xaxis='distance'):
    print('\rPlotting Traces', end='')
    if xaxis == 'distance':
        joined['distance_trace'] = joined['time_trace'] / 2. * 1000. * 343.
        x_field_val = 'distance_trace'

    plt.figure()
    c = ['g', 'b', 'y', 'c', 'm', 'k', 'r'] * 10000
    c_ind = 0
    clr = c[0]
    i = 0
    for index, val in joined.iterrows():
            if isinstance(val['mag_trace'], int):
                print('IQ trace missing')
            else:
                if i == 0:
                    prev_val = val['test_id']
                    i = i+1
                    if type == 'linear':
                        plt.plot(val['distance_trace'] * 1e3, val['mag_trace'],'-o', label=val[leg_field_value], color=clr, marker='')
                    elif type == 'semilog':
                        plt.semilogy(val['distance_trace'] * 1e3, val['mag_trace'],'-o', label=val[leg_field_value], color=clr, marker='')
                else:
                    if val['test_id'] == prev_val:
                        prev_val = val['test_id']
                        if type == 'linear':
                            plt.plot(val['distance_trace'] * 1e3, val['mag_trace'],'-o', label='_nolegned_', color=clr, marker='')
                        elif type == 'semilog':
                            plt.semilogy(val['distance_trace'] * 1e3, val['mag_trace'],'-o', label='_nolegned_', color=clr, marker='')
                    else:
                        c_ind = c_ind+1
                        clr = c[c_ind]
                        prev_val = val['test_id']
                        if type == 'linear':
                            plt.plot(val['distance_trace']*1e3, val['mag_trace'],'-o', label=val[leg_field_value], color = clr, marker='')
                        elif type == 'semilog':
                            plt.semilogy(val['distance_trace']*1e3, val['mag_trace'],'-o', label=val[leg_field_value], color = clr, marker='')
            # else:
            #     print('empty')
    if leg == True:
        plt.legend()
    else:
        pass
    plt.xlabel('distance, mm')
    plt.ylabel('mag iq trace, LSB')
    plt.grid()
    plt.show()


def plot_joined_iq_traces_itw(joined, leg=False, leg_field_value='unique_test_pointer', type='linear', plot_lrd=True):

    joined.loc[:, 'distance_trace'] = joined.loc[:, 'time_trace'] / 2. * 1000. * 343
    s1 = lambda x: (np.abs(x - 245)).argmin()
    s2 = lambda x: (np.abs(x - 415)).argmin()
    indexes = lambda x: [s1(x), s1(x) + 1, s1(x) + 2, s1(x) + 3, s2(x), s2(x) + 1, s2(x) + 2, s2(x) + 3]

    plt.figure()
    c = ['g', 'b', 'y', 'c', 'm', 'k', 'r'] * 10000
    c_ind = 0
    unique_list = np.unique(joined.loc[:, 'unique_test_pointer'].values)
    for pointer in unique_list:
        sub_joined = joined[joined.loc[:, 'unique_test_pointer'] == pointer]
        is_first = True
        for index, val in sub_joined.iterrows():
            if isinstance(val['mag_trace'], int):
                print('IQ trace missing \n')
                # print(val)
            elif len(val['time_trace']) == 0:
                print('time trace missing \n')
            else:
                if is_first:
                    legstring = val[leg_field_value]
                    is_first = False
                else:
                    legstring = '_none_'
                plt.plot(val['time_trace'] * 1e3, val['mag_trace'], label=legstring, color=c[c_ind])

                if plot_lrd:
                    try:
                        all_idx = indexes(val['distance_trace'])
                        x_lrd, y_lrd = val['time_trace'][all_idx]*1e3, val['mag_trace'][all_idx]
                    except:
                        x_lrd, y_lrd = [], []
                    plt.plot(x_lrd, y_lrd, 'ko')
        c_ind += 1

    if type == 'semilog':
        plt.yscale('log')
    if leg == True:
        plt.legend(fontsize=8)
    else:
        pass
    plt.xlabel('time, ms')
    plt.ylabel('mag iq trace, LSB')
    plt.grid()
    plt.show()

def plot_iq_traces(measurement, leg=False, type='linear'):
    plt.figure()
    c = ['g', 'b', 'y', 'c', 'm', 'k', 'r'] * 10000
    c_ind = 0
    clr = c[0]
    i = 0
    for index, val in measurement.iterrows():
            # print(val['mag_trace'])
            # print(val['mag_trace'].dtype)
            # print(len(val['mag_trace']))
            if isinstance(val['mag_trace'], int):
                print('IQ trace missing')
            else:
                if i == 0:
                    prev_val = val['test_id']
                    i = i+1
                    if type == 'linear':
                        plt.plot(val['time_trace'] * 1e3, val['mag_trace'], label=val['test_id'], color=clr)
                    elif type == 'semilog':
                        plt.semilogy(val['time_trace'] * 1e3, val['mag_trace'], label=val['test_id'], color=clr)
                else:
                    if val['test_id'] == prev_val:
                        prev_val = val['test_id']
                        if type == 'linear':
                            plt.plot(val['time_trace'] * 1e3, val['mag_trace'], label='_nolegned_', color=clr)
                        elif type == 'semilog':
                            plt.semilogy(val['time_trace'] * 1e3, val['mag_trace'], label='_nolegned_', color=clr)
                    else:
                        c_ind = c_ind+1
                        clr = c[c_ind]
                        prev_val = val['test_id']
                        if type == 'linear':
                            plt.plot(val['time_trace']*1e3, val['mag_trace'], label=val['test_id'], color = clr)
                        elif type == 'semilog':
                            plt.semilogy(val['time_trace']*1e3, val['mag_trace'], label=val['test_id'], color = clr)
            # else:
            #     print('empty')
    if leg == True:
        plt.legend()
    else:
        pass
    plt.xlabel('time, ms')
    plt.ylabel('mag iq trace, LSB')
    plt.grid()
    plt.show()

def plot_iq_traces_one_per_test(measurement, leg=False, type='linear'):
    plt.figure()
    c = [[n/255 for n in m] for m in [(128, 128, 128), (0, 255, 0), (255, 255, 0), (255, 0, 255), (77, 148, 255), (255, 128, 0),
                      (255, 0, 0), (0, 0, 204), (128, 0, 255), (128, 0, 0), (124, 82, 39), (0, 128, 128),
                      (128, 128, 0), (0, 255, 255), (0, 128, 0), (0, 255, 127)]]*10000
    c_ind = 0
    clr = c[0]
    prev_val = -1
    for index, val in measurement.iterrows():
        if val['test_id'] != prev_val:
            prev_val = val['test_id']
            if type == 'linear':
                plt.plot(val['time_trace']*1e3, val['mag_trace'], label=val['test_id'], color=clr)
            elif type == 'semilog':
                plt.semilogy(val['time_trace']*1e3, val['mag_trace'], label=val['test_id'], color=clr)
            c_ind = c_ind+1
            clr = c[c_ind]

    if leg == True:
        plt.legend()
    else:
        pass
    plt.xlabel('time, ms')
    plt.ylabel('mag iq trace, LSB')
    plt.show()


def plot_traces(measurement):
    plt.figure()
    # plt.plot(measurement['time_trace'][0], np.abs(measurement['iq_data'][0]))
    for i in range(0, 99, 10):
        print(np.abs(measurement['iq_data'][i]))
        # plt.plot(np.abs(measurement['iq_data'][i]), label=str(i))
    plt.legend()
    plt.show()

def calc_long_ringdown_magnitude(time_trace, mag_trace):
    distance_trace = time_trace / 2. * 1000. * 343.
    try:
        # print(distance_trace)
        # s1 = (np.abs(distance_trace-245)).argmin()
        # s2 = (np.abs(distance_trace-415)).argmin()
        s1 = (np.abs(distance_trace-245)).argmin()
        s2 = (np.abs(distance_trace-415)).argmin()

        indexes = [s1, s1+1, s1+2, s1+3, s2, s2+1, s2+2, s2+3]
        long_ringdown_mag = np.mean(mag_trace[indexes])
        # plt.semilogy(distance_trace, np.abs(mag_trace), color='red')
        # plt.semilogy(distance_trace[indexes], np.abs(mag_trace[indexes]), 'o', color='black')
        # # plt.semilogy(np.abs(mag_trace), color='red')
        # # plt.semilogy(indexes, np.abs(mag_trace[indexes]), 'o', color='black')
    except:
        long_ringdown_mag = np.nan
    return pd.Series({'long_ringdown_magnitude': long_ringdown_mag})

def calc_long_ringdown_magnitude_CH201(time_trace, mag_trace):
    distance_trace = time_trace / 2. * 1000. * 343.
    try:
        # print(distance_trace)
        s1 = (np.abs(distance_trace-686)).argmin()
        s2 = (np.abs(distance_trace-1375)).argmin()
        indexes = [s1, s1+1, s1+2, s1+3, s2, s2+1, s2+2, s2+3]
        long_ringdown_mag = np.mean(mag_trace[indexes])
        # plt.semilogy(distance_trace, np.abs(mag_trace), color='red')
        # plt.semilogy(distance_trace[indexes], np.abs(mag_trace[indexes]), 'o', color='black')
        # # plt.semilogy(np.abs(mag_trace), color='red')
        # # plt.semilogy(indexes, np.abs(mag_trace[indexes]), 'o', color='black')
    except:
        long_ringdown_mag = np.nan
    return pd.Series({'long_ringdown_magnitude': long_ringdown_mag})

def calc_long_ringdown_magnitude_CH201_FT(time_trace, mag_trace):
    distance_trace = time_trace / 2. * 1000. * 343.
    try:
        # print(distance_trace)
        # DESIRED: 1.51 ms = 260 mm
        s1 = (np.abs(distance_trace-262)).argmin()
        indexes = [s1, s1+1, s1+2, s1+3]
        long_ringdown_mag = np.mean(mag_trace[indexes])
        # plt.semilogy(distance_trace, np.abs(mag_trace), color='red')
        # plt.semilogy(distance_trace[indexes], np.abs(mag_trace[indexes]), 'o', color='black')
        # # plt.semilogy(np.abs(mag_trace), color='red')
        # # plt.semilogy(indexes, np.abs(mag_trace[indexes]), 'o', color='black')
    except:
        long_ringdown_mag = np.nan
    return pd.Series({'long_ringdown_magnitude': long_ringdown_mag})

def plot_long_ringdown(measurement):
        measurement['distance_trace'] = measurement['time_trace'] / 2. *1000.* 343

        s1 = lambda x: (np.abs(x - 245)).argmin()
        s2 = lambda x: (np.abs(x - 415)).argmin()
        indexes = lambda x: [x['s1'], x['s1'] + 1, x['s1'] + 2, x['s1'] + 3, x['s2'], x['s2'] + 1, x['s2'] + 2,
                       x['s2'] + 3]
        measurement['s1'] = measurement['distance_trace'].apply(s1)
        measurement['s2'] = measurement['distance_trace'].apply(s2)
        measurement['indexes'] = measurement.apply(indexes, axis=1)

        for i, row in measurement.iterrows():
            plt.semilogy(row['distance_trace'], row['mag_trace'], color='red')
            plt.semilogy(row['distance_trace'][row['indexes']], row['mag_trace'][row['indexes']], 'o', color='black')
        plt.show()

def plot_long_ringdown_CH201(measurement):
        measurement['distance_trace'] = measurement['time_trace'] / 2. *1000.* 343

        s1 = lambda x: (np.abs(x - 686)).argmin()
        s2 = lambda x: (np.abs(x - 1375)).argmin()
        indexes = lambda x: [x['s1'], x['s1'] + 1, x['s1'] + 2, x['s1'] + 3, x['s2'], x['s2'] + 1, x['s2'] + 2,
                       x['s2'] + 3]
        measurement['s1'] = measurement['distance_trace'].apply(s1)
        measurement['s2'] = measurement['distance_trace'].apply(s2)
        measurement['indexes'] = measurement.apply(indexes, axis=1)

        plt.figure()
        for i, row in measurement.iterrows():
            plt.semilogy(row['distance_trace'], row['mag_trace'], color='red')
            plt.semilogy(row['distance_trace'][row['indexes']], row['mag_trace'][row['indexes']], 'o', color='black')
        plt.show()


def plot_long_ringdown_CH201_FT(measurement):
    measurement['distance_trace'] = measurement['time_trace'] / 2. * 1000. * 343

    s1 = lambda x: (np.abs(x - 262)).argmin()
    indexes = lambda x: [x['s1'], x['s1'] + 1, x['s1'] + 2, x['s1'] + 3]
    measurement['s1'] = measurement['distance_trace'].apply(s1)
    measurement['indexes'] = measurement.apply(indexes, axis=1)

    plt.figure()
    for i, row in measurement.iterrows():
        plt.semilogy(row['distance_trace'], row['mag_trace'], color='red')
        plt.semilogy(row['distance_trace'][row['indexes']], row['mag_trace'][row['indexes']], 'o', color='black')
    # plt.yscale('linear')
    plt.show()


def execute_query(query, con,include_measurement = False):
    # executes the database query
    # sorts by test ID
    # optionally pulls the measurement table if include_measurement = True

    print('executing query \n')
    test = pd.read_sql_query(sql=query, con=con)
    test.sort_values(by='test_id', inplace=True)

    if include_measurement:
        # pull related measurements
        test_id_limits = [test['test_id'].min(), test['test_id'].max()]
        query = "SELECT * FROM measurement WHERE test_id BETWEEN " + str(test_id_limits[0]) + " AND " + str(
            test_id_limits[1])
        all_measurement = pd.read_sql_query(sql=query, con=con)
        measurement = all_measurement.loc[all_measurement['test_id'].isin(test['test_id'])].copy()
        measurement.sort_values(by='test_id', inplace=True)
    else:
        measurement = pd.DataFrame()
    print('Number of tests: '+str(len(test)))
    print('Number of measurements: ' + str(len(measurement)))
    return test, measurement


def execute_query_itw(query, con, include_measurement=False):
    # executes the database query
    # sorts by test ID
    # optionally pulls the measurement table if include_measurement = True
    print('executing query \n')
    test = pd.read_sql_query(sql=query, con=con)
    print('%d tests queried \n' % len(test))
    test['unique_test_pointer'] = test['CHECK 2D_BARCODE'] + "_" + test['Retest']
    # test.sort_values(by=['CHECK DATE', 'CHECK TIME'], inplace=True)

    if include_measurement:
        # pull related measurements
        barcode_list = list(np.unique(test['CHECK 2D_BARCODE'].values))

        measurement_df_list = []
        query_count = 0
        N_limit = 3500
        while len(barcode_list) > 0:
            barcode_list_str = str(tuple(barcode_list))
            if len(barcode_list) == 1:
                # reformat query string if only 1 barcode present
                barcode_list_str = barcode_list_str[:-2] + barcode_list_str[-1:]
                barcode_list = []
            elif len(barcode_list) < N_limit:
                barcode_list = []
            elif len(barcode_list) >= N_limit:
                # SQL query length limited; if very long, break into multiple queries
                barcode_list_str = str(tuple(barcode_list[0:N_limit]))
                del barcode_list[0:N_limit]

            query = "SELECT * FROM dbo.FT_Measurements WHERE dbo.FT_Measurements.[2DBARCODE_ID] IN " + barcode_list_str
            measurement_current = pd.read_sql_query(sql=query, con=con)
            query_count += 1
            print('query %d complete, %d parts remaining' % (query_count, len(barcode_list)))
            measurement_df_list.append(measurement_current)
        measurement = pd.concat(measurement_df_list)


        # if n_barcode == 1:
        #     # re-format query string if only 1 barcode present
        #     barcode_list_str = barcode_list_str[:-2]+barcode_list_str[-1:]
        #
        # if len(barcode_list) > 3500:
        #     measurement_df_list = []
        #     for ii in range(len(barcode_list) // 3500):
        #         barcode_list_short = barcode_list[3500 * ii: min(len(barcode_list), 3500 * (ii + 1))]
        #         query = "SELECT * FROM dbo.FT_Measurements WHERE dbo.FT_Measurements.[2DBARCODE_ID] IN " + barcode_list_short
        #         measurement_ii = pd.read_sql_query(sql=query, con=con)
        #         measurement_df_list.append(measurement_ii)
        # else:
        #     query = "SELECT * FROM dbo.FT_Measurements WHERE dbo.FT_Measurements.[2DBARCODE_ID] IN " + barcode_list_str
        #     pd.read_sql_query(sql=query, con=con)
        #
        #     query = query.replace(' TOP (%d)' % len(test), '')
        #     query = query.replace('FT_Tests', 'FT_Measurements')
        # print(query)
        # print('querying measurements \n')
        # measurement = pd.read_sql_query(sql=query, con=con)
        print('measurements queried: %d \n' %len(measurement))
        measurement['unique_test_pointer'] = measurement['2DBARCODE_ID'] + "_" + measurement['Retest']
        # print(measurement)
        measurement = measurement[measurement['unique_test_pointer'].isin(test['unique_test_pointer'])]
        # measurement = all_measurement.loc[all_measurement['test_id'].isin(test['test_id'])].copy()
        # measurement.sort_values(by='test_id', inplace=True)
    else:
        measurement = pd.DataFrame()
    print('Number of tests: '+str(len(test)))
    print('Number of measurements: ' + str(len(measurement)))
    return test, measurement

def percent_error_analysis(test, measurement):
    #filter for % bin 6
    test_summary = pd.crosstab(test['lot_num+dataset_time'],test['hard_bin_num'])
    test_summary['total_num'] = test_summary.sum(axis=1, numeric_only=True)
    test_summary['% error'] = test_summary[6]/test_summary['total_num']*100
    print(test_summary.columns)
    print(test_summary.head(5))
    # print(test_summary[6])

    ax = sns.barplot(data=test_summary, x=test_summary.index, y='% error')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()

if __name__ =='__main__':
    pass