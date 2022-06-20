from prod_test_functions import *
# import pandas as pd
# import numpy as np
from datetime import datetime
# from tkinter import Tk, filedialog

# Store date and time for distinguishing filename later in script
now = datetime.now()  # current date and time
time_date = now.strftime("%H%M%m%d")
print("date and time:", time_date)


def file_dialog_jnb():
    root = Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)  # Raise the root to the top of all windows.
    root.filename = filedialog.askopenfilenames(parent=root, title='Choose a file')
    print(root.filename)
    files = root.filename
    c = '/'
    index = [pos for pos, x in enumerate(files[0]) if x == c]
    path = files[0][:index[-1]+1]
    return files, path


def export_df_csv(df, subset_name='default',
                  path="C:\\Users\\krodgers\\OneDrive - Invensense, Inc\\Documents\\Sweep Testing Preliminary Data",
                  compress=False):
    df = df.convert_dtypes()

    if 'test_name' in df:
        title = df['test_name'].head(1).squeeze() + ', ' + df['tag'].head(1).squeeze()
    else:
        title = 'all tests'

    if compress:
        file_name = title + ', ' + subset_name + '-' + time_date + '.zip'
        df.to_csv(os.path.join(path, file_name), compression='zip')
    else:
        file_name = title + ', ' + subset_name + '-' + time_date + '.csv'
        df.to_csv(os.path.join(path, file_name))

    print(f'Exported {file_name}')


def filter_data(data, range_tol=20, linear_step=500, crop=True):

    # Sort the data by sample_number, which will also ensure angles and distances are in expected order
    df = data.sort_values(by=['sample_number'])

    # Data from sweepstage tends to have weird floats for angles, we just want ints so this does that
    if 'angle_round' in df:
        print('angle already rounded')
    else:
        df['angle_round'] = df['angle'].round()

    # Data from sweepstage doesn't know the intended target distance, so we have to determine it from the data
    if 'nom_dist' in df:
        print('nom_dist already determined')
    else:
        ### Nominal target distances determination

        # Looks at the on axis detected range across each linear step to create a list of measurement distances.
        dists = df.loc[(df['angle_round'] == 0), 'bin_range_mm']
        # Round distances for rare cases of messy on-axis data
        dists = dists.round()
        # Removes additional rows of on-axis data
        dists = dists.loc[~dists.duplicated()]
        # Rounds each distance according to the linear step of the test profile
        dists = dists.apply(lambda x: linear_step * round(x/linear_step))
        # Removes any distances that are 0
        dists = dists.loc[(dists != 0)]
        # Converts dists from pandas series to numpy array
        dists = dists.to_numpy()

        # Rounds "radius" column for comparison to range measurements
        df['radius'] = df['radius'].round(-2)
        # Creates a list of the step distances output by the sweepstage
        radii = df['radius'].unique()

        # Assigns nominal distance field to corresponding rows based on radius
        for i in range(len(dists)):
            df.loc[df['radius'] == radii[i], 'nom_dist'] = dists[i]

    # Define bounds for range measurement acceptance using specified range tolerance
    df['lower_bound'] = df['nom_dist'] - range_tol
    df['upper_bound'] = df['nom_dist'] + range_tol

    # Creates a series of bools that correspond to if the rows in df that are within the bounds
    filter_list = df['bin_range_mm'].between(df['lower_bound'], df['upper_bound'])
    # Set df "on_target" field to filter_list results
    df['on_target'] = filter_list

    # If enabled, remove the rows that measured off-target ranges
    if crop:
        df = df[filter_list]

    return df


def get_noise(measurement, noise_start=25, noise_end=35, export_csv=False, path=None):

    # Use 'frequency_lock' as it seems more reliably correct
    measurement['frequency'] = measurement['frequency_lock']

    # Give preview of how many traces are being decoded for wait time
    print('Number of measurements: ' + str(len(measurement)))

    # Getting IQ trace from datapoint
    print('Performing QI decode')
    measurement['qi_decode'] = measurement['encoded_qi_data'].apply(decode_qi_data)

    print('\rProcessing IQ data')
    measurement['iq_data'] = measurement['qi_decode'].apply(iqdata_unwrap)

    print('\rWriting magnitude traces')
    measurement['mag_trace'] = np.abs(measurement['iq_data'])

    print('\rWriting time traces')
    measurement['time_trace'] = measurement.apply(gen_time_trace, axis=1)

    # Initial list to store noise values
    noise_list = []

    # Iterate through datapoints and extract noise magnitude from iq trace
    for m in measurement.index:

        # Isolate trace data from measurement dataframe
        val = measurement.loc[m, ['time_trace', 'mag_trace']]

        mag_data = val['mag_trace']
        time_data = val['time_trace'] / 2. * 1000. * 343

        # Select data from specified sample range
        noise_mag = mag_data[noise_start:noise_end]

        # Add array of noise magnitude values to list
        noise_list.append(noise_mag)

    # Move noise data into dataframe for accessing by sample_number
    measurement['noise_floor'] = noise_list

    # Initialize list for mean sample noise
    mean_sample_noise = []

    # Iterate through samples to get mean of noise sample points
    for sample in measurement.loc[:, 'sample_number']:
        sample_noise = measurement.loc[(measurement['sample_number'] == sample), 'noise_floor']
        mean_sn = sample_noise.mean()
        mean_sample_noise.append(mean_sn)

    # Store noise mean in measurement dataframe
    measurement['noise_mean'] = mean_sample_noise

    # Optional export csv of noise values
    if export_csv:
        df_noise = measurement.loc[:,
                   ('test_name', 'tag', 'angle', 'nom_dist', 'sample_number', 'noise_floor', 'noise_mean')]
        export_df_csv(df_noise, 'noise_data')

    return measurement


def get_iq_traces(measurement, export_csv=False, path=None):

    # Crop dataframe to subset of columns since there will be a ton of rows after iq extraction
    df2 = measurement.loc[:,
          ('angle', 'nom_dist', 'test_name', 'tag', 'sample_number', 'mag_trace', 'time_trace', 'measurement_id')]

    # Get length of df2 before explode
    df2_length_0 = df2.shape[0]

    # Create placeholder copy of dataframe for explode operation
    df3 = df2

    # Restructure data to unpack trace data arrays into dataframe rows [# of rows after is (# datapoints x # iq samples)]
    df2 = df2.explode('mag_trace')
    df3 = df3.explode('time_trace')

    # Join time traces back into df2
    df2['time_trace'] = df3['time_trace']

    # Math for creating distance trace
    df2['distance_trace'] = df2['time_trace'] / 2. * 1000. * 343

    # Get length of df2 after explode, then divide original length to get trace length in samples
    df2_length_1 = df2.shape[0]
    trace_length = df2_length_1/df2_length_0

    samples_n = int(len(df2.sample_number) / trace_length)
    sample_points = list(np.arange(1, (trace_length+1))) * samples_n
    df2['sample_point'] = sample_points

    if export_csv:
        export_df_csv(df2, 'iq_trace_data', compress=False, path=path)

    return df2


def remove_gaps(measurement):
    distances = measurement['distance'].unique()
    distances.sort()

    neg_bound = None
    pos_bound = None
    df_mean_crop = pd.DataFrame()

    for D in distances:
        angles = measurement.loc[(measurement['distance'] == D), 'angle'].unique()
        angles.sort()

        angles_cont = pd.DataFrame()
        angles_cont['angles'] = np.arange(angles.min(), angles.max())

        angles_cont['is_gap'] = angles_cont['angles'].isin(angles)
        gaps = angles_cont.loc[~angles_cont['is_gap'], 'angles']
        pos_gaps = gaps.loc[gaps > 0]
        pos_gaps = pos_gaps.sort_values(ascending=False)
        pos_gaps = pos_gaps.tolist()
        neg_gaps = gaps.loc[gaps < 0]
        neg_gaps = neg_gaps.sort_values(ascending=True)
        neg_gaps = neg_gaps.tolist()

        if len(pos_gaps) == 0:
            pos_bound = angles.max()

        if len(neg_gaps) == 0:
            neg_bound = angles.min()

        gap_thresh = 2
        count_gap = 0
        for pg in pos_gaps:
            at_end = pg == 90 - gap_thresh
            has_neighbor = (pg + 1) in pos_gaps
            if not at_end and has_neighbor:
                count_gap = count_gap + 1
                if count_gap >= gap_thresh:
                    pos_bound = pg
            else:
                count_gap = 0

        count_gap = 0
        for ng in neg_gaps:
            at_end = ng == -90 + gap_thresh
            has_neighbor = (ng - 1) in neg_gaps
            if not at_end and has_neighbor:
                count_gap = count_gap + 1
                if count_gap >= gap_thresh:
                    neg_bound = ng
            else:
                count_gap = 0

        if pos_bound is not None and neg_bound is not None:
            angle_crop = measurement.loc[measurement['distance'] == D, 'angle'].between(neg_bound, pos_bound)
            cropped_data = measurement.loc[measurement['distance'] == D, :]
            cropped_data = cropped_data[angle_crop]
        else:
            angle_crop = measurement.loc[measurement['distance'] == D, 'angle'].between(-91, 91)
            cropped_data = measurement.loc[measurement['distance'] == D, :]
            cropped_data = cropped_data[angle_crop]

        df_mean_crop = df_mean_crop.append(cropped_data)

    return df_mean_crop


def get_mean(measurement):

    # Create a list of nominal distances from test data and make sure they're in order
    distances = measurement.nom_dist.unique()
    distances.sort()

    # Define column names for eventual dataframe creation and initialize list for storing data
    cols = ['distance', 'angle', 'mean_intensity', 'dist_mean_noise', 'dist_75p_noise', 'dist_90p_noise', 'num_range_id', 'mean_range_mm']
    data = []

    # Nested for loop that iterates through each test location and averages the intensity & range
    # Also produces mean and percentile noise values for each distance tested
    for d in distances:

        angles = measurement.loc[(measurement['nom_dist'] == d), 'angle_round'].unique()
        angles.sort()


        dist_noise = measurement.loc[(measurement['nom_dist'] == d), 'noise_floor']
        dist_noise = dist_noise.explode()
        dist_noise_mean = np.mean(dist_noise)
        dist_noise_75 = np.percentile(dist_noise, 75)
        dist_noise_90 = np.percentile(dist_noise, 90)

        for a in angles:
            ang_int = measurement.loc[((measurement['angle_round'] == a) & (measurement['nom_dist'] == d)), 'bin_intensity']
            ang_int = ang_int.loc[~ang_int.duplicated()]
            ang_id = len(ang_int)
            int_mean = ang_int.mean()

            ang_range = measurement.loc[((measurement['angle_round'] == a) & (measurement['nom_dist'] == d)), 'bin_range_mm']
            ang_range = ang_range.loc[~ang_range.duplicated()]
            range_mean = ang_range.mean()

            values = [d, a, int_mean, dist_noise_mean, dist_noise_75, dist_noise_90, ang_id, range_mean]
            zipped = zip(cols, values)
            data_dict = dict(zipped)
            data.append(data_dict)

    new_df = pd.DataFrame(data)
    new_df['mean_frequency'] = int(measurement['bin_frequency'].mean())

    return new_df


def plot_pre_pro(df_mean, smooth_on=False):

    ### Filter out datapoints that were collected from a partial amount of tested sensors
    '''
    Since the goal of the plots is to show average performance across a set of sensors with different frequencies,
    if only one sensor detects the target at a particular location it would be better to exclude that point
    as the majority of the sensors tested didn't detect at that location
    '''
    num_duts = df_mean['num_range_id'].unique()
    # num_duts = num_duts.loc[~num_duts.duplicated()]

    if len(num_duts) > 1:
        df_mean = df_mean.loc[(df_mean['num_range_id'] > 1), :]

    ### Intensity Cutoff
    '''
    Since the thresholds used in the sensor firmware are not a hard cutoff, this is where a hard cutoff can be applied.
    '''
    # Crop data for points with intensity less than desired threshold
    df_mean = df_mean.loc[(df_mean['mean_intensity'] >= 700), :]

    ### Function to remove gaps between angular datapoints
    df_mean = remove_gaps(df_mean)

    # In[9]:

    ### Rolling Average for Smoothing ###

    if smooth_on is True:
        df_mean = df_mean.reset_index()
        mean_intensity_roll = df_mean.groupby('distance')['mean_intensity'].rolling(7, center=True).mean()
        mean_intensity_roll = mean_intensity_roll.reset_index()
        mean_intensity_roll = mean_intensity_roll.groupby('distance')['mean_intensity'].apply(
            lambda group: group.interpolate(method='spline', order=1, limit_direction='both'))
        df_mean['mean_intensity_roll'] = mean_intensity_roll
        print('Smoothing applied to data')

    # sweep_postpro_functions.export_df_csv(df_mean, 'mean_data')


if __name__ =='__main__':
    pass

