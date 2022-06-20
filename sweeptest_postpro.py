from prod_test_functions import *
import sweep_postpro_functions

pd.options.mode.chained_assignment = None  # disables warning about copying dataframes

### Processing Parameters and Options ###
# Data Filter Parameters
crop_data = True    # defines whether or not to crop filtered data
range_tolerance = 20    # how much distance (in millimeters) from nominal target range should be considered "on-target"
linear_step = 500   # Distance (in millimeters) between linear steps of the sweep stage profile, used to find nom_dist

# Noise Parameters
'''
noise_samples selects where in the iq trace noise is pulled from, usually 5 samples somewhere before the target peak
default is 25 to 35 for most CH201 modules with good bandwidth,
for outdoor modules with worse bw, move it 10-20 samples further
'''
noise_samples_start = 25
noise_samples_end = 35

### Options ###
extract_IQ = False
export_noise_file = False

# Static file mode
# file_path = "C:\\Users\\krodgers\\OneDrive - Invensense, Inc\\Documents\\Sweep Testing Preliminary Data\\712 - 45deg Horn, Full Sweeps\\"
# filename = "Flat Target Sweep, 0.5-3m, 712 - 45, nf baf, post pro filtered.csv"

# File dialog
files, file_path = file_dialog()  # opens window to select sweeptest csv
files = list(files)  # converts files selected into a list

# initialize lists for storing dataframes from each file
list_dfs = []
list_dfs_iq = []

# Run post-processing functions on each file selected via for-loop
for f in files:

    ### CSV Intake as Dataframe ###
    # df = pd.read_csv(file_path + filename)  # uncomment for static file mode
    df = pd.read_csv(f) # creates dataframe from csv file
    df = df.loc[:, ~df.columns.duplicated()]  # removes duplicate columns

    ### Run Selected Functions ###

    # Main processing function, filters data with ranges that are not the correct distance
    df = sweep_postpro_functions.filter_data(df, linear_step=linear_step, crop=crop_data)

    # Get_noise function measures noise from iq trace for SNR computation
    df = sweep_postpro_functions.get_noise(df, noise_start=noise_samples_start, noise_end=noise_samples_end,
                                           export_csv=export_noise_file)

    # Adds processed files to outer list
    list_dfs.append(df)  # add current dataframe to list

    # Optionally produce separate file with IQ trace data for each point in sweep
    if extract_IQ:
        df_iq = sweep_postpro_functions.get_iq_traces(df, export_csv=True, path=file_path)
        list_dfs_iq.append(df_iq)  # add current dataframe to list

# Aggregates list of dataframes into single dataframe
df_all = pd.concat(list_dfs)

if extract_IQ:
    df_all_iq = pd.concat(list_dfs_iq)


# Run function to compute mean values at each datapoint of sweep
df_mean = sweep_postpro_functions.get_mean(df_all)

# Export mean data file (will prompt in console for file name)
export_file_name = input("Enter filename for processed file: ")
sweep_postpro_functions.export_df_csv(df_mean, export_file_name, path=file_path)  # export df mean for plotting
