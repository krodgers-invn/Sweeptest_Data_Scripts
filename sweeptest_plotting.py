from prod_test_functions import *
import warnings
from math import floor, ceil
import sweep_postpro_functions
import matplotlib as mpl

'''
After this script was created, pandas announced the .append method would be deprecated in a future version.
To avoid making a lot of changes for now, the future warning is disabled. This should be fine as long as the new version
of pandas that removes .append is not used. So version 1.4.2 or earlier should be safe. 
'''
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # disables warning about copying dataframes

# Define pixel size so figure sizes can be specified in pixels
px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

# In[3]:


### File Dialog to open CSV file ###
# Script expects processed file output from the sweeptest_postpro.py script
file, file_path = sweep_postpro_functions.file_dialog_jnb()  # opens window to select sweeptest csv
file = file[0]

# In[4]:


# Read processed CSV file
df_mean = pd.read_csv(file)

# In[5]:


# Preview data
df_mean.head()

# In[6]:


## GENERAL PARAMS

# Define housing name and target type for plot and file labels
housing_name = 'PN840'
target_type = 'Pole'  # 'Flat' or 'Pole'
FOV = 45  # Nominal FOV in degrees for ploting FOV boundary lines

# Show basic plot of data (angle vs. intensity) to inspect data or compare with smoothed data.
basic_plot_display = True  # True or False to enable or disable this plot

# Enable or Disable Beam Pattern Curve Smoothing
smooth_on = False
# Set smoothing window, larger window smoothes more but will alter shape more
smooth_window = 7

# Select which beam pattern distance to use for all other distances.
noise_dist = 1000

## POLAR PARAMS
# Select which distance of beam pattern curve to generate polar plot from.
'''
Most of the time the closest distance is the best to use, but if there is any sensor saturation 
the next distance further may be better to get the complete beam pattern.
'''
polar_plot_dist = 500

## SNR PARAMS##
'''
See top SNR Contour Plot cell (should be last cell in script) for these parameters.
'''

# In[7]:


### Filter out datapoints that were collected from a partial amount of tested sensors
'''
Since the goal of the plots is to show average preformance across a set of sensors with different frequencies,
if only one sensor detects the target at a particular location it would be better to exclude that point
as the majority of the sensors tested didn't detect at that location
'''
num_duts = df_mean['num_range_id'].unique()
# num_duts = num_duts.loc[~num_duts.duplicated()]

if len(num_duts) > 1:
    df_mean = df_mean.loc[(df_mean['num_range_id'] > 1), :]

### Various coarse filters to isolate data
'''
Since the thresholds used in the sensor firmware are not a hard cutoff, this is where a hard cutoff can be applied.
'''
# Crop data for points with intensity less than desired threshold
df_mean = df_mean.loc[(df_mean['mean_intensity'] >= 700), :]

# If for some reason you want to exclude certain angles you can do it like this:
# df_mean = df_mean.loc[(df_mean['angle'] <= 30), :]

### Function to remove gaps between angular datapoints
df_mean = sweep_postpro_functions.remove_gaps(df_mean)

# In[8]:


### Rolling Average for Smoothing ###
'''
So far the smoothing has only been used for beam patterns with oscillations that usually occur with wider FOV housings.
The smooth_window set in the parameters section defines the size (in # of rows) of the rolling average window used to smooth the data.
A larger smoothing window will smooth more, but with more change to the shape of the beam pattern, especially at the two ends of the curve. 
Ideally the smallest window that produces a smoothed curve should be used, usually between 3-7 degrees.
Keep in mind the smooth window is a # of rows, so it may need to be changed for different angular steps. 
'''

if smooth_on is True:
    # Reset index to ensure consistent order of rows
    df_mean = df_mean.reset_index()
    # Apply rolling average to the beam pattern curve at each distance
    mean_intensity_roll = df_mean.groupby('distance')['mean_intensity'].rolling(smooth_window, center=True).mean()
    # Reset index again to ensure rows are in order
    mean_intensity_roll = mean_intensity_roll.reset_index()
    # Linear interpolation at the end of each beam pattern curve since the rolling average misses smooth_window/2 rows at the ends of the beam pattern.
    mean_intensity_roll = mean_intensity_roll.groupby('distance')['mean_intensity'].apply(
        lambda group: group.interpolate(method='spline', order=1, limit_direction='both'))
    # Add rolling averaged mean intensity to dataframe
    df_mean['mean_intensity_roll'] = mean_intensity_roll
    print('Smoothing applied to data')

# In[9]:


### Noise Usage for SNR Plot ###
'''
Though noise is calculated for each distance tested, the noise determined from certain distances is probably cleaner.
This is because the samples used for the noise calculation are in a small iq trace window between ringdown and the first target.
With more space between ringdown and the target (e.g. 1000mm for CH201), it is more likely the window captured just noise. 
This is all to say that it may be good to just use one noise value from one distance and apply it to the rest of the beam patterns.  
'''

# Assign new column for selected statistical method of noise calculation
# dist_90p_noise has always been used but it could also be set to dist_mean_noise or dist_75p_noise
df_mean['dist_noise'] = df_mean['dist_90p_noise']

# Narrow noise to single value to be applied to rest of data.
noise_single = df_mean.loc[df_mean['distance'] == noise_dist, 'dist_noise'].unique()
print(noise_single)
# Reformat data type from series to float
noise_single = noise_single[0]  # if script throws an error here, check that the noise_dist used is actually in data.

# Apply single noise value to all other distances, this could also be used to manually specify a noise value
df_mean.loc[:, 'dist_noise'] = noise_single

# Print noise value to give visibility and sanity check that it's reasonable (for CH201 it's usually 70-100 LSB)
print(df_mean['dist_noise'].unique())

# In[10]:


# df_mean['dist_range_mm'] = df_mean.groupby('distance')['mean_range_mm'].transform(np.mean)

# Remove any rows with NaN values as they will prevent the contour plot from working
df_mean = df_mean.loc[~df_mean.mean_intensity.isna()]

# Create column for mean intensity in dBs rather than LSB
df_mean['mean_intensity_db'] = df_mean['mean_intensity'].apply(lambda x: 20 * np.log10(x))

# If smoothing is enabled, use rolling averaged intensity in dB instead
if smooth_on is True:
    df_mean['mean_intensity_db'] = df_mean['mean_intensity_roll'].apply(lambda x: 20 * np.log10(x))

# Create new column for noise data in dBs
df_mean['dist_noise_db'] = df_mean['dist_noise'].apply(lambda x: 20 * np.log10(x))

# Create column for calculated SNR values
df_mean['SNR'] = df_mean['mean_intensity_db'] - df_mean['dist_noise_db']

# Calucate max and min SNR values and convert to ints for use in setting plot limits
max_snr = df_mean['SNR'].max()
max_snr = np.floor(max_snr)
min_snr = df_mean['SNR'].min()
min_snr = np.ceil(min_snr)

# Convert from degrees to radians for polar to cartesian calculations
df_mean['angle_rad'] = df_mean['angle'].apply(np.deg2rad)
# Create columns for coverting original polar coordinates to cartesian, which are then used in the contour plot.
df_mean['x_vals'] = df_mean.apply(lambda x: x['mean_range_mm'] * np.sin(x['angle_rad']), axis=1)
df_mean['y_vals'] = df_mean.apply(lambda x: x['mean_range_mm'] * np.cos(x['angle_rad']), axis=1)

# In[11]:


### Comparison Plot of Original vs. Smoothed Data ###
# This can be used to evaluated the effect of the smoothing settings chosen before moving on to other plots.
# If smoothing is not enabled, it will just plot the original data by itself.

# Create figure
fig2 = plt.figure(1)

# Plot Original Data
plt.scatter(df_mean['angle'], df_mean['mean_intensity'], c=df_mean['distance'], cmap='winter',
            norm=mpl.colors.LogNorm())

if smooth_on is True:
    # Plot Smoothed Data on same axes
    plt.scatter(df_mean['angle'], df_mean['mean_intensity_roll'], c=df_mean['distance'], cmap='autumn',
                norm=mpl.colors.LogNorm())

# Enable Plot Grid
plt.grid(True)

if basic_plot_display is True:
    plt.show()

# In[14]:


### Polar Plot ###

# Crop dataframe to distance specified for use in polar plot
df_mean_pp = df_mean.loc[(df_mean['distance'] == polar_plot_dist), :]

# Calculate normalized amplitude in dB for use in polar plot
polar_max = df_mean_pp['mean_intensity_db'].max()
df_mean_pp['amp_norm'] = df_mean_pp['mean_intensity_db'] - polar_max

# Get normalized amplitude min as int for polar plot label
polar_min_norm = df_mean_pp['amp_norm'].min()
polar_min_norm = np.ceil(polar_min_norm)

### Polar Plot Settings ###

# Create figure and define size
fig_p = plt.figure(figsize=(600 * px, 600 * px))

# Create axis object and set to polar
ax_p = fig_p.add_subplot(projection='polar')

# Create plot of angle (takes angle in radians but will display as degrees) vs. normalized db amplitude
plt.plot(df_mean_pp['angle_rad'], df_mean_pp['amp_norm'])

# Some settings to get polar plot to display correctly as hemisphere
ax_p.set_theta_zero_location('N')
ax_p.set_theta_direction(-1)
ax_p.set_thetamin(90)
ax_p.set_thetamax(-90)

# Axis labels and ticks
ax_p.set_xlabel('Amplitude (dB)', labelpad=-60)
ax_p.set_xticks(np.arange(np.deg2rad(-90), np.deg2rad(91), np.deg2rad(10)))

# Amplitude axis settings
'''
*** POLAR PLOT AXIS SETTING ***
This is where the amplitude axis settings for the polar plot are defined.
The lower bound and step may need to be tuned for each housing so that the beam pattern is properly framed in the plot. 
'''
ax_p.set_yticks(np.arange(polar_min_norm, 1, 6))

'''
This is where the title of the polar plot is set
'''
ax_p.set_title(f'Polar Beam Pattern Plot, {target_type} Target', pad=-40)

# Save Plot
'''
After plot setting have been tuned to fit the housing's data, enable this line to save the plot.
Plot can also be saved manually from the figure window if "mpl.use('Qt5Agg')" is enabled.  
File will be saved in the same folder as the input CSV read by file dialog.
'''
# plt.savefig((file_path + f'{housing_name}_{target_type}_polar_plot.jpg'))

plt.show()

# In[13]:


### SNR CONTOUR PLOT ###

'''
The levels used in the contour plot will need to be tuned for each housing, depending on the spread between min and max SNR.
The reason why the manual setting is required, is because the min SNR contours may not be closed, but ideally the levels are set 
so that the lowest level displayed is the minimum closed contour. 
Usually the min_snr and max_snr are a good place to start to get an idea of where the levels should be moved.
One approach to this is to start with the min and max levels, with automatic labels on, and the matplotlib backend set to inline.
'''
contour_levels = np.arange((min_snr), (max_snr), step=3)
# contour_levels = np.arange(35, 42, step=3)

min_radius = 510

# Define x and y-axis limits
xlim = 1500
ylim = 1750
# Define Grid Step size
grid_step = 250

manual_contour_label = True

# Create triangulation to map contour plot onto.
'''
Creating a triangulation basically creates a mesh from the x and y coordinates of the test sweep.
This then allows for removing certain undesired parts of plot from displaying.
The main use for this is to removed the semicircular region between the sensor position (0,0) and the 
first arc of test points (r=500 for example).
Because the first distance arc isn't always the same, it needs to be adjusted for each housing. 
'''

# Define triangluation using rectangular coordinates.
triang = mpl.tri.Triangulation(df_mean['x_vals'], df_mean['y_vals'])

# Create lists of x and y triangles for use in masking calculation
x = df_mean['x_vals']
x = x.to_numpy()
y = df_mean['y_vals']
y = y.to_numpy()
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)

# Mask off unwanted triangles in minimum radius semicircular zone
mask = np.where(xmid * xmid + ymid * ymid < min_radius * min_radius, 1, 0)
triang.set_mask(mask)  # Disable this line to disable masking, a good way to see what it is that's being removed.

# Create figure and define size
fig, ax = plt.subplots(figsize=(750 * px, 750 * px))

# Uncomment the following line to show the test point in x & y coordinates. (Useful for visualize limits of contours)
# plt.scatter(df_mean['x_vals'], df_mean['y_vals'], marker='.', c='k', alpha=0.5)

# Add red triangle to symbolize sensor location on plot
plt.scatter(0, 0, marker='^', c='r', s=1000, label="Sensor")

# Generate contour plot using triangulation and SNR data. cmap can be changed to change color scheme of plot.
contour = plt.tricontour(triang, df_mean['SNR'], levels=contour_levels, cmap='plasma', extend='min')

# Plot Title and Labels
ax.set_title(f'SNR Contour Plot, {target_type} Target')
ax.set_ylabel('y Distance (mm)')
ax.set_xlabel('x Distance (mm)')

# Set x and y axis limits
ax.set_xlim(-xlim, xlim)
ax.set_ylim(0, ylim)

# Set axis tick labels
xticks = np.arange(-xlim, xlim + 1, grid_step)
ax.set_xticks(xticks)
yticks = np.arange(0, ylim + 1, grid_step)
ax.set_yticks(yticks)

# Enable grid
ax.grid(True)

# Reformat FOV angle to fit line plotting function input.
fov_ang_plot = 90 - (FOV / 2)
line_slope = np.tan(np.deg2rad(fov_ang_plot))
# Add lines representing FOV of sensor to contour plot
plt.axline([0, 0], slope=line_slope, linestyle=':', c='k', label='FoV')
plt.axline([0, 0], slope=-line_slope, linestyle=':', c='k')

# Add legend to plot, showing sensor icon and FOV lines
plt.legend(*(
    [x[i] for i in [1, 0]]
    for x in plt.gca().get_legend_handles_labels()), labelspacing=2, loc='upper right')


# Formating function used in SNR Contour Plot for contour labels
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} dB"


# Contour curve labeling, either automatic or manually (must enable "mpl.use('Qt5Agg')" at top of script for manual to work)
if manual_contour_label is True:
    ax.clabel(contour, inline_spacing=3, colors='k', manual=True, fmt=fmt, fontsize=10)
else:
    ax.clabel(contour, inline_spacing=3, colors='k', fmt=fmt, fontsize=10)

plt.show()

'''
Since contour labels will probably need to be done manually, the contour plot should be saved from the plot window after labels are added
*** In order to be able to rerun this cell without restarting kernal and rerunning whole script, 
it is necessary to cancel the manual labeling after labels are placed by clicking the scroll wheel anywhere on the plot before
closing the window. This allows the cell to finish running correctly, and it should print "done" if it ends correctly. *** 
'''
print('done')

# In[ ]:




