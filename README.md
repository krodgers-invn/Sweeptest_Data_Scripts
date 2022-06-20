# Sweeptest_Data_Scripts
## Uses:
•	Trim data to remove datapoints that measured distance incorrectly.

•	Aggregate data from multiple sensors of the same sweep profile and average them. 

•	Extract noise floor data from the iq traces of sweep test data. 

•	Decode iq traces embedded in data and output them in a JMP friendly CSV file. 

•	Prepare data for plotting and create polar beam pattern and SNR contour plots. 

## Getting Data:
### 0. csv_from_test_v2.exe

This is a tool to retrieve the sweeptest data from the database. It is run through the terminal and takes the test number as input, then creates a labeled CSV file with the test data. The data can also be pulled from JMP via SQL query, but the tables will need to be joined in the same way as csv_from_test_v2 to ensure the post-processing scripts have the right columns. 
The CSV file will be saved to the same folder that the .exe is in, which can be moved to wherever is convenient. 

## Post-process Data and Plot:
The post-processing scripts are broken up into 3 separate scripts for ease of use and modification. 

### 1. sweeptest_postpro.py 
Top-level control script, which handles input of CSV files, options for processing, and calls functions from other scripts. It is where the user will input CSV’s from the sweep test and get a processed file ready for plotting as output. 

Input : One or more CSV files from the csv_from_test_v2.exe tool, must all be same housing and target type. 

Output : A single processed CSV file that combines all original CSVs for use in sweeptest_plotting.py. Can also be used to output CSV files with IQ trace data or noise data. 

### 2. sweep_postpro_functions.py
This script is what does most of the data processing, with several functions to break-up the tasks into smaller pieces. The functions in this script will be called by the other two scripts, so it’s important that it is in the same folder.   

### 3. sweeptest_plotting.py or sweeptest_plotting.ipynb (Jupyter Notebook version)
This script has some functionality related to preparing data for plotting and generates 3 different types of plots from a processed file. There are two versions of the script, one for use in Pycharm (or any other IDE) and one for use in Jupyter Notebooks. The Jupyter Notebooks version is much easier to use, especially when multiple files are being plotted, but the Pycharm version is also available so that the script can be used without needing to set up Jupyter. 

Input : Single processed CSV file from sweeptest_postpro.py

Output: Polar Beam Pattern Plot and SNR Contour Plot. 

## Note on prod_functions.py:

This script was pulled from sweng.chirpmicro-analysis_python/Stefon/prod_test on github, and is mainly used for the file dialog function and IQ trace decoding. It is also where most of the required packages are imported, so that the other scripts import everything from prod_test_functions.py. 

