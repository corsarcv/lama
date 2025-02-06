numbers = [1010,1020.1, 1030.3, 1040.5, 1050.7]
from scipy.stats import linregress

x = list(range(len(numbers)))  # x-axis values
slope, intercept, r_value, p_value, std_err = linregress(x, numbers)  
print(slope, intercept, r_value, p_value, std_err)
import statistics
print(slope/statistics.mean(numbers))

