import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Load the sample financial dataset from your specified file path
data = pd.read_csv(r"C:\Users\Navid\Desktop\441\apple_stock_ytd_close.csv", parse_dates=['Date'])

# Sort the dataset based on Date
data.sort_values('Date', inplace=True)

# Merge Sort Implementation (for educational purposes, not necessary to use Python's built-in sort)
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

# Example usage of merge sort:
prices = list(data['Price'])
merge_sort(prices)  # Now the data is sorted

# Kadane’s Algorithm to find maximum gain or loss period
def kadane(arr):
    max_current = max_global = arr[0]
    start = end = s = 0
    
    for i in range(1, len(arr)):
        if arr[i] > max_current + arr[i]:
            max_current = arr[i]
            s = i
        else:
            max_current += arr[i]

        if max_current > max_global:
            max_global = max_current
            start = s
            end = i

    return max_global, start, end

# Calculate daily changes (gain/loss)
data['Daily_Change'] = data['Price'].diff().fillna(0)

# Apply Kadane's algorithm
max_gain, start_idx, end_idx = kadane(data['Daily_Change'].values)
print(f"Maximum gain: {max_gain} from {data.iloc[start_idx]['Date']} to {data.iloc[end_idx]['Date']}")

# Closest Pair of Points Algorithm for Anomaly Detection
def closest_pair(points):
    def dist(p1, p2):
        return distance.euclidean(p1, p2)
    
    def closest_pair_recursive(pts):
        if len(pts) <= 3:
            return min([(dist(pts[i], pts[j]), pts[i], pts[j]) 
                        for i in range(len(pts)) for j in range(i+1, len(pts))], default=(float('inf'), None, None))

        mid = len(pts) // 2
        left_closest = closest_pair_recursive(pts[:mid])
        right_closest = closest_pair_recursive(pts[mid:])
        min_dist = min(left_closest, right_closest, key=lambda x: x[0])

        return min_dist
    
    points.sort(key=lambda x: x[0])
    return closest_pair_recursive(points)

# Convert data points into (Date, Price) pairs
points = list(zip(data['Date'].astype(np.int64), data['Price']))
min_dist, point1, point2 = closest_pair(points)
print(f"Closest pair distance: {min_dist} between points {point1} and {point2}")

# Visualization of the Maximum Gain/Loss Period
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Price'], label='Stock Price')
plt.axvspan(data.iloc[start_idx]['Date'], data.iloc[end_idx]['Date'], color='yellow', alpha=0.3, label='Max Gain Period')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices with Maximum Gain Period Highlighted')
plt.legend()
plt.show()

# Visualization of Anomalies Detected Using Closest Pair of Points
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Price'], label='Stock Price')
plt.scatter([pd.to_datetime(point1[0]), pd.to_datetime(point2[0])], [point1[1], point2[1]], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices with Detected Anomalies')
plt.legend()
plt.show()
