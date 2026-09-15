import numpy as np

test_input = [[9, 39, 44, 46], [50, 48, 48, 41], [13, 27, 44, 30], [25, 42, 19, 40], [3, 3, 44, 39]]

# Filter rows with completely unique values
unique_rows = [row for row in test_input if len(row) == len(set(row))]

# Reflect horizontally (reverse each row)
result = [row[::-1] for row in unique_rows]

# Print in the required format
print(np.array(result).tolist())
