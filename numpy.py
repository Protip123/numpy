import numpy as np
from time import process_time
python_list = [i for i in range(10000)]
start_time = process_time()
python_list = [i+5 for i in python_list]
end_time = process_time()
print(end_time-start_time)

np_array = np.array([i for i in range(10000)])
start_time = process_time()
np_array += 5
end_time = process_time()
print(end_time-start_time)

list1