from subprocess import call
import numpy as np

total_time = 700
n_splits = 2
time = np.split(np.arange(0,total_time,0.01),n_splits)

# Append the last time point to the beginning of the next batch
for n,i in enumerate(time):
    if n>0:
        time[n] = np.append(i[0]-0.01,i)

np.save("time",time)

# call successive batches with a new python subprocess and pass the batch number
for i in range(n_splits):
    call(['python','run.py',str(i)])

print("Simulation Completed.")
