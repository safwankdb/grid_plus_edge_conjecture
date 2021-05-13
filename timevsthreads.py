import subprocess
from time import time
import numpy as np
from matplotlib import pylab as plt
import seaborn as sns

m = 1
nt = "10"
n_list = list(range(1, 9))


omp = []
subprocess.run(["g++", "-O3", "-fopenmp",
                "four_checker_omp.cpp", "-o", "omp_out"], stdout=subprocess.DEVNULL)
for n in n_list:
    for i in range(m):
        start = time()
        subprocess.run(['env', f'OMP_NUM_THREADS={n}', "./omp_out", nt, nt],
                       stdout=subprocess.DEVNULL)
        time_taken = time()-start
        print(
            f"OpenMP for {nt}x{nt} on {n} threads, time taken = {time_taken:.02f} sec")
        omp.append(time_taken)

mpi = []
subprocess.run(["mpic++", "-O3", "four_checker_mpi.cpp",
                "-o", "mpi_out"], stdout=subprocess.DEVNULL)
for n in n_list:
    for i in range(m):
        start = time()
        subprocess.run(["mpirun", "-np", str(n), "--use-hwthread-cpus", "--allow-run-as-root",
                        "./mpi_out", nt, nt], stdout=subprocess.DEVNULL)
        time_taken = time()-start
        print(
            f"MPI for {nt}x{nt} on {n} threads, time taken = {time_taken:.02f} sec")
        mpi.append(time_taken)


omp = np.array(omp).reshape(-1, m).mean(1)
mpi = np.array(mpi).reshape(-1, m).mean(1)

sns.set()
plt.figure(figsize=(12, 8))
plt.plot(n_list, omp, color='r', label="OpenMP")
plt.plot(n_list, mpi, color='b', label="MPI")
plt.legend()
plt.xlabel("Number of threads")
plt.ylabel("Time take in sec")
plt.title("Timing Study")
plt.savefig("imgs/2.jpg")
