""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc

# Exc1
def approximate_pi(n):
    nc = 0
    inside_x = []
    inside_y = []
    outside_x = []
    outside_y = []
    for _ in range(n):
         x = random.uniform(-1, 1)
         y = random.uniform(1, -1)
         if (x**2 + y**2 <= 1):
              inside_x.append(x)
              inside_y.append(y)
              nc = nc + 1
              
         else:
              outside_x.append(x)
              outside_y.append(y)
    pi_approx = 4 * nc / n
    print(f"number of points n: {n}")
    print(f"approximation pi: {pi_approx}")

    plt.figure(figsize=(6,6))

    plt.scatter(inside_x, inside_y, c='red', s=5)
    plt.scatter(outside_x, outside_y, c = 'blue', s = 5)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.title(f"Monte Carlo pi approximation (n={n})")

    plt.savefig(f"pi_approx_{n}.png")
    plt.close()
    # Write your code here
    return pi_approx

# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points
    # d is the number of dimensions of the sphere 
    points = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    distances = list(map(lambda p:sum(x ** 2 for x in p), points))
    valid_distances = list(filter(lambda dist: dist <= 1, distances))

    nc = len(valid_distances)
    approx_vol = (2 ** d) * (nc / n)
    return approx_vol

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points
    # d is the number of dimensions of the sphere 
    numerator = m.pi ** (d/2)
    denominator = m.gamma(d/2 + 1 )
    return numerator / denominator


     


#Exc3: numba version
from numba import njit
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere
    #np is the number of processes
    valid_point = 0
    for _ in range(n):
         dist = 0.0
         for _ in range(d):
              x = random.uniform(-1, 1)
              dist = dist + (x ** 2)
         if dist <= 1:
              valid_point = valid_point + 1
    approx_vol = (2 ** d) * (valid_point / n)
    return approx_vol


#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes

    chunk = n // np
    n_list = [chunk] * np
    d_list = [d] * np
    with future.ProcessPoolExecutor(max_workers=np) as executor:
         results = list(executor.map(sphere_volume, n_list, d_list))
    average_value = sum(results) / np

    return average_value
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    print(f"Approx volume of {d} dimentional sphere = {sphere_volume(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    print(f"Approx volume of {d} dimentional sphere = {sphere_volume(n,d)}")

    # Exc3
    n = 1000000
    d = 11

    print("\n--- Exc3: Conventional Python (sphere_volume) ---")
    for i in range(1, 4):
         start = pc()
         sphere_volume(n, d)
         stop = pc()
         print(f"Call {i}: Time = {stop - start:.4f} seconds")


    print("\n--- Exc3: Numba JIT Acceleration (sphere_volume_numba) ---")
    for i in range(1, 4):
         start = pc()
         sphere_volume_numba(n, d)
         stop = pc()
         print(f"Call {i}: Time = {stop - start:.4f} seconds")


     
    # Exc4
    n = 1000000
    d = 11
    np = 10
    print("\n--- Exc4: Sequential vs Parallel ---")
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Sequential time: {stop - start:.4f} seconds")

    start = pc()
    sphere_volume_parallel(n, d, np)
    stop = pc()
    print(f"Parallel time ({np} cores): {stop - start:.4f} seconds")

    
    

if __name__ == '__main__':
	main()
