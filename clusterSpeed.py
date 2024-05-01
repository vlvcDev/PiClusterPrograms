import timeit
from mpi4py import MPI

def multiply(arr):
    k = 1
    for i in arr:
        k *= i
    return k

def divide(arr):
    k = 1
    for i in arr:
        k /= i
    return k

def add(arr):
    k = 1
    for i in arr:
        k += i
    return k

def subtract(arr):
    k = 1
    for i in arr:
        k -= i
    return k

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    iterations = 50000
    arrayLength = 960

    array = [i+1 for i in range(arrayLength)]
    
    start = timeit.default_timer()
    
    for _ in range(iterations):
        local_result = 0
        for i in range(rank, arrayLength, size):
            local_result = add(array)
            local_result = subtract(array)
            local_result = multiply(array)
            local_result = divide(array)
        global_result = comm.reduce(local_result, op=MPI.SUM, root=0)
    
    if rank == 0:
        stop = timeit.default_timer()
        print('Time: ', stop - start)

    MPI.Finalize()

if __name__ == "__main__":
    main()
