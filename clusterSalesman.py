from mpi4py import MPI
import numpy as np
import itertools

def calculate_distance(route, distance_matrix):
    total = 0
    num_cities = len(route)
    for i in range(num_cities):
        total += distance_matrix[route[i]][route[(i+1)%num_cities]]
    return total

def tsp_mpi(distance_matrix):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_cities = len(distance_matrix)
    cities = list(range(num_cities))

    shortest_route = None
    shortest_distance = None

    for route in itertools.permutations(cities):
        route = list(route)
        if route[0] != 0:  # skip permutations with the same route
            break
        if rank == route[1] % size:  # distribute tasks among processes
            distance = calculate_distance(route, distance_matrix)
            if shortest_distance is None or distance < shortest_distance:
                shortest_route = route
                shortest_distance = distance

    shortest_route = comm.gather(shortest_route, root=0)
    shortest_distance = comm.gather(shortest_distance, root=0)

    if rank == 0:
        shortest = min(shortest_distance)
        index = shortest_distance.index(shortest)
        return shortest_route[index], shortest

    return None

def main():
    distance_matrix = np.array([
        [0, 29, 20, 21, 16, 31, 100, 12, 4, 31, 18, 19],
        [29, 0, 15, 29, 28, 40, 72, 21, 29, 41, 12, 27],
        [20, 15, 0, 15, 14, 25, 81, 9, 23, 27, 13, 14],
        [21, 29, 15, 0, 4, 12, 92, 12, 25, 13, 25, 13],
        [16, 28, 14, 4, 0, 16, 94, 9, 20, 16, 22, 22],
        [31, 40, 25, 12, 16, 0, 95, 24, 36, 3, 37, 12],
        [100, 72, 81, 92, 94, 95, 0, 90, 101, 99, 84, 82],
        [12, 21, 9, 12, 9, 24, 90, 0, 15, 25, 13, 14],
        [4, 29, 23, 25, 20, 36, 101, 15, 0, 35, 18, 23],
        [31, 41, 27, 13, 16, 3, 99, 25, 35, 0, 38, 12],
        [18, 12, 13, 25, 22, 37, 84, 13, 18, 38, 0, 25],
        [19, 27, 14, 13, 22, 12, 82, 14, 23, 12, 25, 0]
    ])
    print(tsp_mpi(distance_matrix))

    MPI.Finalize()

if __name__ == "__main__":
    main()