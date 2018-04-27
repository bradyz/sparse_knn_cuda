import numpy as np
import matplotlib.pyplot as plt


def get_file(filename):
    result = None

    with open(filename, 'r') as f:
        result = [line.strip().split() for line in f]

    return result


def find_relevant_mine(query, n, d, k, s, points):
    result = list()

    for (n_i, d_i, k_i, s_i), val in points.items():
        if n is not None and n != n_i:
            continue
        if d is not None and d != d_i:
            continue
        if k is not None and k != k_i:
            continue
        if s is not None and s != s_i:
            continue

        item = None
        item = item or (n_i if n is None else None)
        item = item or (d_i if d is None else None)
        item = item or (k_i if k is None else None)
        item = item or (s_i if s is None else None)

        result.append((item, val[query]))

    return result


def find_relevant_theirs(n, d, k, points):
    result = list()

    for (n_i, d_i, k_i), val in points.items():
        if n is not None and n != n_i:
            continue
        if d is not None and d != d_i:
            continue
        if k is not None and k != k_i:
            continue

        item = None
        item = item or (n_i if n is None else None)
        item = item or (d_i if d is None else None)
        item = item or (k_i if k is None else None)

        result.append((item, val))

    return result


def parse_mine(lines):
    points = dict()

    i = 0

    while i < len(lines):
        n, d, k, s = lines[i]
        n = int(n)
        d = int(d)
        k = int(k)
        s = float(s)

        total = float(lines[i+1][1])
        conv = float(lines[i+2][1])
        mult = float(lines[i+3][1])
        norm = float(lines[i+4][1])
        sort = float(lines[i+5][1])
        select = float(lines[i+6][1])

        points[(n, d, k, s)] = {
                'total': total,
                'conv': conv,
                'mult': mult,
                'norm': norm,
                'sort': sort,
                'select': select
                }

        i += 7

    return points


def parse_theirs(lines):
    points = dict()

    i = 0

    while i < len(lines):
        n, d, k = lines[i]
        n = int(n)
        d = int(d)
        k = int(k)

        points[(n, d, k)] = float(lines[i+1][-1])

        i += 2

    return points


def main():
    my_times = get_file('times.txt')
    their_times = get_file('knn_cuda.txt')

    query = 'total'
    n = 4096
    d = 1024
    s = 0.5
    k = None

    missing = None
    missing = missing or ('number ref/query' if n is None else None)
    missing = missing or ('dimensions' if d is None else None)
    missing = missing or ('neighbors' if k is None else None)
    missing = missing or ('sparsity' if s is None else None)

    relevant_mine = find_relevant_mine(query, n, d, k, s, parse_mine(my_times))
    relevant_mine.sort()

    relevant_theirs = find_relevant_theirs(n, d, k, parse_theirs(their_times))
    relevant_theirs.sort()

    relevant_mine = np.float32(relevant_mine)
    relevant_theirs = np.float32(relevant_theirs)

    n = n or '-'
    d = d or '-'
    s = s or '-'
    k = k or '-'

    plt.title('n: %s, k: %s, d: %s, sparsity: %s' % (n, k, d, s))

    plt.plot(relevant_mine[:,0], relevant_mine[:,1], '--bo', label='ours')

    if len(relevant_theirs) == 1:
        plt.axhline(
                y=relevant_theirs[0][1],
                color='g', linestyle='--', marker='o',
                label='knn cuda')
    else:
        plt.plot(relevant_theirs[:,0], relevant_theirs[:,1], '--go', label='knn cuda')

    plt.xlabel(missing)
    plt.ylabel('time (seconds)')
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
