import numpy as np
import matplotlib.pyplot as plt


def get_file(filename):
    result = None

    with open(filename, 'r') as f:
        result = [line.strip().split() for line in f]

    return result


def find_relevant_mine(query, n, d, s, points):
    result = list()

    for (n_i, d_i, s_i), val in points.items():
        if n is not None and n != n_i:
            continue
        if d is not None and d != d_i:
            continue
        if s is not None and s != s_i:
            continue

        item = None
        item = item or (n_i if n is None else None)
        item = item or (d_i if d is None else None)
        item = item or (s_i if s is None else None)

        result.append((item, val[query]))

    return result


def find_relevant_theirs(n, d, points):
    result = list()

    for (n_i, d_i), val in points.items():
        if n is not None and n != n_i:
            continue
        if d is not None and d != d_i:
            continue

        item = None
        item = item or (n_i if n is None else None)
        item = item or (d_i if d is None else None)

        result.append((item, val))

    return result


def parse_mine(lines):
    points = dict()

    i = 0

    while i < len(lines):
        n, d, s = lines[i]
        n = int(n)
        d = int(d)
        s = float(s)

        total = float(lines[i+1][1])
        mult = float(lines[i+2][1])
        norm = float(lines[i+3][1])
        sort = float(lines[i+4][1])

        points[(n, d, s)] = {'total': total, 'mult': mult, 'norm': norm, 'sort': sort}

        i += 5

    return points


def parse_theirs(lines):
    points = dict()

    i = 0

    while i < len(lines):
        n, d = lines[i]
        n = int(n)
        d = int(d)

        points[(n, d)] = float(lines[i+1][0])

        i += 2

    return points


def main():
    my_times = get_file('times.txt')
    their_times = get_file('knn_toolkit_times.txt')

    query = 'total'
    n = 4096
    d = None
    s = 0.001

    missing = None
    missing = missing or ('number points' if n is None else None)
    missing = missing or ('dimensions' if d is None else None)
    missing = missing or ('sparsity' if s is None else None)

    relevant_mine = find_relevant_mine(query, n, d, s, parse_mine(my_times))
    relevant_mine.sort()

    relevant_theirs = find_relevant_theirs(n, d, parse_theirs(their_times))
    relevant_theirs.sort()

    relevant_mine = np.float32(relevant_mine)
    relevant_theirs = np.float32(relevant_theirs)

    plt.title('n: %s, d: %s, sparsity: %s' % (n, d, s))

    plt.xlabel(missing)
    plt.ylabel('time (seconds)')

    plt.plot(relevant_mine[:,0], relevant_mine[:, 1], '--o', label='ours')
    plt.plot(relevant_theirs[:,0], relevant_theirs[:, 1], '--o', label='knn toolkit')

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
