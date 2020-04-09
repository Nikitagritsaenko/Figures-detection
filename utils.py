from functools import reduce
import numpy as np


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx**2 + dy**2)


def convex_hull_graham(points):
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]), 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)
    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in range(1, len(u) - 1)) or l


def rotate_figure(figure, alpha):
    fig = []

    for i in range(len(figure)):
        point = figure[i]
        qx = np.cos(alpha) * point[0] - np.sin(alpha) * point[1]
        qy = np.sin(alpha) * point[0] + np.cos(alpha) * point[1]
        fig.append((qy, qx))
    return fig


def scale_figure(f, scale):
    res = []
    for i in range(len(f)):
        p = f[i]
        res.append((p[0] * scale, p[1] * scale))
    return res


def shift_figure(f, shift):
    res = []
    for i in range(len(f)):
        p = f[i]
        res.append((p[0] + shift[0], p[1] + shift[1]))
    return res


def compare_figures(fig1, fig2):
    f1 = fig1.copy()
    f2 = fig2.copy()

    s = 0
    for i in range(len(f1)):
        min_d = np.Inf
        idx = 0
        for j in range(len(f2)):
            d = dist(f1[i], f2[j])
            if min_d > d:
                min_d = d
                idx = j
        f2.pop(idx)
        s += min_d
    return s


def rotate_list(l, n):
    return l[n:] + l[:n]
