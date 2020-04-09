import cv2 as cv
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_opening
from skimage.filters import threshold_otsu
from skimage import color
from utils import *


def find_shapes_point_list(src):
    im = src.copy()
    shape_list = []

    q = []
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            if im[x, y] != 0:
                shape = [(x, y)]
                q.append((x, y))
                while len(q) != 0:
                    elem = q.pop()
                    for u in range(elem[0] - 1, elem[0] + 2):
                        for v in range(elem[1] - 1, elem[1] + 2):
                            if im[u, v] != 0:
                                im[u, v] = 0
                                if not ((u, v) in q):
                                    shape.append((u, v))
                                    q.append((u, v))
                shape_list.append(shape)

    return shape_list


def get_rid_of_noise(img):
    dst = img.copy()
    threshold = threshold_otsu(dst)
    dst = dst > threshold

    dst = binary_fill_holes(dst)
    dst = binary_opening(dst)

    return dst


def find_figure_corners(img):
    figure_corners_list = []
    dst = img.copy()
    dst = np.float32(dst)
    dst = cv.cornerHarris(dst, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    for x in range(dst.shape[0]):
        for y in range(dst.shape[1]):
            threshold = 0.1 * dst.max()
            if dst[x, y] > threshold:
                local_max = 0
                local_max_ind = (0, 0)
                r = 5
                u_left = np.max([x - r, 0])
                u_right = np.min([x + r, dst.shape[0] - 1])
                v_left = np.max([y - r, 0])
                v_right = np.min([y + r, dst.shape[1] - 1])
                for u in range(u_left, u_right):
                    for v in range(v_left, v_right):
                        if local_max < dst[u, v]:
                            local_max = dst[u, v]
                            local_max_ind = (u, v)
                        dst[u, v] = 0
                if local_max > threshold:
                    figure_corners_list.append(local_max_ind)

    return figure_corners_list


def find_angle_and_shift(basis, fig, scale):
    best_alpha = 0
    best_shift = (0, 0)
    min_err = np.Inf

    for alpha in range(-180, 180, 1):
        for i in range(len(fig)):
            figure = fig.copy()

            transformed = scale_figure(basis, scale)
            transformed = rotate_figure(transformed, np.deg2rad(alpha))

            shift = (figure[i][0] - transformed[0][0],
                     figure[i][1] - transformed[0][1])

            transformed = shift_figure(transformed, shift)
            err = compare_figures(figure, transformed)

            if err < min_err:
                min_err = err
                best_alpha = alpha
                best_shift = shift

    return best_alpha, best_shift


def solve_problem(N, basis_figures, figure_corners):
    template_id = -1
    min_var = np.Inf
    best_orientation = []
    best_template = []
    mean = 0

    for i in range(N):
        basis_figure_points = basis_figures[i]
        template = []
        for j in range(0, len(basis_figure_points) - 1, 2):
            template.append((basis_figure_points[j], basis_figure_points[j+1]))

        if len(template) != len(figure_corners):
            continue

        m = len(template)

        for k in range(m):
            figure_sides = []
            template_sides = []
            for j in range(m - 1):
                figure_sides.append(dist(figure_corners[j], figure_corners[j + 1]))
                template_sides.append(dist(template[j], template[j + 1]))
            figure_sides.append(dist(figure_corners[0], figure_corners[m - 1]))
            template_sides.append(dist(template[0], template[m - 1]))

            scale_ratio = np.array(figure_sides) / np.array(template_sides)
            if min_var > np.var(scale_ratio):
                template_id = i
                best_template = template
                min_var = np.var(scale_ratio)
                mean = np.mean(scale_ratio)
                best_orientation = figure_corners.copy()

            figure_corners = rotate_list(figure_corners, 1)

    scale = int(mean)

    rotation, shift = find_angle_and_shift(best_template, best_orientation, scale)

    return template_id, scale, rotation, int(shift[1]), int(shift[0])


def find_figures(N, basis_figures, src_image):
    answer_list = []

    img = src_image.copy()
    img = get_rid_of_noise(img)

    shapes_points = find_shapes_point_list(img)
    for i in range(len(shapes_points)):
        img_with_one_figure = img.copy()

        for j in range(len(shapes_points)):
            if i == j:
                continue
            shape = shapes_points[j]
            for p in shape:
                img_with_one_figure[p[0], p[1]] = 0

        figure_corners = find_figure_corners(img_with_one_figure)
        figure_corners = convex_hull_graham(figure_corners)
        answer_list.append((solve_problem(N, basis_figures, figure_corners)))

    return answer_list


def main(args):
    src_file = open(args.structure, "r")

    N = int(src_file.readline())
    basis_figures = []
    for i in range(N):
        new_figure = [int(number) for number in src_file.readline().split(',')]
        basis_figures.append(new_figure)

    src_image = plt.imread(args.image)
    src_image_gray = color.rgb2gray(src_image)
    result = find_figures(N, basis_figures, src_image_gray)
    print(len(result))
    for ans in result:
        answer_list = list(ans)
        print(*answer_list, sep=', ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Serve the application')
    parser.add_argument('-i', '--image', default='./test_300_200.png')
    parser.add_argument('-s', '--structure', default='./002_line_in.txt')
    args = parser.parse_args()
    main(args)
