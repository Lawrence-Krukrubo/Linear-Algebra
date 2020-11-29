from .ge_helper import GeHelper
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import string
from .vector import Vector


class GaussianElimination(GeHelper):
    """A class to solve the problem of
        linear intersections for linear
        objects/planes in n-dimensional-space
        where n >= 3, using the
        Gaussian-Elimination Algorithm
    """
    alpha = string.ascii_uppercase
    alpha = alpha[-3:] + alpha[:-3]

    No_Solution = 'INCONSISTENT:(System of Equations with No Solution)'
    Infinite_Solution = 'INFINITE INTERSECTIONS:(System of Equations with Infinite Solutions)'
    Unique_Solution = 'INTERSECTION:(System of Equations with One Unique Solution)'

    def compute_triangular_form(self):
        """Compute and return the triangular
            form of a system of equations

        :return: The triangular form of a
                System of Equations.
        """
        start = 1
        while True:
            self.sort_plane_objects()
            check_list = self.first_non_zero_index()

            if start == len(check_list):
                break

            for ind, value in enumerate(check_list):
                if value == start-1 and ind >= start:
                    base = self.plane_objects[start-1]
                    x = base.coefficients[start-1]
                    y = self.plane_objects[ind].coefficients[start-1]
                    self.multiply_row(start-1, y), self.multiply_row(ind, x)
                    self.subtract_rows(start-1, ind)

                    self.plane_objects[start-1] = base
            start += 1

    def compute_rref(self):
        """Coding the Reduced-Row-Echelon-Form
            by transforming the Triangular Form
            to unique variables per system of equation

        :return: A reduced system of equations
        """
        # Now compute the triangular form
        self_ = deepcopy(self)
        self_.compute_triangular_form()
        self_.round_off()
        start = self_.dimension - 1

        while start:
            lists = [True if i.coefficients[start] else False for i in self_.plane_objects]
            lists.reverse()
            base_mark = True
            count = len(self_.plane_objects)-1
            x, y, base_row, base = [None]*4

            for val in lists:
                if val and base_mark:
                    base_row = count
                    base = self_.plane_objects[base_row]
                    x = base.coefficients[start]
                    base_mark = False
                elif val:
                    y = self_.plane_objects[count].coefficients[start]
                    self_.multiply_row(count, x), self_.multiply_row(base_row, y)
                    self_.subtract_rows(base_row, count)
                    self_.plane_objects[base_row] = base
                count -= 1
            start -= 1

        self_.round_off()  # Round-off each coefficient to max 4 D.P

        if self_.is_inconsistent():
            return self_

        # Now divide each equation by its pivot variable to get
        # the value of the pivot and any free variables.

        check = self_.first_non_zero_index()

        for ind, val in enumerate(check):
            if val == -float('inf'):
                continue
            v = self_.plane_objects[ind].coefficients[val]
            self_.divide_row(ind, v)

        self_.round_off(3)
        return self_

    def unique_intersection(self):
        """Confirm if SoE has a unique point
        of intersection

        This occurs when there are only distinct
        pivot variables and the number of equations
        are at least the number of dimension.

        :return: Return the tuple of unique points
                or False
        """
        p = self.compute_rref()
        non_zero_index = p.first_non_zero_index()

        points = []
        for ind, plane in enumerate(p.plane_objects):
            if sum(plane.coefficients) <= 1:
                points.append(plane.constant_term)
            else:
                x = plane.coefficients[non_zero_index[ind]] == 1
                if x:
                    # It's same line
                    return float('inf')
                # Or, No intersection
                return None

        return tuple(points)

    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {},{}'.format(i+1, p.coefficients, p.constant_term)
                for i, p in enumerate(self.plane_objects)]
        ret += '\n'.join(temp)
        return ret

    def plot_lines(self):
        """Plot lines2D objects in 2D or
        Planes3D objects in 3D.

        If there's an intersection, lines/planes
        are plotted to show the intersection point.

        :return: None (Plots lines/planes in 2D or 3D)
        """
        slopes, intercepts = [], []
        intersect = self.unique_intersection()

        for plane in self.plane_objects:
            for slope, intercept in [plane.find_slope_and_intercept()]:
                slopes.append(slope), intercepts.append(intercept)

        x = np.linspace(-10, 10, 500)

        v1, v2, k = -5, 5, 0

        if type(intersect) is tuple:
            kmin = min(intersect)
            kmax = max(intersect)
            v1, v2 = kmin-3, kmax+3

        plt.xlim(v1, v2)
        plt.ylim(v1, v2)

        count = 1
        for slope, intercept in zip(slopes, intercepts):
            plt.plot(x, x * slope + intercept, label=f"Line: {count}")
            count += 1
        plt.legend(loc="upper right")

        plt.title('Linear System of Equations', fontsize=12)
        if type(intersect) is tuple:
            print('Intersection:', (intersect[0], intersect[1]))
            plt.scatter(intersect[0], intersect[1], color='black')
            plt.annotate('intersect', (intersect[0]+0.1, intersect[1]+0.1))
        elif type(intersect) is float:
            print('Infinitely Many Solutions:')
            plt.annotate('Infinite', (0, intercepts[0]+0.1))
        else:
            print('No Intersection: (no solution)')
            plt.annotate('None', (0, slopes[-1]+0.1))

        plt.grid(linestyle='dotted')
        plt.show()

    @staticmethod
    def plot_point(x=None, y=None, z=None):
        """Plot a point in 2D or 3D

        Example: GE.plot_point(2, 3) # 2D
                GE.plot_point(2,3,4) # 3D

        :param x: The x coordinate(Int or Float)
        :param y: The y coordinate(Int or Float)
        :param z: The z coordinate(Int or Float)
        :return: None (Just plots the point)
        """
        assert x is not None and y is not None, 'Points Must Have 2 or 3 Coordinates'

        title_dict = {'size': 14, 'weight': 'bold'}
        label_dict = {'size': 12, 'weight': 'bold'}
        plt.style.use('seaborn-white')

        if z is None:
            plt.scatter(x, y)
            plt.xlim(min(x, y) - 3, max(x, y) + 3)
            plt.ylim(min(x, y) - 3, max(x, y) + 3)
            plt.title(f'Point in 2D: (X={x}, Y={y})', fontdict=title_dict)
            plt.xlabel('X', fontdict=label_dict)
            plt.ylabel('Y', fontdict=label_dict, rotation=1.4)
            plt.grid(linestyle='dotted')

        else:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x, y, z, c='r', marker='o')
            ax.set_xlabel('X', fontdict=label_dict)
            ax.set_ylabel('Y', fontdict=label_dict)
            ax.set_zlabel('Z', fontdict=label_dict)
            ax.set_title(f'Point in 3D: (X={x}, Y={y}, Z={z})', fontdict=title_dict)

        plt.show()

    @staticmethod
    def plot_points_2_vec(point1, point2):
        """Given two Points in 2D or 3D,
        plot a Vector from the 1st point
        to the 2nd point

            For Example in 2D:
                    point1 = (1, 2)
                    point2 = (3, 4)
                    GE.plot_points_2_vec(point1, point2)

            For Example in 3D:
                    point1 = (1, 2, 3)
                    point2 = (3, 4, 5)
                    GE.plot_points_2_vec(point1, point2)

        :param point1: A tuple/triple of Int or Float
        :param point2: A tuple/triple of Int or Float
        :return: None (Plots the vector connecting the points)
        """
        assert 2 <= len(point1) == len(point2) <= 3, "Dimensions must be 2 or 3 and Equal!"

        sample = [round(i, 1) for i in point1]
        temp = [round(i, 1) for i in point2]
        head_length = 0.5
        x = [j - k for j, k in zip(temp, sample)]

        x_mag = Vector(x).magnitude()
        dx, dy = [i/x_mag for i in x]
        x_mag = x_mag - head_length

        ax = plt.axes()
        title_dict = {'size': 14.5, 'weight': 'bold'}
        label_dict = {'size': 12.5, 'weight': 'bold'}
        plt.style.use('seaborn-white')

        if len(sample) == 2:
            ax.arrow(sample[0], sample[1], dx*x_mag, dy*x_mag,
                     head_width=0.4, head_length=head_length, fc='red', ec='black')
            plt.grid()
            if x[1] < 0 <= x[0]:
                plt.xlim(sample[0] - 1, sample[0] + x[0] + 1)
                plt.ylim((sample[1] + x[1]) - 1, sample[1] + 1)
                plt.annotate(f'({sample[0]}, {sample[1]})', (sample[0], sample[1] + 0.15), fontweight='bold')
                plt.annotate(f'({temp[0]}, {temp[1]})', (temp[0], temp[1] - 0.25), fontweight='bold')
            elif x[1] >= 0 <= x[0]:
                plt.xlim(sample[0] - 1, sample[0] + x[0] + 1)
                plt.ylim(sample[1] - 1, sample[1] + x[1] + 1)
                plt.annotate(f'({sample[0]}, {sample[1]})', (sample[0], sample[1] - 0.15), fontweight='bold')
                plt.annotate(f'({temp[0]}, {temp[1]})', (temp[0], temp[1] + 0.15), fontweight='bold')
            elif x[0] < 0 <= x[1]:
                plt.xlim((sample[0] + x[0]) - 1, sample[0] + 1)
                plt.ylim(sample[1] - 1, (sample[1] + x[1]) + 1)
                plt.annotate(f'({sample[0]}, {sample[1]})', (sample[0], sample[1] + 0.15), fontweight='bold')
                plt.annotate(f'({temp[0]}, {temp[1]})', (temp[0], temp[1] + 0.15), fontweight='bold')
            else:
                plt.xlim((sample[0] + x[0]) - 1, sample[0] + 1)
                plt.ylim((sample[1] + x[1]) - 1, sample[1] + 1)
                plt.annotate(f'({sample[0]}, {sample[1]})', (sample[0], sample[1] - 0.15), fontweight='bold')
                plt.annotate(f'({temp[0]}, {temp[1]})', (temp[0], temp[1] - 0.25), fontweight='bold')

            plt.scatter([point1[0], point2[0]], [point1[1], point2[1]], color='black')
            plt.title(f'Vector({x[0]},{x[1]}): X= {x[0]}, Y= {x[1]}', fontdict=title_dict)
            plt.xlabel('X', fontdict=label_dict)
            plt.ylabel('Y', fontdict=label_dict, rotation=1.4)

        plt.show()

    def summary(self):
        """Give a Summary of the
        System of Equation

        :return:
        """
        x = deepcopy(self)
        y = deepcopy(self)

        # For Inconsistent Equations
        x.compute_triangular_form()
        if x.is_inconsistent():
            return self.No_Solution

        # For Unique Intersection
        if y.unique_intersection():
            intersection = y.unique_intersection()
            if type(intersection) is tuple:
                d = {}
                for ind, item in enumerate(intersection):
                    d[y.alpha[ind]] = item
                print(y.Unique_Solution)
                return d
            # For infinite Intersections
            else:
                return self.Infinite_Solution

# if __name__ == '__main__':
#     # CODING GE-SOLUTION
#     p1 = Planes3D((5.862, 1.178, -10.366), -8.15)
#     p2 = Planes3D((-2.931, -0.589, 5.183), -4.075)
#
#     p3 = Planes3D((8.631, 5.112, -1.816), -5.113)
#     p4 = Planes3D((4.315, 11.132, -5.27), -6.775)
#     p5 = Planes3D((-2.158, 3.01, -1.727), -0.831)
#
#     p6 = Planes3D((5.262, 2.739, -9.878), -3.441)
#     p7 = Planes3D((5.111, 6.358, 7.638), -2.152)
#     p8 = Planes3D((2.016, -9.924, -1.367), -9.278)
#     p9 = Planes3D((2.167, -13.543, -18.883), -10.567)
#
#     # CODING-PARAMETRIZATION
#     p10 = Planes3D((0.786, 0.786, 0.588), -0.714)
#     p11 = Planes3D((-0.138, -0.138, 0.244), 0.319)
#
#     m = np.array([[8.631, 5.112, -1.816], [4.315, 11.132, -5.27], [-2.158, 3.01, -1.727]])
#     b = np.array([-5.113, -6.775, -0.831])
#     print(np.linalg.solve(m, b))
#
#     p12 = Planes3D((8.631, 5.112, -1.816), -5.113)
#     p13 = Planes3D((4.315, 11.132, -5.27), -6.775)
#     p14 = Planes3D((-2.158, 3.01, -1.727), -0.831)
#
#     p15 = Planes3D((0.935, 1.76, -9.365), -9.955)
#     p16 = Planes3D((0.187, 0.352, -1.873), -1.991)
#     p17 = Planes3D((0.374, 0.704, -3.746), -3.982)
#     p18 = Planes3D((-0.561, -1.056, 5.619), 5.973)
#
#     # MORE GAUSSIAN-ELIMINATION PRACTICE
#     p19 = Planes3D((1, -2, 1), -1)
#     p20 = Planes3D((1, 0, -2), 2)
#     p21 = Planes3D((-1, 4, -4), 0)
#
#     p22 = Planes3D((0, 1, -1), 2)
#     p23 = Planes3D((1, -1, 1), 2)
#     p24 = Planes3D((3, -4, 1), 1)
#
#     pa = Planes3D((4, 2, 1), 11)
#     pb = Planes3D((-2, 4, -2), -16)
#     pc = Planes3D((1, -2, 4), 17)
#
#     one = Lines2D((4.046, 2.836), 1.21)
#     two = Lines2D((10.115, 7.09), 3.025)
#
#     objects = [one, two]
#     ge = GaussianElimination(2, objects)
#     print(ge.compute_rref())
