from .ge_helper import GeHelper
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


class GaussianElimination(GeHelper):
    """A class to solve the problem of
        linear intersections for linear
        objects/planes in n-dimensional-space
        where n >= 3, using the
        Gaussian-Elimination Algorithm
    """
    No_Solution = 'INCONSISTENT:(system of equations)'
    Infinite_No_Variables = 'INFINITE INTERSECTIONS:(No Pivot and Free Variables)'

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
        self.compute_triangular_form()
        self.round_off()
        start = self.dimension - 1

        while start:
            lists = [True if i.coefficients[start] else False for i in self.plane_objects]
            lists.reverse()
            base_mark = True
            count = len(self.plane_objects)-1
            x, y, base_row, base = [None]*4

            for val in lists:
                if val and base_mark:
                    base_row = count
                    base = self.plane_objects[base_row]
                    x = base.coefficients[start]
                    base_mark = False
                elif val:
                    y = self.plane_objects[count].coefficients[start]
                    self.multiply_row(count, x), self.multiply_row(base_row, y)
                    self.subtract_rows(base_row, count)
                    self.plane_objects[base_row] = base
                count -= 1
            start -= 1

        self.round_off()  # Round-off each coefficient to max 4 D.P

        if self.is_inconsistent():
            print(self)
            return self.No_Solution

        # Now divide each equation by its pivot variable to get
        # the value of the pivot and any free variables.

        check = self.first_non_zero_index()

        for ind, val in enumerate(check):
            if val == -float('inf'):
                continue
            v = self.plane_objects[ind].coefficients[val]
            self.divide_row(ind, v)

        self.round_off(3)
        return self

    def unique_intersection(self):
        """Confirm if SoE has a unique point
        of intersection

        This occurs when there are only distinct
        pivot variables and the number of equations
        are at least the number of dimension.

        :return: Return the tuple of unique points
                or False
        """
        x = deepcopy(self)
        x = x.compute_rref()
        points = []
        for ind, plane in enumerate(x.plane_objects):
            if sum(plane.coefficients) <= 1:
                points.append(plane.constant_term)
            else:
                return 0

        return tuple(points)

    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {},{}'.format(i+1, p.coefficients, p.constant_term)
                for i, p in enumerate(self.plane_objects)]
        ret += '\n'.join(temp)
        return ret

    def plot_lines(self):
        slopes, intercepts = [], []
        intersect = self.unique_intersection()

        for plane in self.plane_objects:
            for slope, intercept in [plane.find_slope_and_intercept()]:
                slopes.append(slope), intercepts.append(intercept)

        x = np.linspace(-10, 10, 500)

        v1, v2, k = -5, 5, 0

        if intersect:
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
        if intersect:
            print('Intersection:', (intersect[0], intersect[1]))
            plt.scatter(intersect[0], intersect[1], color='black')
            plt.annotate('intersect', (intersect[0]+0.1, intersect[1]+0.1))

        plt.grid(linestyle='dotted')
        plt.show()


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
