from .vector import Vector
from .decimal_ import MyDecimal
import numpy as np


class Lines2D(object):
    """A class for finding Intersections,
          Coefficients, and other properties
          for lines in 2-Dimensions.
    """
    X, Y = None, None

    def __init__(self, coefficients, constant_term):
        self.coefficients = coefficients
        self.constant_term = constant_term
        self.dimension = 2
        self.normal_vector = Vector(self.coefficients)

        num = [int, float, np.int32, np.float32, np.int64, np.float64]
        assert self.dimension == len(self.coefficients), 'len(coefficients) Must Be == 2!'
        assert type(self.constant_term) in num, 'Constant-Term Must Be a Number!'
        for i in self.coefficients:
            assert type(i) in num, 'Coefficients  Must Be Numerical!'

    def is_parallel_to(self, other):
        """Confirm if two lines are parallel
        Two lines are parallel if the coefficients
        of one line is a scalar multiple of the
        other line, irrespective of absolute value
        :param other: A line with
        :return:
        """
        assert self.dimension == other.dimension, 'Dimensions Must Be Equal!'

        a, b = self.normal_vector, other.normal_vector
        if a.is_zero_vector(b):
            return True

        val = round((a.coordinates[0] / b.coordinates[0]), 2)
        for x, y in zip(a.coordinates, b.coordinates):
            try:
                assert round((x / y), 2) == val
            except AssertionError:
                return False

        return True

    def dir_vec(self):
        """Given a line, find it's
            direction vector
            The direction vector in 2D
            can be got by reversing the
            coefficients of a line and
            negating one
        :return: a tuple, the direction vector
        """

        # First assert i`t's not the zero-vector
        try:
            assert not MyDecimal.is_near_zero(sum(self.coefficients))
        except AssertionError:
            return 0

        a = self.normal_vector
        x, y = self.coefficients[-1], -(self.coefficients[0])
        b = Vector([x, y])

        if a.is_orthogonal_to(b):
            return b

        return None

    def find_point(self):
        """Given a line in 2D
            find any given point.
            Specifically, find the value
            of y, if x is 0. (y-intercept)
        :return: a Tuple of x,y coordinates
        """
        # Let's assume x = 0
        # to find y, we substitute
        x = 0
        y = round(self.constant_term / self.coefficients[-1], 3)
        return x, y

    def find_point2(self):
        """Given a line in 2D
            find any given point.
            Specifically, find the value
            of x, if y is 0. (x-intercept)
        :return: a Tuple of x,y coordinates
        """
        # Let's assume y = 0
        # to find , we substitute
        y = 0
        x = round(self.constant_term / self.coefficients[0], 3)
        return x, y

    def find_slope_and_intercept(self):
        """Find the slope and intercept of
            a linear equation

        :return:
        """
        point1 = self.find_point()
        point2 = self.find_point2()

        slope = round((point1[-1] - point2[-1]) / (point1[0] - point2[0]), 4)
        intercept = round(point1[-1], 4)

        return slope, intercept

    def is_equal_to(self, other):
        """Assert that two parallel lines
            are equal and the same
        Two parallel lines in 2D are equal/same if
        The direction vector of one line
        is orthogonal to the normal vector of the
        other line.
        :param other: a line with same dim as self
        :return: True or False
        """
        try:
            assert self.is_parallel_to(other)
        except AssertionError:
            return False
        # find one point on self and other
        self_point = self.find_point()
        other_point = other.find_point()

        # find the vector between those points
        points_vec = (self_point[0] - other_point[0], self_point[1] - other_point[1])
        points_vec = Vector(points_vec)

        return points_vec.is_orthogonal_to(other.normal_vector)

    def radians(self, vec2):
        """This method calculates the angle between two lines
                in radians and returns a float.

        The angle is the arc-cosine of the dot-product of the two,
        vectors, divided by the product of their magnitudes.
        """

        return self.normal_vector.radians(vec2.normal_vector)

    def degrees(self, vec2):
        """This method calculates the angle between two lines
                in degrees and returns a float.

        Simply call the radians method on the vectors
        and multiply the radians value by (180/pi) to get degrees.
        """

        return self.normal_vector.degrees(vec2.normal_vector)


if __name__ == '__main__':
    one = Lines2D((4.046, 2.836), 1.21)
    two = Lines2D((10.115, 7.09), 3.025)

    three = Lines2D((7.204, 3.182), 8.68)
    four = Lines2D((8.172, 4.114), 9.883)

    five = Lines2D((1.182, 5.562), 6.744)
    six = Lines2D((1.773, 8.343), 9.525)

