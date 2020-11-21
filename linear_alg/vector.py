import math


class Vector(object):
    """ Class for creating and manipulating vector objects."""

    def __init__(self, coordinates):
        """ Initialise the Vector Class

        @Param:
        coordinates are vector coordinates that:-
        Must be a list or tuple, containing int or float elements.

        """
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = list(coordinates)
            self.dimension = len(coordinates)
        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, other):
        return self.coordinates == other.coordinates

    def add(self, *args):
        """ Add arbitrary number of vectors

        @:param: args must be vectors of equal dimensions
        @:return: a vector of sum of all coordinates
        """
        try:
            for i in args:
                assert self.dimension == i.dimension
                count = 0
                for j in i.coordinates:
                    self.coordinates[count] += j
                    count += 1
        except AssertionError as e:
            return e, 'vectors must have same dimensions'

        return self

    def minus(self, *args):
        """ Minus arbitrary number of vectors

        @:param: args must be vectors of equal dimensions
        @:return: a vector of subtraction of all coordinates
        """
        try:
            for i in args:
                assert self.dimension == i.dimension
                count = 0
                for j in i.coordinates:
                    self.coordinates[count] -= j
                    count += 1
        except AssertionError as e:
            return e, 'vectors must have same dimensions'

        return self

    def scalar_multiply(self, scalar):
        """ Scalar multiply a vector

        Note that multiplying a vector by a negative number,
        causes the vector to point in the opposite direction,
        as well as possibly changing its' magnitude.

        :param: scalar must be an int or float, pos or neg.
        :return: self.coordinates times scalar
        """

        for i in range(len(self.coordinates)):
            self.coordinates[i] *= scalar
            self.coordinates[i] = round(self.coordinates[i], 6)

        return self

    def magnitude(self):
        """Compute the magnitude or length of a vector.

        The magnitude of a vector is the square root of,
        calling the dot-product on itself.

        :return: Returns a scalar of type int or float
        """
        dot_multiply = sum([i**2 for i in self.coordinates])
        magnitude = math.sqrt(dot_multiply)

        return round(magnitude, 6)

    def normalize(self):
        """Compute the unit vector.

        First, compute the vector magnitude then,
        multiply the inverse magnitude by the vector.

        :return: Return the unit vector
        """
        magnitude = self.magnitude()

        # If magnitude > 0, meaning not the zero vector,
        # then return unit vector, else return 0
        if magnitude:
            inv = 1 / magnitude
            return self.scalar_multiply(inv)
        else:
            return 0

    def is_zero_vector(self, vec2, tolerance=1e-7):
        """Check if either vector is a zero-vector

        The zero-vector is a vector that has a
        magnitude of zero. It is both parallel and
        orthogonal to itself and all other vectors.

        :param vec2: a vector with same dimension as self
        :param tolerance: A minute floating limit to accommodate
                        small floating point differences
        :return: True or false
        """
        try:
            assert self._has_equal_dim(vec2)
        except AssertionError as e:
            return e

        return abs(self.magnitude()) < tolerance or abs(vec2.magnitude()) < tolerance

    def _has_equal_dim(self, vec2):
        """Asserts two vectors have same dimension

        :param vec2: A vector that should have same dim as self
        :return: returns True or False
        """

        return self.dimension == vec2.dimension

    def dot_product(self, vec2):
        """This method returns the dot-product between 2 vectors.

        The dot-product is the sum of element-wise multiplication
        on both vectors. It's a number

        @:param: vec2 is a vector of equal dimension with self
        @:return: A float or int
        """
        try:
            if self.is_zero_vector(vec2):
                return 0
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        def dot_multiply(vec_x, vec_y):
            """A recursive function to multiply
            corresponding elements of two vectors

            :param vec_x: the first vector
            :param vec_y: the second vector
            :return: A new vector with each element a
                     product of corresponding elements
            """
            new_list = []
            if not vec_x:
                return new_list
            new_list.append(vec_x[0] * vec_y[0])
            return new_list + dot_multiply(vec_x[1:], vec_y[1:])

        dot_product = round(sum(dot_multiply(self.coordinates, vec2.coordinates)), 6)

        return dot_product

    def radians(self, vec2):
        """This method calculates the angle between two vectors
                in radians and returns a float.

        The angle is the arc-cosine of the dot-product of the two,
        vectors, divided by the product of their magnitudes.
        """
        try:
            if self.is_zero_vector(vec2):
                return 0
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        dot_product = round(self.dot_product(vec2))
        magnitudes_multiply = round(self.magnitude() * vec2.magnitude())
        theta = round(math.acos(dot_product / magnitudes_multiply), 6)
        return theta

    def degrees(self, vec2):
        """This method calculates the angle between two vectors
            in degrees and returns a float.

        Simply call the radians method on the vectors
        and multiply the radians value by (180/pi) to get degrees.
        """
        radian = self.radians(vec2)
        degree = round(radian * (180 / math.pi), 6)

        return degree

    def is_parallel(self, vec2):
        """Check if one vector is a scalar multiple,
        of the other vector and vice versa

        Two vectors are parallel if one is a scalar
        multiple of the other. If the scalar is
        a negative number, then both vectors will be
        opposite and have angle of 180 degrees between.
        else, both vectors will have angle of 0 degrees
        as they point in the same direction.

        :param vec2: A vector of same dimension as self
        :return: Return a boolean True or False
        """
        try:
            if self.is_zero_vector(vec2):
                return True
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        is_180 = round(self.degrees(vec2)) == 180
        is_zero = round(self.degrees(vec2)) == 0

        return (is_180 + is_zero) == 1

    def is_orthogonal(self, vec2):
        """Check if two vectors are orthogonal

        Two vectors are orthogonal if their dot-product is 0,
        this usually happens if one is a zero-vector or they are at
        right angles to each other

        :param vec2: Vector with same dimension as self
        :return: True or False
        """
        try:
            if self.is_zero_vector(vec2):
                return True
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        return round(self.degrees(vec2)) == 90

    def v_parallel(self, vec2):
        """Find the component of self parallel to
        the basis vector vec2, given that self is
        projected on vec2.

        To compute v_parallel, we multiply unit_vector vec2
        by the dot-product of (unit_vector vec2 and self)

        :param vec2: A vector with same dimension as self
        :return: A vector (v_parallel)
        """
        try:
            if self.is_zero_vector(vec2):
                return 0
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        unit_vec2 = vec2.normalize()
        self_dot_unit_vec2 = self.dot_product(unit_vec2)
        v_para = unit_vec2.scalar_multiply(self_dot_unit_vec2)

        return v_para

    def v_perp(self, vec2):
        """ Find the component of self orthogonal to
        vec2, given that self is projected on vec2

        Any non-zero vector can be represented as the
        sum of its component orthogonal/perpendicular
        to the basis vector and its component
        parallel to the basis vector

        :param vec2: Vector with same dimension as self
        :return: Vector (v_perp)
        """
        try:
            if self.is_zero_vector(vec2):
                return 0
            assert self._has_equal_dim(vec2)
        except AssertionError:
            return 'ERROR: Both dimensions must be equal!'

        v_parallel = self.v_parallel(vec2)
        v_perp = self.minus(v_parallel)

        return v_perp

    def cross_product(self, vec2):
        """Find the cross-product of self and vec2

        The cross product is non-commutative.
        It is only applicable to vectors in 3 dimensions
        and is useful for computing the area of the parallelogram
        spanned by these 2 vectors. Cross product must be orthogonal
        to both vectors. Orthogonality assertion is included here.

        :param vec2: vector with 3 dimension equal to self
                    if dim < 3, append dim of 0.
        :return: Vector (cross-product)
        """
        try:
            assert self.dimension <= vec2.dimension <= 3
            while True:
                if self.dimension + vec2.dimension == 6:
                    break
                if self.dimension < 3:
                    self.coordinates.append(0)
                    self.dimension += 1
                if vec2.dimension < 3:
                    vec2.coordinates.append(0)
                    vec2.dimension += 1
        except AssertionError:
            return 'DimensionError: dim must be <= 3!'
        x1, y1, z1 = self.coordinates
        x2, y2, z2 = vec2.coordinates

        cross_vec = Vector([0, 0, 0])
        cross_vec.coordinates[0] = (y1 * z2) - (y2 * z1)
        cross_vec.coordinates[1] = -((x1 * z2) - (x2 * z1))
        cross_vec.coordinates[2] = (x1 * y2) - (x2 * y1)
        try:
            assert self.is_orthogonal(cross_vec)
            assert vec2.is_orthogonal(cross_vec)
        except AssertionError:
            return 'Cross-Vector must be orthogonal to both vectors!'

        return cross_vec

    def area_of_parallelogram(self, vec2):
        """Return the area of the parallelogram
        spanned by two vectors

        The area of the parallelogram spanned by
        two vectors is simply the magnitude of
        the cross-product of these two vectors.

        :param vec2: vector of no more than 3 dimension
                    same as self.
        :return: A number (magnitude of cross-product)
        """
        cross_vector = self.cross_product(vec2)
        parallelogram_area = cross_vector.magnitude()

        return round(parallelogram_area, 6)

    def area_of_triangle(self, vec2):
        """Return the area of the triangle
        spanned by two vectors

        The area of the triangle spanned by
        two vectors is simply the cross product
        of these two vectors, divided by 2.

        :param vec2: vector of no more than 3 dimension
                    same as self.
        :return: A number (area of triangle of 2 vectors)
        """
        cross_vector = self.cross_product(vec2)
        area_of_parallelogram = cross_vector.magnitude()
        triangle_area = area_of_parallelogram / 2.

        return round(triangle_area, 6)