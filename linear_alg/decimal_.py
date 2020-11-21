from decimal import Decimal


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-5):
        return abs(self) < eps