# class Solution:
#     def myPow(self, x: float, n: int) -> float:
#         result = 1
#         if n > 0:
#             for i in range (n):
#                 result = result * x
#             return result
#         elif n == 0:
#             return 1
#         else:
#             n = -n
#             for i in range (n):
#                 result = result * x
#             return 1/result
            

class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1.0
        if n < 0:
            x = 1.0 / x
            n = -n

        result = 1.0
        while n > 0:
            if n % 2 == 1:
                result *= x
            x *= x
            n //= 2
        return result
