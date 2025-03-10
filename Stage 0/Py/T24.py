from typing import List


class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        money_5 = 0
        money_10 = 0
        for bill in bills:
            if bill == 5:
                money_5 += 1
            elif bill == 10:
                if money_5 == 0:
                    return False
                else:
                    money_10 += 1
                    money_5 -= 1
            elif bill == 20:
                if money_10 >= 1 and money_5 >= 1:
                    money_5 -= 1
                    money_10 -= 1
                elif money_10 == 0 and money_5 >= 3:
                    money_5 -= 3
                else:
                    return False                
        return True
    
solution = Solution()
print(solution.lemonadeChange([5,5,10,20,5,5,5,5,5,5,5,5,5,10,5,5,20,5,20,5]))
