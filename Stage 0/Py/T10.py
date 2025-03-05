from typing import List

class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 2:
            return True
        up = True
        down = True
        for i in  range(n-1):
            if nums[i] >= nums[i+1]:
                down = False
            if nums[i] <= nums[i+1]:
                up = False
        
        return up or down