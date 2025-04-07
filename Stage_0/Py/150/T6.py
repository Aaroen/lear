from typing import List

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        def reverse(arr, start, end):
            # 内部函数，用于反转数组的指定区间
            while start < end:
                arr[start], arr[end] = arr[end], arr[start]
                start += 1
                end -= 1
        
        n = len(nums)
        k %= n  # 处理k大于数组长度的情况，取模简化操作次数
        if k == 0:
            return  # 若k为0，无需操作
        
        # 三步反转法
        reverse(nums, 0, n-1)        # 1. 反转整个数组
        reverse(nums, 0, k-1)        # 2. 反转前k个元素
        reverse(nums, k, n-1)        # 3. 反转剩下的元素

# 调用示例
if __name__ == "__main__":
    solution = Solution()
    
    # 示例1
    nums1 = [1,2,3,4,5,6,7]
    solution.rotate(nums1, 3)
    print("示例1输出:", nums1)  # 输出: [5,6,7,1,2,3,4]
    
    # 示例2
    nums2 = [-1,-100,3,99]
    solution.rotate(nums2, 2)
    print("示例2输出:", nums2)  # 输出: [3,99,-1,-100]
