from typing import List
class Solution:
    def maxArea(self, height: list[int]) -> int:
        left, right = 0, len(height) - 1  
        max_area = 0                       
        while left < right :
            low = min(height[left],height[right])
            now_area = low*(right-left)
            max_area = max(low*(right-left),max_area)
            if height[left] > height[right]:
                right -= 1
            else:
                left += 1
        return max_area

# class Solution:
#     def maxArea(self, height: List[int]) -> int:
#         left, right = 0, len(height) - 1  # 初始化左右指针
#         max_area = 0                       # 初始化最大面积

#         while left < right:
#             width = right - left             # 计算宽度
#             current_height = min(height[left], height[right]) # 计算高度（取较矮的线）
#             current_area = width * current_height  # 计算当前面积
#             max_area = max(max_area, current_area) # 更新最大面积

#             if height[left] < height[right]:
#                 left += 1  # 左边线更矮，向右移动左指针
#             else:
#                 right -= 1 # 右边线更矮或相等，向左移动右指针

#         return max_area


height1 = [1,8,6,2,5,4,8,3,7]
solution = Solution()
result1 = solution.maxArea(height1)
print(f"输入: {height1}, 输出: {result1}, 是否正确: {result1 == 49}") # 输出: 输入: [1, 8, 6, 2, 5, 4, 8, 3, 7], 输出: 49, 是否正确: True
