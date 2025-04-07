from typing import List


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        请勿返回任何内容，直接修改 nums1。
        """
        # 初始化三个指针：
        # p1 指向 nums1 中已排序部分的末尾
        # p2 指向 nums2 的末尾
        # p 指向 合并后数组 nums1 的末尾 (从后往前填充)
        p1 = m - 1
        p2 = n - 1
        p = m + n - 1

        # 当 p1 和 p2 都还指向有效元素时，进行循环
        while p1 >= 0 and p2 >= 0:
            # 比较 nums1[p1] 和 nums2[p2] 的大小
            if nums1[p1] >= nums2[p2]:
                # 如果 nums1[p1] 大于等于 nums2[p2]，则将 nums1[p1] 放入合并数组的末尾
                nums1[p] = nums1[p1]
                # p1 向前移动一位
                p1 -= 1
            else:
                # 如果 nums2[p2] 大于 nums1[p1]，则将 nums2[p2] 放入合并数组的末尾
                nums1[p] = nums2[p2]
                # p2 向前移动一位
                p2 -= 1
            # p 指针向前移动一位，指向下一个要填充的位置
            p -= 1

        # 循环结束后，可能 nums2 中还有剩余元素没有被合并到 nums1 中
        # 因为 nums2 是排序的，且要合并到 nums1 的前部，所以如果 nums2 还有剩余，
        # 剩余元素一定是 nums2 中最小的那些元素，直接将 nums2 剩余的元素复制到 nums1 的头部即可
        while p2 >= 0:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1
        # 此时 nums1 就已经合并了 nums2，并且保持了非递减顺序
        # （ps:该while循环用于处理sum2数组后处理而未被合并入sum1数组的情况；sum1数组后处理时无需再进行操作：因为全部数据位置已经合并正确）

# 示例1
nums1 = [1,2,3,0,0,0]
m = 3
nums2 = [2,5,6]
n = 3
Solution().merge(nums1, m, nums2, n)
print("示例1合并结果:", nums1)
