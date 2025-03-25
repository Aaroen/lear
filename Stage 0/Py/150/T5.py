class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 初始化候选多数元素为数组的第一个元素
        candidate = nums[0]
        # 初始化计数器为 1
        count = 1

        # 从第二个元素开始遍历数组
        for num in nums[1:]:
            if num == candidate:
                # 如果当前元素与候选元素相同，计数器加 1
                count += 1
            else:
                # 如果当前元素与候选元素不同，计数器减 1
                count -= 1
                if count == 0:
                    # 如果计数器变为 0，更换候选元素为当前元素，并将计数器重置为 1
                    candidate = num
                    count = 1

        return candidate
    """
        找出数组中的多数元素。

        多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
        假设数组非空且总是存在多数元素。

        算法思路：摩尔投票法 (Boyer-Moore Majority Vote Algorithm)

        1. 初始化：
           - candidate: 候选多数元素，初始设为数组的第一个元素。
           - count: 计数器，初始设为 1。

        2. 遍历数组（从第二个元素开始）：
           - 如果当前元素与 candidate 相同，则增加 count。
           - 如果当前元素与 candidate 不同，则减少 count。
           - 如果 count 变为 0，则更换 candidate 为当前元素，并将 count 重置为 1。

        3. 返回 candidate：
           - 遍历结束后，candidate 即为多数元素。

        原理：
        摩尔投票法的核心思想是在遍历数组时，通过“抵消”不同元素的计数来筛选出可能的多数元素。
        由于多数元素出现的次数超过数组长度的一半，即使多数元素与其他元素进行“抵消”，
        最终多数元素仍然会保留下来成为候选人。

        参数:
        nums: 整数数组

        返回:
        多数元素 (整数)
        """
