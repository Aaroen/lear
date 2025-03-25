class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
            n = len(nums)
            if n <= 2:
                return n
            new = 2
            for i in range(2,n):
                if nums[i] != nums[new-2]:
                    nums[new] = nums[i]
                    new +=1
            return new
            # 为什么比较 nums[fast] 和 nums[slow - 2]？ 目的是确保对于每个数字，最多保留两个。
            #  nums[slow - 2] 代表了在修改后的数组中，当前考虑的数字（或者之前的数字）倒数第二次出现的位置的值。 
            # 如果 nums[fast] 大于 nums[slow - 2]，这意味着 nums[fast] 是一个新的数字，或者即使和之前的数字相同，我们也可以再保留一个，
            # 因为它不会导致同一个数字出现超过两次。 
            # 因为数组是排序的，如果 nums[fast] > nums[slow-2]，
            # 意味着 nums[fast] 一定比 nums[slow-2] 以及之前的所有 nums[slow-3], nums[slow-4]... 都大或者相等。 
            # 如果相等，因为数组已经排过序，那么 nums[slow-2] 应该是和 nums[fast] 相同的数字，那么可以继续保留。