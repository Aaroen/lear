import random
class RandomizedSet:

    def __init__(self):
        self.nums = []
        self.valToIndex = {}

    def insert(self, val: int) -> bool:
        if val in self.valToIndex:
            return False
        self.nums.append(val)
        self.valToIndex[val] = len(self.nums) - 1
        return True

    def remove(self, val: int) -> bool:
        if val not in self.valToIndex:
            return False
        index = self.valToIndex[val]
        last = self.nums[-1]
        self.nums[index] = last
        self.valToIndex[last] = index
        del self.valToIndex[val]
        self.nums.pop()
        
        return True

    def getRandom(self) -> int:
        return random.choice(self.nums)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

# 调用示例和解释
randomizedSet = RandomizedSet()
print(randomizedSet.insert(1))  # 输出: true，集合现在包含 [1]，哈希表 {1: 0}
print(randomizedSet.remove(2))  # 输出: false，集合中不存在 2
print(randomizedSet.insert(2))  # 输出: true，集合现在包含 [1, 2]，哈希表 {1: 0, 2: 1}
print(randomizedSet.getRandom()) # 输出: 1 或 2，随机返回集合中的一个元素
print(randomizedSet.remove(1))  # 输出: true，集合现在包含 [2]，哈希表 {2: 0}
print(randomizedSet.insert(2))  # 输出: false，2 已在集合中
print(randomizedSet.getRandom()) # 输出: 2，集合中只有 2，总是返回 2