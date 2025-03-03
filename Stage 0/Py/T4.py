class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        comp_s = sorted(s)
        comp_t = sorted(t)
        return comp_s == comp_t