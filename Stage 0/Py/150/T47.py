class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []  # 初始化一个空栈，用于存储简化路径的目录名
        parts = path.split('/')  # 使用 '/' 分割输入路径字符串 path，得到一个目录/文件名列表 parts

        for part in parts:  # 遍历分割后的列表 parts
            if part == '' or part == '.':  # 如果 part 是空字符串 "" (由连续斜杠产生) 或 "." (当前目录)
                continue  # 则忽略，继续处理下一个 part
            elif part == '..':  # 如果 part 是 ".." (上一级目录)
                if stack:  # 检查栈是否为空。只有栈不为空时，才能执行 pop 操作
                    stack.pop()  # 弹出栈顶元素，模拟返回上一级目录
                # 如果栈为空，说明已经在根目录，".." 操作无效，忽略
            else:  # 如果 part 是其他字符串 (有效的目录名，例如 "home", "user", "..." 等)
                stack.append(part)  # 将该目录名压入栈中

        simplified_path = "/" + "/".join(stack)  # 将栈中的所有目录名用 "/" 连接起来，并在最前面加上 "/"，构成简化后的路径字符串
        return simplified_path if simplified_path != "//" else "/" # 考虑栈为空的情况。如果栈为空，"/".join(stack) 会得到空字符串 ""，加上前导 "/" 变成 "/"，但是如果栈不为空但是所有元素都是空字符串的时候，可能会出现 "//" 的情况，这里做了一个特殊处理，如果结果是 "//" 就返回 "/"，否则直接返回 simplified_path

# 示例 1
path1 = "/home/"
result1 = Solution().simplifyPath(path1)
print(f"输入: {path1}, 输出: {result1},  判断: {result1 == '/home'}") # 输出: 输入: /home/, 输出: /home,  判断: True

# 示例 2
path2 = "/home//foo/"
result2 = Solution().simplifyPath(path2)
print(f"输入: {path2}, 输出: {result2},  判断: {result2 == '/home/foo'}") # 输出: 输入: /home//foo/, 输出: /home/foo,  判断: True

# 示例 3
path3 = "/home/user/Documents/../Pictures"
result3 = Solution().simplifyPath(path3)
print(f"输入: {path3}, 输出: {result3},  判断: {result3 == '/home/user/Pictures'}") # 输出: 输入: /home/user/Documents/../Pictures, 输出: /home/user/Pictures,  判断: True

# 示例 4
path4 = "/../"
result4 = Solution().simplifyPath(path4)
print(f"输入: {path4}, 输出: {result4},  判断: {result4 == '/'}") # 输出: 输入: /../, 输出: /,  判断: True

# 示例 5
path5 = "/.../a/../b/c/../d/./"
result5 = Solution().simplifyPath(path5)
print(f"输入: {path5}, 输出: {result5},  判断: {result5 == '/.../b/d'}") # 输出: 输入: /.../a/../b/c/../d/./, 输出: /.../b/d,  判断: True

# 示例 6 (空路径，虽然题目说明是有效路径，但可以测试一下边界情况)
path6 = "/"
result6 = Solution().simplifyPath(path6)
print(f"输入: {path6}, 输出: {result6},  判断: {result6 == '/'}") # 输出: 输入: /, 输出: /,  判断: True
