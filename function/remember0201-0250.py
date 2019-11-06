# 207 Course Schedule, 检查是否有环路，也就是后向边
def canFinish(numCourses, prerequisites):
    visited, graph = [-1 for _ in range(numCourses)], [[] for _ in range(numCourses)]

    for node1, node2 in prerequisites:
        graph[node1].append(node2)

    def dfs(i):
        if visited[i] == 0:# 有环路
            return False
        elif visited[i] == 1:
            return True

        visited[i] = 0 # 正在访问

        for node in graph[i]:
            if not dfs(node):
                return False
        visited[i] = 1
        return True

    for i in len(numCourses):
        if not dfs(i):
            return False
    return True

# 208 Implement Trie (Prefix Tree)
# Trie 树的节点由一个大小为26的子节点数组，一个标志符标记到当前位置为止是否为一个词
class TrieNode(object):
    def __init__(self):
        self.isWord = False
        self.child = [None] * 26


class Trie(object):
    def __init__(self):
        self.root = TrieNode() # 根节点不包含字符

    def insert(self, word):
        p = self.root

        for c in word:
            i = ord(c) - ord("a")
            if p.child[i] is None:
                p.child[i] = TrieNode()
            p = p.child[i]
        p.isWord = True

    def search(self, word):
        p = self.root

        for c in word:
            i = ord(c) - ord("a")

            if p.child[i] is None:
                return False
            p = p.child[i]
        return p.isWord

    def startWith(self, prefix):
        p = self.root

        for c in prefix:
            i = ord(c) - ord("a")

            if p.child[i] is None:
                return False
            p = p.child[i]
        return True

class Trie(object): # 用字典来表示的也特别好
    def __init__(self):
        self.root = {}

    def insert(self, word):
        p = self.root

        for c in word:

            if c not in p:
                p[c] = {}
            p = p[c]
        p["END"] = True

    def search(self, word):
        p = self.root

        for c in word:
            if c not in p:
                return False
            p = p[c]
        return "END" in p

    def startWith(self, prefix):
        p = self.root

        for c in prefix:
            if c not in p:
                return False
            p = p[c]
        return True
        

# 211 Add and Search Word - Data structure design
class WordDictionary(object):
    def __init__(self):
        self.root = {}

    def addWord(self, word):
        p = self.word

        for c in word:
            if c not in p:
                p[c] = {}
            p = p[c]
        p["END"] = True

    def search(self, word): # 按层递归

        return help(word, 0, self.root)

def help(word, i, p):
    if i == len(word):
        return "END" in p

    elif word[i] != ".":
        if word[i] not in p:
            return False
        return help(word, i + 1, p[word[i]])

    else:
        for char in p: # 注意 char 的值不能为 "END"
            if char !="END" and help(word, i+1, p[char]):
                return True
        return False

# 214 Shortest Palindrome, 翻转过来然后开始从前比到后
def shortestPalindrome(s):
    t = s[::-1]

    if t == "":
        return ""

    for i in range(len(s), 0, -1):
        if s[:i] == t[len(t) - i:]:
            break

    return t[:len(t) - i] + s

# 215 Kth Largest Element in an Array
def findKthLargest(nums, k):
    lo, hi = 0, len(nums)-1

    while True:
        index = partition(nums, lo, hi)

        if index == k-1:
            return nums[index]
        elif index > (k-1):
            hi = index - 1
        else:
            lo = index + 1

def partition(nums, lo, hi):
    pivot, i = nums[hi], lo-1

    for j in range(lo, hi):
        if nums[j] >= pivot:
            i = i+1
            nums[i], nums[j] = nums[j], nums[i]

    nums[i+1], nums[hi] = nums[hi], nums[i+1]
    return i+1

# 216 Combination Sum III
def combinationSum3(k, n):
    res, out = [], []

    def dfs(level, m):
        if m < 0:
            return
        if m == 0 and len(out) == k:
            res.append(out[:])
            return

        for i in range(level, 10):
            out.append(i)
            dfs(i+1, m - i)
            out.pop()

    dfs(1, n)
    return res

# 222 Count Complete Tree Nodes, 最左边和最右边是否是长度一样，一样就是完美二叉树
# 不一样就接着递归操作
def countNodes(root):
    import math
    hleft, hright = 0, 0
    pl, pr = root, root

    while pl:
        hleft += 1
        pl = pl.left
    while pr:
        hright += 1
        pr = pr.right

    if hleft == hright:
        return math.pow(2, hleft) - 1

    return countNodes(root.left) + countNodes(root.right) + 1

# 223 Rectangle Area, 先判断哪些不重叠， 而后计算重叠面积
def computeArea(A, B, C, D, E, F, G, H):
    area1, area2 = (C - A) * (D - B), (G - E) * (H - F)

    if A >= G or E >= C or B >= H or D <= F:
        return area1 + area2

    return area1 + area2 - (min(G, C) - max(A, E)) * (min(D, H) - max(B, F))


# 224 Basic Calculator, 中缀表达式求值，套路就是先转后缀表达式，然后求值
def calculate(s):
    vals = in2post(s)
    return post(vals)

def in2post(s): # 中缀转后缀
    num, op, st = 0, [], []

    for i in range(len(s)):

        if s[i].isdigit():
            while i < len(s) and s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1

            st.append(num)
            num = 0

        elif s[i] in "+-()":
            if s[i] == "(":
                op.append(s[i])

            elif s[i] == ")":
                while op and op[-1] != "(":
                    st.append(op.pop())
                op.pop()

            else: # 遇到+-, 同等优先级的也一直pop,最后加入当前符号
                while st and st[-1] != "(":
                    st.append(op.pop())
                op.append(s[i])

    while op:
        st.append(op.pop())
    return st

def post(vals): # 计算后缀
    if len(vals) == 1:
        return vals[0]

    st = []

    for val in vals:
        if type(val) != unicode:
            st.append(val)

        else:
            a = st.pop()
            b = st.pop()

            if val == "+":
                st.append(b+a)

            if val == "-":
                st.append(b-a)

    return st.pop()

# 229 Majority Element II, 两个候选人
def majorityElement(nums):
    cand1 = cand2 = cnt1 = cnt2 = 0
    res = []
    for num in nums:
        if num == cand1:
            cnt1 += 1
        elif num == cand2:
            cnt2 += 1
        elif cnt1 == 0:
            cand1 = num
            cnt1 = 1
        elif cnt2 == 0:
            cand2 = num
            cnt2 = 1
        else:
            cnt1 -= 1
            cnt2 -= 1
    cnt1 = cnt2 = 0
    for num in nums:
        if num == cand1:
            cnt1 += 1
        elif num == cand2:
            cnt2 += 1

    if cnt1 > len(nums) / 3:
        res.append(cand1)
    if cnt2 > len(nums) / 3:
        res.append(cand2)
    return res

# 232 Implement Queue using Stacks, 一个栈入队用一个栈出队用
class MyQueue(object):
    def __init__(self):
        self.instack = []
        self.outstack = []

    def push(self, x):
        self.instack.append(x)

    def pop(self, x):
        self.peek()
        return self.outstack.pop()

    def peek(self):
        if not self.outstack:
            while self.instack:
                self.outstack.append(self.instack.pop())
        return self.outstack[-1]

    def empty(self):
        return not self.instack and not self.outstack


# 236 Lowest Common Ancestor of a Binary Tree, 后续遍历，逻辑要能理清
def lowestCommonAncestor(root, p, q):
    if not root or p == root or q == root:
        return root

    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root

    return left if left else right

# 237 Delete Node in a Linked List
def deleteNode(node):
    cur, pre = node, node
    while cur.next:
        cur.val = cur.next.val
        pre = cur
        cur = cur.next
    pre.next = None

# 238 Product of Array Except Self, 计算该元素前面的乘积和后面的乘积
def productExceptSelf(nums):
    res, right = [1 for _ in range(len(nums))], 1

    for i in range(1, len(nums)):
        res[i] = res[i-1] * nums[i-1] # 计算该元素之前的乘积

    for i in range(len(nums) - 1, -1, -1): # 乘以之后的
        res[i] *= right
        right *= nums[i]

    return res


# 239 Sliding Window Maximum # 双向队列
def maxSlidingWindow(nums, k):
    from collections import deque

    res, q = [], deque()

    for i in range(len(nums)):
        while q and nums[i] > nums[q[-1]]:
            q.pop()

        q.append(i)

        if i >= k and q and q[0] == i - k:
            q.popleft()

        if i >= k - 1:
            res.append(nums[q[0]])

    return res