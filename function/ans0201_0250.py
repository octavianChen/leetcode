# 203 Remove Linked List Element
def removeElements(head, val):
    dumpy = ListNode(-1)
    cur = dumpy

    while head:
        if head.val != val:
            cur.next = head
            cur = cur.next
        else:
            cur.next = None
        head = head.next

    return dumpy.next

# 205 Isomorphic Strings, 一对一映射
def isIsomorphic(s, t):
    m1, m2 = {}, {}

    for c1, c2 in zip(s, t):
        val1, val2 = m1.get(c1, None), m2.get(c2, None)

        if val1 is None and val2 is None:
            m1[c1], m2[c2] = c2, c1

        elif val1 != c2 or val2 != c1:
            return False
    return True

# 206 Reverse Linked List
def reverseList(head):

    dumpy, cur1 = ListNode(-1), dumpy
    
    while head:
        t, head = head, head.next
        t.next = cur1.next
        cur1.next = t

    return dumpy.next

# 207 Course Schedule
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

# 209 Minimum Size Subarray Sum
def minSubArrayLen(s, nums):
    val, left, i, res = 0, 0, 0, float("inf")

    while i < len(nums):
        val += nums[i]

        while val >= s:
            res = min(res, i - left + 1)
            val, left = val - nums[left], left + 1
        
        if val < s:
            i += 1

    return res if res < float("inf") else 0


# 210 Course Schedule II
class Solution(object):
    def findOrder(numCourses, prerequisites):
        graph, finish = [[] for _ in range(numCourses)], [0 for _ in range(numCourses)]
        color, self.time = [-1 for _ in range(numCourses)], 0
        for edge in prerequisites:# 图的构建
            graph[edge[1]].append(edge[0])


        def dfs(i):
            if color[i] == 0:
                return False

            if color[i] == 1:
                return True

            self.time, color[i] = self.time + 1, 0

            for j in graph[i]:
                if not dfs(j):
                    return False

            self.time, color[i] = self.time + 1, 1
            finish[i] = self.time
            return True

        for i in range(numCourses):
            if not dfs(i):
                return []

        return sorted(range(len(finish)), key=lambda k:-finish[k])

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

# 213 House Robber II
def rob(nums):
    if len(nums) == 1:
            return nums[0]
            
    p12 = p11 = p22 = p21 = ans1 = ans2 = 0

    for i in range(len(nums)):
        if 0 <= i < len(nums) - 1:
            ans1 = max(nums[i] + p12, p11)
            p12, p11 = p11, ans1
        if 0 < i <= len(nums) -1:
            ans2 = max(nums[i] + p22, p21)
            p22, p21 = p21, ans2

    return max(ans1, ans2)

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

# 217 Contains Duplicate
def containsDuplicate(nums):
    m = {}
    for num in nums:
        if num in m:
            return True
    return False

# 219 Contains Duplicate II
def containsNearbyDuplicate(nums, k):
    m, res = {}, True

    for i in range(len(nums)):
        if nums[i] in m:
            if abs(i - m[nums[i]]) <= k:
                return True
        m[nums[i]] = i
    return False

# 221 Maximal Square
def maximalSquare(matrix):
    if not matrix or not matrix[0]:
        return 0
    res, m, n = 0, len(matrix), len(matrix[0])

    height = [0 for _ in range(n)]

    for i in range(m):
        for j in range(n):
            height[j] = 0 if matrix[i][j] == "0" else 1 + height[j]
        res = max(res, help(height[:]))
    return res

def help(height):
    res, st = 0, []
    height.append(0)

    for i in range(len(height)):
        while st and height[st[-1]] > height[i]:
            cur = st.pop()

            wide = i - 1 - st[-1] if st else i

            a = min(height[cur], wide)

            res = max(res, a * a)
        st.append(i)
    return res 


# 222 Count Complete Tree Nodes
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
                while st and st[-1] !+= "(":
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

# 225 Implement Stack using Queues
class MyStack(object):
    def __init__(self):
        self.q1 = Queue.Queue()
        self.q2 = Queue.Queue()

    def push(self, x):
        if not self.q1.empty():
            self.q1.put(x)
        else:
            self.q2.put(x)

    def pop(self):
        full = self.q1 if self.q2.empty() else self.q2
        empty = self.q2 if self.q2.empty() else self.q1

        while full.qsize() != 1:
            empty.put(full.get())
        return full.get()

    def top(self):
        full = self.q1 if self.q2.empty() else self.q2
        empty = self.q2 if self.q2.empty() else self.q1

        while full.qsize() != 1:
            empty.put(full.get())
        res = full.get()
        empty.put(res)
        return res

    def empty(self):
        return self.q1.empty() and self.q2.empty()

# 226 Invert Binary Tree
def invertTree(root):
    if not root:
        return root

    left, right = invertTree(root.left), invertTree(root.right)
    root.left, root.right = right, left

    return root


# 227 Basic Calculator II
def calculate(s):
    vals = in2post(s)

    return post(vals)

def in2post(s):
    op, st, num, i, m = [], [], 0, 0, {"+":1, "-":1, "*":2,"/":2}

    while i < len(s):
        if s[i].isdigit():
            while i < len(s) and s[i].isdigit():
                num = num * 10 + int(s[i])
                i+= 1
            st.append(num)
            num = 0

        elif s[i] in "+-*/":
            while op and m[s[i]] <= m[op[-1]]:
                st.append(op.pop())

            op.append(s[i])
            i += 1

        else:
            i += 1

    while op:
        st.append(op.pop())

    return st

def post(vals):
    if len(vals) == 0:
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

            if val == "*":
                st.append(b*a)

            if val == "/":
                st.append(b//a)
    return st.pop()


# 228 Summary Ranges
def summaryRanges(nums):
    res, i, j = [], 0, 0
    nums.append(float("inf"))

    while j < len(nums):
        if j == 0 or nums[j] - nums[j-1] == 1:
            j += 1

        else:
            if j - i == 1:
                res.append(str(nums[i]))

            else:
                res.append(str(nums[i]) + "->" + str(nums[j-1]))

            i, j = j, j + 1

    return res

# 229 Majority Element II
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


# 230 Kth Smallest Element in a BST
def kthSmallest(root, k):
    cnt, st, p = 1, [], root
    
    while st or p:
        while p:
            st.append(p)
            p = p.left

        p = st.pop()
        if cnt == k:
            return p.val
        p, cnt = p.right, cnt + 1

# 231 Power of Two, 只需要判断二进制中是否只有一个1
def isPowerOfTwo(n):
    return n&(n-1) if n>0 else False

# 232 Implement Queue using Stacks
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

# 234 Palindrome Linkded list
def isPalindrome(head):
    if not head or not head.next:
        return True
    slow, fast, dumpy = head, head, ListNode(0)
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    cur = head

    while cur != slow:
        tmp = dumpy.next
        dumpy.next = cur
        cur = cur.next
        dumpy.next.next = tmp

    if fast:
        slow = slow.next

    cur = dumpy.next

    while slow:
        if cur.val != slow.val:
            return False
        cur, slow = cur.next, slow.next
    return True

# 235 Lowest Common Ancestor of a Binary Search Tree
def lowestCommonAncestor(root, p, q):
    if not root or p == root or q == root:
        return root

    if p.val < root.val < q.val or q.val < root.val < p.val:
        return root

    if p.val > root.val and p.val > root.val:
        return lowestCommonAncestor(root.right, p, q)
    return lowestCommonAncestor(root.left, p, q)

# 236 Lowest Common Ancestor of a Binary Tree
def lowestCommonAncestor(root, p, q):
    if not root or p == root or q == root:
        return root

    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    if left and right:
        return head

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



# 240 Search a 2D Matrix II
def searchMatrix(matrix, target):
    if not matrix or not matrix[0]:
        return False

    m, n = 0, len(matrix[0]) - 1

    while m < len(matrix) and n >= 0:
        if matrix[m][n] == target:
            return True

        elif matrix[m][n] > target:
            n -= 1

        else:
            m += 1
    return False


# 242 Valid Anagram
def isAnagram(s, t):
    if len(s) != len(t):
        return False
    for c in set(s):
        if s.count(c) != t.count(c):
            return False
    return True

