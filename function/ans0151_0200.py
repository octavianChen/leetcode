class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 151 Reverse Words in a String
def reverseWords(s):
    return " ".join(word for word in s.split()[::-1])


# 152 Maximum Product Subarray
def maxProduct(nums):
    res, lp, sp = -float("inf"), 1, 1

    for num in nums:
        if num > 0:
            lp, sp = max(lp * num, num), min(sp * num, num)
        else:
            lp, sp = max(sp * num, num), min(lp * num, num) 

        res = max(res, lp)
    return res

# 153 Find Minimum in Rotated Sorted Array
def findMin(nums):
    lo, hi = 0, len(nums) - 1

    if nums[lo] > nums[hi]:
        while lo < hi:
            mid = (lo + hi) // 2
            if hi - lo == 1:
                return nums[mid + 1]
            elif nums[mid] > nums[lo]:
                lo = mid
            else:
                hi = mid
    return nums[lo]


# 154 Find Minimum in Rotated Sorted Array II
def findMin(nums):
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        if lo == hi - 1:
            break

        if nums[lo] < nums[hi]:
            return nums[lo]

        mid = (lo + hi) // 2
        if nums[lo] > nums[mid]:
            hi = mid
        elif nums[mid] > nums[hi]:
            lo = mid

        else:
            while lo < mid:
                if nums[lo] == nums[mid]:
                    lo += 1
                elif nums[lo] < nums[mid]:
                    return nums[lo]
                else:
                    hi = mid
                    break


# 155 Min Stack
class MinStack(object):
    def __init__(self):
        self.container = []
        self.min = []

    def push(self, x):
        self.container.append(x)
        self.min.append(min(x, self.min[-1])) if self.min else self.min.append(x)

    def pop(self):
        self.container.pop()
        self.min.pop()

    def top(self):
        return self.container[-1]

    def getMin(self):
        return self.min[-1] 


# 160 Intersection of Two Linkde Lists
def getIntersectionNode(headA, headB):
    curA, curB, cntA, cntB = headA, headB, 0, 0
    while curA or curB:
        if curA:
            cntA += 1
            curA = curA.next

        if curB:
            cntB += 1
            curB = curB.next

    if cntA > cntB:
        for _ in range(cntA - cntB):
            headA = headA.next
    else:
        for _ in range(cntB - cntA):
            headB = headB.next

    while headA:
        if headA == headB:
            return headA
        headA, headB = headA.next, headB.next
    return None


# 162 Find Peak Element
def findPeakElement(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] < nums[mid + 1]:
            lef = mid + 1
        else:
            right = mid
    return right


# 169 Majority Element
def majorityElement(nums):
    candi, times = 0, 0

    for num in nums:
        if times == 0:
            candi, times = num, times+1
        elif num != candi:
            times -= 1
        else:
            times += 1
    return candi


# 173 Binary Search Tree Iterator
class BSTIterator(object):
    def __init__(self, root):
        self.c = []

        def help(root):
            if root is None:
                return

            help(root.right)
            self.c.append(root.val)
            help(root.left)

        help(root)

    def next(self):
        return self.c.pop()

    def hasNext(self):
        return len(self.c) != 0

# 189 Rotate Array
def rotate(nums, k):
    k = k % len(nums)

    def reverse(lo, hi):
        while lo < hi:
            nums[lo], nums[hi] = nums[hi], nums[lo]
            lo, hi = lo + 1, hi - 1

    reverse(0, len(nums) - k - 1)
    reverse(len(nums) - k, len(nums) - 1)
    reverse(0, len(nums) - 1)    


# 191 Number of 1 Bits
def hammingWeight(n):
    res = 0
    while n:
        res += 1
        n = (n-1) & n
    return res


# 198 House Robber
def rob(self, nums):
    pre2, pre1, ans = 0, 0, 0

    for i in range(len(nums)):
        ans = max(nums[i] + pre2, pre1)
        pre2, pre1 = pre1, ans

    return ans


# 199 Binary Tree Right Side View
import Queue
def rightSideView(root):
    res, q, lev = [], Queue.Queue(), []

    if root:
        q.put(root)

    while not q.empty():
        for _ in range(q.qsize()):
            node = q.get()
            lev.append(node.val)

            if node.left:
                q.put(node.left)
            if node.right:
                q.put(node.right)
        res.append(lev.pop())
    return res

# 200 Number of Islands
def numIslands(grid):
    if not grid or not grid[0]:
        return 0

    def dfs(i, j):
        direction = [[-1, 0],[1, 0],[0, -1],[0, 1]]
        grid[i][j] = "0"

        for dirs in direction:
            r, c = i + dirs[0], j + dirs[1]

            if 0 <= r < m and 0 <= c <n and grid[r][c] == "1":
                dfs(r, c)

    res, row, col = 0, len(grid), len(grid[0])

    for i in range(row):
        for j in range(col):
            if grid[i][j] == "1":
                dfs(i, j)
                res += 1
    return res
