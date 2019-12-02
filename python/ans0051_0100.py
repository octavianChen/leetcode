# !/usr/bin/python
# -*- coding=UTF-8 -*-


from functools import reduce


class ListNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None


# 54 Spiral Matrix
def spiralMatrix(matrix):
    m = len(matrix)
    res = []
    if m == 0;
        return res
    n = len(matrix[0])

    tx, ty, bx, by = 0, 0, m-1, n-1
    while tx <= bx and ty <= by:
        if tx == bx:
            for i in range(ty, by+1):
                res.append(matrix[tx][i])
        elif ty == by:
            for i in range(tx, bx+1):
                res.append(matrix[i][ty])
        else:
            for i in range(ty, by):
                    res.append(matrix[tx][i])
            for i in range(tx, bx):
                res.append(matrix[i][by])
            for i in range(by, ty, -1):
                res.append(matrix[bx][i])
            for i in range(bx, tx, -1):
                res.append(matrix[i][ty])
        tx, ty, bx, by = tx + 1, ty + 1, bx - 1, by -1
    return res


# 55 Jump Game, dp[i]表示到i位置的剩余跳力
def canJump(nums):
    dp = [0 for _ in range(len(nums))]

    for i in range(1, len(nums)):
        dp[i] = max(dp[i-1], nums[i-1]) - 1

        if dp[i] < 0:
            return False
    return True

# 贪心算法，这里只关心是否能达到末尾，也就是说我们只对到达的最远位置感兴趣，维护一个变量
# reach，表示最远到达的位置
def canJump(nums):
    n, reach = len(nums), 0

    for i in range(n):
        if i > reach or reach >= n - 1:
            break
        reach = max(reach, i + nums[i]) # i + nums[i] 表示当前能到的最远位置

    return reach >= n - 1 

    

# 56 Merge Intervals
def merge(intervals):
    intervals, res = sorted(intervals, key=lambda x:x[0]), []

    for num in intervals:
        if not res or num[0] > res[-1][1]:
            res.append(num)

        else:
            res[-1][1] = max(res[-1][1], num[1])

    return res
    
# 58 Length of Last Word
def lengthOfLastWord(s):
    res = s.strip().split(" ")
    return len(res[-1])


# 59 Spiral Matrix II
def generateMatrix(n):
    res = [[0 for _ in range(n)] for _ in range(n)]
    tx, ty, bx, by = 0, 0, n-1, n-1
    num = 1
    while tx <= bx:
        if tx == bx:
            res[tx][bx] = num
            return res
        for i in range(ty, by):
            res[tx][i] = num
            num += 1
        for i in range(tx, bx):
            res[i][by] = num
            num += 1
        for i in range(by, ty, -1):
            res[bx][i] = num
            num += 1
        for i in range(bx, tx, -1):
            res[i][ty] = num
            num += 1
        
        tx, ty, bx, by = tx + 1, ty + 1, bx - 1, by - 1
    return res

# 61 Rotate List
def rotateList(head, k):
    if head is None or head.next is None:
        return head

    count, m, dumpy, cur = 1, {}, ListNode(0), head

    while cur:
        m[count], cur, count = cur, cur.next, count + 1

    k = k % (count - 1)

    if not k:
        return head

    dumpy.next, m[count-1].next, m[count-k-1].next = m[count-k], head, None
    return dumpy.next

# 62 Unique Paths
def uniquePaths(m, n):
    dp = [[1] * (n+1)]*(m+1)

    for i in range(2, m+1):
        for j in range(2, n+1):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m][n]


# 63 Unique Path II
def uniquePathWithObstacle(obstacleGrid):
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0 for _ in range(n)] for _ in range(m)]
    dp[0][0] = 0 if obstacleGrid[0][0] else 1
    for i in range(1, m):
        if obstacleGrid[i][0] == 1 or dp[i-1][0] == 0:
            dp[i][0] = 0
        else:
            dp[i][0] = 1

    for i in range(1, n):
        if obstacleGrid[0][i] == 1 or dp[0][i-1] == 0:
            dp[0][i] = 0
        else:
            dp[0][i] = 1

    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] == 1:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

# 64 Minimum Path Sum
def minPathSum(grid):
    res, m, n = 0, len(grid), len(grid[0])

    dp = [[grid[0][0] for _ in range(n)] for _ in range(m)]

    for i in range(1, m):
        dp[i][0] = grid[i][0] + dp[i-1][0]

    for i in range(1, n):
        dp[0][i] = grid[0][i] + dp[0][i-1]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]


# 66 Plus One
def plusOne(digits):
    ans, res = 0, []
    for digit in digits:
        ans = ans * 10 + digit
    ans += 1
    while ans:
        res.append(ans % 10)
        ans //= 10
    return res[::-1]


# 67 Add Binary
def addBinary(a, b):
    return help(a[::-1], b[::-1])

def help(a, b):
    res, mode = "", 0

    for i in range(max(len(a), len(b))):
        a1 = int(a[i]) if i < len(a) else 0
        b1 = int(b[i]) if i < len(b) else 0

        res += str((a1 + b1 + mode) % 2)
        mode = (a1 + b1 + mode) // 2

    if mode:
        res += "1"

    return res[::-1]

# 69 Sqrt(x)
def mySqrt(x):
    lo, hi = 0, x
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * mid <= x:
            lo = mid
        else:
            hi = mid -1
    return lo


# 70 Climbing Stairs
def climbStairs(n):
    old, new = 1, 2
    for i in range(n):
        new, old = new + old, new
    return new if n>=2 else 1


# 71 Simplify Path
def simplifyPath(path):
    st = []

    for dirs in path.split("/"):

        if not dirs or dirs == ".":
            continue

        elif dirs == "..":
            if st:
                st.pop()

        else:
            st.append(dirs)
    return "/" + "/".join(st)


# 72 Edit Distance, 注意初始化条件和状态转移方程
def minDistance(word1, word2):
    m, n = len(word1), len(word2)

    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i

    for i in range(n+1):
        dp[0][i] = i

    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
    return dp[m][n]


# 73 Set Matrix Zeros, 利用第一行第一列来记录该行和该列是否有0,
def setZeros(matrix):
    if not matrix or not matrix[0]:
        return matrix
    m, n, rowzero, colzero = len(matrix), len(matrix[0]), False, False

    for i in range(m):
        if matrix[i][0] == 0:
            colzero = True
    for i in range(n):
        if matrix[0][i] == 0:
            rowzero = True

    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0], matrix[0][j] = 0, 0

    for i in range(1, m):
        for j in range(1, n):
            if matrix[0][j] == 0 or matrix[i][0] == 0:
                matrix[i][j] = 0

    if rowzero:
        for i in range(n):
            matrix[0][i] = 0
    if colzero:
        for i in range(m):
            matrix[m][0] = 0


# 74 Search a 2D matrix
def searchMatrix(matrix, target):
    if not matrix or not matrix[0]:
        return False

    row, col = 0, len(matrix[0]) - 1
    while row <= len(matrix) - 1 and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    return False


# 75 Sort Colors
def sortColors(nums):
    blue, red, i = -1, len(nums), 0

    while i < red:
        if nums[i] == 0:
            blue += 1
            nums[blue], nums[i] = nums[i], nums[blue]
            i += 1

        elif nums[i] == 2:
            red -= 1
            nums[red], nums[i] = nums[i], nums[red]

        else:
            i += 1


# 76 Minimum Window Substring, 滑窗法，注意字典的处理，以及cnt的加减为什么要这样
def minWidow(s, t):
    res, m, left, cnt, minLen = "", {}, 0, 0, float("inf")

    for c in t:
        m[c] = m.get(c, 0) + 1

    for i in range(len(s)):
        if s[i] in m:
            m[s[i]] -= 1
            cnt = cnt + 1 if m[s[i]] >= 0 else cnt # 必须要大于等于0才增加1

        while cnt == len(t):
            if minLen > i - left + 1:
                minLen, res = i - left + 1, s[left:i+1]

            if s[left] in m:
                m[s[left]] += 1
                cnt = cnt - 1 if m[s[left]] > 0 else cnt # 必须大于 0 才减少 1

            left += 1

    return res


# 77 Combinations
def combine(n, k):
    out, res = [], []
    dfs(n, 1, k, out, res)
    return res

def dfs(n, start, k, out, res):
    if len(out) == k:
        res.append(out[:])
        return

    for i in range(start, n+1):
        out.append(i)
        dfs(n. i+1, k, out, res)
        out.pop()


# 78 Subsets
def subsets(nums):
    res, ans = [], []
    dfs(nums, 0, ans, res)
    return res

def dfs(nums, pos, ans, res):
    res.append(ans[:])

    for i in range(pos, len(nums)):
        ans.append(nums[i])
        dfs(nums, i+1, ans, res)
        ans.pop()


# 79 Word Search, 一个简单的 dfs
def exist(board, word):
    if not board or not board[0]:
        return False

    m, n, index = len(board), len(board[0]), 0

    visited = [[-1 for _ in range(n)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            if dfs(board, i, j, word, index, visited):
                return True
    return False


def dfs(board, i, j, word, index, visited):
    if index == len(word):
        return True

    if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or visited[i][j] == 0:
        return False

    if board[i][j] != word[index]:
        return False

    visited[i][j] = 0

    direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    for dirs in direction:
        r, c = i + dirs[0], j + dirs[1]

        if 0 <= r < len(board) and 0 <= j < len(board[0]):
            if dfs(board, r, c, word, index + 1, visited):
                return True
    visited[i][j] = -1
    return False


# 80 Remove duplicates from sorted arrayII, 快慢指针法
def removeDuplicates(nums):
    i = 0
    j = 1
    count = 1
    while j < len(nums):
        if nums[i] == nums[j] and count == 0:
            j += 1
        else:
            if nums[i] == nums[j]:
                count -= 1
            else:
                count = 1
            i += 1
            nums[i] = nums[j]
            j += 1
    return i + 1


# 81 Search in Rotated Sorted Array II
def search(nums, target):
    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if nums[mid] == target: # 找到目标
            return True

        elif nums[mid] > nums[hi]: # [lo, mid] 是有序的
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1

            else:
                lo = mid + 1

        elif nums[mid] < nums[hi]:#[mid, hi] 是有序的
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1

        else:
            hi -= 1

    return False

# 82 Remove Duplicates from Sorted List II
def removeDuploicates(head):
    if head is None or head.next is None:
        return head
    dumpy = ListNode(0)
    cur = dumpy

    slow, fast, count = head, head.next, 0

    while fast:
        if slow.val == fast.val:
            count += 1
        else:
            if not count:
                cur.next = slow
                cur = cur.next
            else:
                count = 0
            slow = fast
        fast = fast.next

    cur.next = None if count else slow
    return dumpy.next


# 83 Remove Duplicates from Sorted List
def deleteDuplicates(head):
    slow, fast = head, head

    while fast:
        if slow.val != fast.val:
            slow.next = fast
            slow = slow.next

    if slow:
        slow.next = None
    return head


# 84 Largest Rectangle in Histogram, 局部峰值，然后往回计算矩形最大值，非常秀
def largestRectangleArea(heights):
    res = 0

    for i in range(len(heights)):
        if i + 1 < len(heights)  heights[i] > heights[i+1]:
            minH = heights[i]

            for j in range(i, -1, -1):
                minH = min(minH, heights[j])
                area = minH * (i-j+1)
                res = max(res, area)

    return res

# 接下来的方法是单调栈的总结
def largestRectangleArea(heights):
    # 维护一个递增栈，遇到较小的数字只是一个触发，表示要开始计算矩形面积了
    res, stk = 0, []
    # 为使最后一个元素也得到计算，在heights 后加 0 
    heights.append(0)

    for i in range(len(heights)):
        while st and heights[stk[-1]] >= heights[i]:
            cur = st.pop()
            wide = i - st[-1] - 1 if st else i
            res = max(res, heights[cur] * wide)
        stk.append(i)
    return res


# 85 Maximal Rectangle, 神奇的把二维矩阵转化成求直方图的最大面积
def maximalRectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    res, m, n = 0, len(matrix), len(matrix[0])

    height = [0 for _ in range(n)]

    for i in range(m):
        for j in range(n):
            height[j] = 0 if matrix[i][j] == "0" else 1 + height[j]
        res = max(res, help(height))

    return res

def help(height):
    res, st = 0, []
    height.append(0)

    for i in range(len(height)):
        while st and height[st[-1]] > height[i]:
            cur = st.pop()

            wide = i - 1 - st[-1] else i

            res = max(res, height[cur] * wide)
        
        st.append(i)
    return res
        
# 86 Partition List
def partition(head, x):
    dumpy1, dumpy2 = ListNode(0), ListNode(0)
    cur1, cur2, cur = dumpy1, dumpy2, head

    while cur:
        if cur.val < x:
            cur1.next = cur
            cur1 = cur1.next

        else:
            cur2.next = cur
            cur2 = cur2.next
        cur = cur.next
    cur2.next, cur1.next = None, dumpy2.next
    return dumpy1.next



# 88 Merge Sorted Array
def merge(nums1, m, nums2, n):
    i, j, k = m-1, n-1, m+n-1

    while i>=0 and j>=0:
        if nums1[i] >= nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums1[j]
            j -= 1
        k -= 1
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1

# 90 Subsets II
def subsetsWithDup(nums):
    res, out, nums = [], [], sorted(nums)

    def dfs(index):
        res.append(out)

        for i in range(index, len(nums)):
            if i == index or nums[i] != nums[i-1]:
                out.append(nums[i])
                dfs(i+1)
                out.pop()

    dfs(0)
    return res


# 91 Decode Ways, 这题是个动态规划，初始条件和状态方程要能看出来
def numDecodings(s):
    if not s or s[0] == "0":
        return 0

    dp = [1 for _ in range(len(s) + 1)]

    for i in range(1, len(s) + 1):
        dp[i] = 0 if s[i-1] == "0" else dp[i-1]

        if i > 1 and (s[i-2] == "1" or (s[i-2] == "2" and s[i-1] <= "6")):
            dp[i] += dp[i-2]

    return dp[-1]
    

# 92 Reverse Linked List II
def reverseBetween(head, m, n):
    dumpy = ListNode(-1)
    dumpy.next, pre, cur, cnt = head, dumpy, head, 1

    while cur:

        if cnt < m:
            pre, cnt, cur = cur, cnt + 1, cur.next
        elif cnt == m:
            for _ in range(m, n):
                t = cur.next
                cur.next = t.next
                t.next = pre.next
                pre.next = t
            return dumpy.next


# 94 Binary Tree Inorder Traversal
def inorderTraversal(root):
    m, res, p = [], [], root

    while m or p:
        while p:
            m.append(p)
            p = p.left

        p = m.pop()
        res.append(p.val)
        p = p.right
    return res


# 95 Unique Binary Search Trees II
def generateTree(n):
    if n == 0:
        return []

    def help(lo, hi):
        if lo > hi:
            return [None]
        res = []
        for i in range(lo, hi+1):
            ll, rl = help(lo, i-1), help(i+1, hi)

            for ln in ll:
                for rn in rl:
                    root = TreeNode(i)
                    root.left, root.right = ln, rn
                    res.append(root)
        return res

    return help(1, n)
    

# 96 Unique Binary Search Tree
def numTrees(n):
    dp = [1 for i in range(n+1)]
    for i in range(2, n+1):
        for j in range(0, i):
            dp[i] += dp[j] * dp[i-j-1]
    return dp[n]


# 97 Interleaving String, dp[i][j]表示s1的前i个字符和s2的前j个字符是否匹配s3的前 i+j 个字符
def isInterleave(s1, s2, s3):
    m, n = len(s1), len(s2)

    if m + n != len(s3):
        return False

    dp = [[True for _ in range(n+1)] for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]

    for i in range(1, n+1):
        dp[0][i] = dp[0][i-1] and s2[i-1] == s3[i-1]

    for i in range(m+1):
        for j in range(n+1):
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or (dp[i][j-1] and s2[j-1] == s3[i+j-1])

    return dp[m][n]


# 98 Validate Binary Search Tree
def isValidBST(root):
    pval = -float("inf")

    def help(node):
        if not node:
            return True

        st, p = [], root

        while st or p:
            while p:
                st.append(p)
                p = p.left

            p = st.pop()

            if p.val <= pval:
                return False

            pval = p.val
            p = p.right
        return True
    return help(root)

# 99 Recover Binary Search Tree
def recoverTree(root):
    if not root:
        return 

    st, err, p, pre = [], [None, None], root, None

    while st or p:
        while p:
            st.append(p)
            p = p.left

        p = st.pop()

        if pre:
            if p.val < pre.val and not err[0]:
                err[0], err[1] = pre, p

            elif p.val < pre.val and p.val < err[1].val:
                err[1] = p


        pre, p = p, p.right

    err[0].val, err[1].val = err[1].val, err[0].val


# 100 Same Tree
def isSameTree(p, q):
    if not p or not q:
        return True
    elif not p or not q or p.val != q.val:
        return False
    else:
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)