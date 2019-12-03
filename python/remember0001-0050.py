class ListNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None


# 3 Longest substring without repeating characters 滑窗法
def lengthOfLongestSubstring(s):
    n = len(s)
    left = -1
    m = {}
    res = 0

    for i in range(n):
        if s[i] in m and left < m[s[i]]:
            left = m[s[i]]

        m[s[i]] = i
        res = max(res, i - left)
    return res


# 4 Median of Sorted Arrays
def findMedianSortedArrays(nums1, nums2):
    m = len(nums1)
    n = len(nums2)
    left = (m+n+1) // 2
    right = (m+n+1) // 2

    return(findMedianSortedArrays1(nums1, 0, nums2, 0, left) + 
        findMedianSortedArrays1(nums1, 0, nums2, 0, right))/2.0

def findMedianSortedArrays1(nums1, i, nums2, j, k):
    if i >= len(nums1):
        return nums2[j + k - 1]
    if j >= len(nums2):
        return nums1[i + k - 1]
    if k == 1:
        return min(nums[i], nums[j])
    midv1 = nums1[i + k//2 - 1] if i + k //2 -1 < len(nums1) else float("inf")
    midv2 = nums2[j + k//2 - 1] if j + k //2 -1 < len(nums2) else float("inf")

    if midv1 < midv2:
        return findMedianSortedArrays1(nums1, i + k//2, nums2, j, k-k//2)
    else:
        return findMedianSortedArrays1(nums1, i, nums2, j + k//2, k-k//2)


# python 的另一种解法
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        
        
        def getKthSmallestNumbFromTwoLists(list_1, start_1, end_1, list_2, start_2, end_2, k):
            length_1 = end_1 - start_1 + 1
            length_2 = end_2 - start_2 + 1

            # 在我们的helper function set up中 希望list1 是那个短的 且在process的过程中 也是list_1 短
            # if length_1 > length_2:
            #     return getKthSmallestNumbFromTwoLists(list_2, start_2, end_2, list_1, start_1, end_1, k)

            # Three base case 
            if length_1 == 0 and length_2 > 0:
                return list_2[start_2 + k -1]
            if length_1 > 0 and length_2 == 0:
                return list_1[start_1 + k -1]
            
            if k == 1: #在文档描述中也有，若k== 1， 则只需看两个list中第一个数字哪个小即可
                return min(list_1[start_1], list_2[start_2])


            #此处i, j 为何等于这个 画图即知
            i = start_1 + min(length_1, k//2) -1
            j = start_2 + min(length_2, k//2) -1


            if list_1[i] > list_2[j]:
                return getKthSmallestNumbFromTwoLists(list_1, start_1, end_1, list_2, j+1, end_2, k - (j-start_2+1)) #同上，这里以及下面为啥是k - (j-start_2+1) 画图即知
            else:
                return getKthSmallestNumbFromTwoLists(list_1, i+1, end_1, list_2, start_2, end_2, k - (i-start_1+1))
            
            
    
        m = len(nums1)
        n = len(nums2)
        if (m+n)%2 == 0:
            return (getKthSmallestNumbFromTwoLists(nums1, 0, m-1, nums2, 0, n-1, (m+n)/2) + getKthSmallestNumbFromTwoLists(nums1, 0, m-1, nums2, 0, n-1, (m+n)/2+1))*0.5
        else:
            return getKthSmallestNumbFromTwoLists(nums1, 0, m-1, nums2, 0, n-1, (m+n)/2+1)
        


# 10 Regular Expression Matching
def isMatch(s, p):
    # p 为空
    if not p:   
        return True if not s else False

    # p 的长度为 1且第一个字符匹配
    if len(p) == 1: 
        return (len(s) == 1) and (s[0] == p[0] or p[0] == ".")

    if p[1] != "*":
        if not s:
            return False
        else:
            return (s[0] == p[0] or p[0] == ".") and isMatch(s[1:], p[1:])

    # * 的时候处理很复杂
    while s and (s[0] == p[0] or p[0] == "."):
        if isMatch(s, p[2:]):
            return True
        s = s[1:]

    return isMatch(s, p[2:])

def isMatch(s, p):
    dp = [[False for _ in range(len(p) + 1)] for _ in range(len(s)+1)]
    dp[0][0] = True

    for i in range(2, len(p) + 1):
        if p[i-1] == "*":
            dp[0][i] = dp[0][i-2]

    for i in range(1, len(s)+1):
        for j in range(1, len(p) + 1):
            if s[i-1] == p[j-1] or p[j-1] == ".":
                dp[i][j] = dp[i-1][j-1]

            elif p[j-1] == "*": # dp[i][j-1]匹配一次dp[i][j-2] 匹配零次, 
                dp[i][j] = dp[i][j-1] or dp[i][j-2] or (dp[i-1][j] and (s[i-1] == p[j-2] or p[j-2]=="."))

    return dp[len(s)][len(p)]


# 11 Container with the most water,双指针滑窗法
def maxAreas(height):
    res, i, j = 0, 0, len(height) - 1

    while i < j:
        res = max(res, min(height[i], height[j]) * (j - i))
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return res


# 13 Roman to Integer, 两种规则记住不难
def romanToInt(s):
    res = 0
    m = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}

    for i in range(len(s)):
        val = m[s[i]]
        if i == len(s) - 1 or m[s[i+1]] <= m[s[i]]:
            res += val
        else:
            res -= val
    return res


# 16 3Sum closest, 排序后用一个变量记录 diff 值
def threeSumClosest(nums, target):
    nums, closest = sorted(nums), nums[0]+nums[1]+nums[2]
    diff = abs(target-closest)

    for i in range(len(nums)-2):
        lo, hi = i + 1, len(nums) - 1

        while lo < hi:
            ans = nums[i] + nums[lo] + nums[hi]
            newdiff = abs(ans - target)

            if diff > newdiff:
                diff, closest = newdiff, ans

            if sum < target:
                lo += 1
            else:
                hi -= 1
    return closest

# 17 Letter Combinations of a Phone Number, 回溯法套路
def letterCombinations(digits):
    if not digits:
        return ""
        
    m = {"2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs",
    "8":"tuv", "9":"wxyz"}

    res = []

    def dfs(digits, level, out):
        if level == len(digits):
            res.append(out)
            return

        for char in m[digits[level]]:
            dfs(digits, level + 1, out + char)

    dfs(digits, 0, "")

    return res


# 18 4sum, 注意去重的操作
def foursum(nums, target):
    nums, res = sorted(nums), []

    for i in range(len(nums)-3):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        for j in range(i+1, len(nums)-2):
            if j > i + 1 and nums[j] == nums[j-1]:
                continue
            lo, hi = j + 1, len(nums) - 1

            while lo < hi:
                val = nums[i] + nums[j] + nums[lo] + nums[hi]

                if val == target:
                    res.append([nums[i], nums[j], nums[lo], nums[hi]])
                    while lo < hi and nums[lo] == nums[lo + 1]:
                        lo += 1
                    while lo < hi and nums[hi] == nums[hi - 1]:
                        hi -= 1
                    lo, hi = lo + 1, hi - 1

                elif val > target:
                    hi -= 1
                else:
                    lo += 1
    return res


# 22 Generate Parentheses, 字符串只有左右括号两种形式，最终结果必定是 n 个左括号, n
# 个右括号，left right 表示剩余左右括号的个数
def generateParenthesis(n):
    res = []

    def dfs(left, right, out):

        if left == 0 and right == 0:
            res.append(out)
            return

        if left > 0:
            dfs(left - 1, right, out + "(")

        if left < right: # 此时可以开始加右括号
            dfs(left, right - 1, out + ")")

    dfs(n, n, "")

    return res


# 23 Merge k Sorted lists
def mergeKLists(lists):
    n = len(lists)
    while n > 1:
        k = (n + 1) // 2
        for i in range(n//2):
            lists[i] = mergeTwoLists(lists[i], lists[i+k])
        n = k
    return lists[0]

def mergeTwoLists(l1, l2):
    dumpy = ListNode(0)
    cur = dumpy
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 if l1 else l2
    return dumpy.next


# 25 Reverse Nodes in k-Group, 注意每k组一翻转的操作
def reverseKGroup(head, k):
    dumpy = ListNode(-1)
    pre, cur, num, dumpy.next = dumpy, head, 0, head

    while cur:
        num += 1
        cur = cur.next

    while num >= k:
        cur = pre.next

        for i in range(1, k):
            t = cur.next
            cur.next = t.next
            t.next = pre.next
            pre.next = t

        pre, num = cur, num - k
    return dumpy.next


# 26 Remove duplicates from sorted array, 快慢指针法
def removeDuplicates(nums):
    n = len(nums)
    if n == 0:
        return 0
    i = 0
    j = 0
    while j < n:
        if nums[i] != nums[j]:
            i += 1
            nums[i] = nums[j]
        j += 1
    return i + 1


# 31 Next Permutation, 从后往前找到递增数列结束的位置，然后从后往前找比该位置
# 大的第一个数，然后交换位置，然后从 i+1 到结尾翻转
def nextPermutation(nums):
    n, j = len(nums), len(nums) - 1

    for i in range(n-2, -1, -1):
        if nums[i+1] > nums[i]:
            while j > i:
                if nums[j] > nums[i]:
                    break
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]

            for k in range(i+1, (n+i)//2+1):
                nums[k], nums[n-k+i] = nums[n-k+i], nums[k]
            return
    
    for i in range(n//2):
        nums[i], nums[n-1-i] = nums[n-1-i], nums[i]


# 32 Longest Valid Parentheses, 通项公式dp[i] 表示以 s[i-1] 结尾的最长长度
def longestValidParentheses(s):
    dp, res = [0 for _ in range(len(s)+1)], 0

    for i in range(1, len(s)+1):
        j = i - 2 - dp[i-1]

        if s[i-1] == "(" or j < 0 or s[j] == ")":
            dp[i] = 0

        else:
            dp[i] = dp[i-1] + 2 + dp[j]
            res = max(res, dp[i])

    return res


# 33 Search in Rotated Sorted Array, 注意旋转数组与最右的数字比较大小而不是最左
def search(nums, target):
    low = 0
    high = len(nums) - 1
    
    while low <= high:
        mid = low + (high - low) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < nums[high]:#右半段有序
            if nums[mid] < target and target <= nums[high]:
                low = mid + 1
            else:
                high = mid - 1
        else:
            if nums[mid] > target and target >= nums[low]:
                high = mid - 1
            else: 
                low = mid + 1
    return -1


# 34 Find First and Last Position of Element in Sored Array, 注意 mid 的值在两次查询中的设置
def searchRange(nums, target):
    res = [-1, -1]
    if not nums:
        return res

    lo, hi = 0, len(nums)-1

    # 查找最左侧值
    while lo < hi:
        mid = （lo + hi）// 2
        if nums[mid] >= target:
            hi = mid
        else:
            lo = mid + 1
    if nums[lo] != target:
        return res

    # 查找最右侧值
    res[0], hi = lo, len(nums) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if nums[mid] <= target:
            lo = mid
        else:
            hi = mid - 1
    res[1] = lo
    return res


# 39 Combination Sum
def combinationSum(candidates, target):
    res, ans =[], []
    dfs(candidates, 0, target, ans, res)
    return res

def dfs(candidates, i, target, ans, res):
    if target == 0:
        res.append(ans[:])
        return
    elif target < 0:
        return
    else:
        for start in range(i, len(candidates)):
            ans.append(nums[start])
            dfs(candidates, start, target-candidates[start], ans, res)
            ans.pop()

# 40 Combination Sum II
def combinationSum2(nums, target):
    ans, res = [], []
    dfs(nums, 0, target, ans, res)
    return res

def dfs(nums, start, target, ans, res):
    if target == 0:
        res.append(ans[:])
        return
    elif target < 0:
        return
    for i in range(start, len(nums)):
        if i == start or nums[i] != nums[i-1]:
            out.append(nums[i])
            dfs(nums, i+1, target-nums[i], ans, res)
            out.pop()

# 41 First Missing Positive, 将 i 放到下角标 i - 1 处
def firstMissingPositive(nums):
    i = 0
    while i < len(nums):
        if 0 < nums[i] <= len(nums):
            index = nums[i]
            if nums[i] != nums[index - 1]:
                nums[i], nums[index - 1] = nums[index - 1], nums[i]
                i -= 1
        i += 1

    for i in range(len(nums)):
        if nums[i] != i + 1:
            return i + 1
    return len(nums) + 1  

# 42 Trapping Rain Water, 递减栈，遇到高于栈顶的元素触发计算, 此时弹出栈顶
# 元素，这就是坑，现在的栈顶元素与当前高度较小的那个是挡板，减去坑，乘以宽
def trap(height):
    st, i, res, n = [], 0, 0, len(height)

    while i < n:

        if not st or height[st[-1]] >= height[i]:
            st.append(i)

        else:
            cur = st.top()

            if not st:
                continue

            res += (min(height[i], height[st[-1]]) - height[cur]) * (i - 1 - st[-1])
    return res


# 43 Multiply Strings,不要急着进位，不然就炸了
def multiply(num1, num2):
    return help(num1[::-1], num2[::-1])


def help(a, b):
    vals, carry = [0] * (len(a) + len(b)), 0

    for i in range(len(a)):
        for j in range(len(b)):
            vals[i+j] += int(a[i]) * int(b[j])

    for i in range(len(vals)):
        val = vals[i] + carry
        vals[i], carry = val % 10, val // 10

    for i in range(len(vals)-1, -1, 0): # 将为首的0去掉
        if vals[i] != 0:
            break

    vals = vals[:i+1]

    return "".join(str(x) for x in vals[::-1])
    

# 44 Wildcard Matching, 状态转移方程要确定
def isMatch(s, p):
    dp = [[False for _ in range(len(p)+1)] for _ in range(len(s)+1)]
    dp[0][0] = True

    for i in range(1, len(p)+1):
        if p[i-1] == "*": #初始状态一定要确定好
            dp[0][i] = dp[0][i-1]

    for i in range(1, len(s)+1):
        for j in range(1, len(p)+1):
            if p[j-1] == "*":#等于*的处理情况
                dp[i][j] = dp[i-1][j] or dp[i][j-1]

            else:
                dp[i][j] = (p[j-1] == "?" or s[i-1] == p[j-1]) and dp[i-1][j-1]

    return dp[len(s)][len(p)]

# 45 Jump Game II, cur 表示当前能跳的最远位置， last 表示上一步能跳的最远位置
# 当 last 大于 i 的时候，需要更新res了，并且上一步最远的位置需要更新了
def jump(nums):
    res, cur, last = 0, 0, 0

    for i in range(len(nums)):
        if i > last: 
            res += 1
            last = cur
        cur = max(cur, i + nums[i])
    return res


# 46 Permutations
def permute(nums):
    out, res, visited = [], [], [0] * len(nums)
    dfs(nums, 0, out, res, visited)
    return res

def dfs(nums, start, out, res, visited):
    if start == len(nums):
        res.append(out[:])
    for i in range(0, len(nums)):
        if not visited[i]:
            visited[i] = 1
            out.append(nums[i])
            dfs(nums, start+1, out, res, visited)
            out.pop()
            visited[i] = 0

# 48 Rotate Image, 记住转置只走对角线的循环，别走整个，不然不变
def rotate(matrix):
    m = len(matrix)
    if m == 0:
        return
    n = len(matrix[0])
    for i in range(m):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    for i in range(m):
        matrix[i] = matrix[i][::-1]
            
# 49 Group Anagrams, 排序之后这些错位词就都一样了
def groupAnagrams(strs):
    m = {}
    res = []
    
    for word in strs:
        key = "".join(sorted(word))
        if key not in m:
            m[key] = []
        m[key].append(word)
    
    for key in m:
        res.append(m[key])
    return res

# 50 Pow(x, n)
def myPow(x, n):
    if n == 0:
        return 1
    na = abs(n)
    half = myPow(x, na//2)
    res = half * half * x if na % 2 else half * half
    return res if n > 0 else 1 / res