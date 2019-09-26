# !/usr/bin/python
# -*- coding=UTF-8 -*-

from functools import reduce


class ListNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None


# 1 two sum
def twosum(nums, target):
    res = []
    n = len(nums)
    m = {}

    for i in range(n):
        t = target - nums[i]
        if t in m and i != m[t]:
            res.append(m[t])
            res.append(i)
            break
        m[nums[i]] = i
    return res


# 2 add two numbers
def addtwonum(l1, l2):
    res = ListNode(-1)
    cur = res
    carry = 0

    while l1 or l2:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0

        sum = (val1 + val2 + carry)%10
        carry = (val1 + val2 + carry) // 10

        cur.next = ListNode(sum)
        cur = cur.next

        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    if carry:
        cur.next = ListNode(carry)
    return res


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


# 4 Meidan of Two Sorted Arrays
def findMedianSortedArrays(self, nums1, nums2):
    m = len(nums1)
    n = len(nums2)
    left = (m + n + 1) // 2
    right = (m + n + 2) // 2
    return findKth(nums1, 0, nums2, 0, left) + findKth(nums1, 0, nums2, 0, right)


def findKth(nums1, i, nums2, j, k):
    if i >= len(nums1):
        return nums2[j + k - 1]
    if j >= len(nums2):
        return nums1[i + k - 1]
    if k == 1:
        return min(nums1[i], nums2[j])

    midval1 = nums1(i + k//2 - 1) if i + k//2 - 1 < len(nums1) else float("inf")
    midval2 = nums2(j + k//2 - 1) if j + k//2 - 1 < len(nums2) else float("inf") 

    if midval1 < midval2:
        return findKth(nums1, i + k//2, nums2, j, k - k//2)
    else:
        return findKth(nums1, i, nums2, j + k//2, k - k//2)


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
        return min(nums1[i], nums2[j])
    midv1 = nums1[i + k//2 - 1] if i + k //2 -1 < len(nums1) else float("inf")
    midv2 = nums2[j + k//2 - 1] if j + k //2 -1 < len(nums2) else float("inf")

    if midv1 < midv2:
        return findMedianSortedArrays1(nums1, i + k//2, nums2, j, k-k//2)
    else:
        return findMedianSortedArrays1(nums1, i, nums2, j + k//2, k-k//2)


# 5 Longest Palindromic Substring
def longestPalindrome(self, s):
    n = len(s)
    dp = [[None for _ in range(n)] for _ in range(n)]
    res = 0
    start = 0
    
    if n == 0:
        return s
    
    for l in range(1, n+1):
        for i in range(n-l+1):
            j = i + l - 1
            if l <= 2:
                dp[i][j] = (s[i] == s[j])
            else:
                dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
            
            if dp[i][j]:
                res = max(res, j-i+1)
                start = i
    return s[start:start+res]


# 6 ZigZag Conversion
def convert(s, numRows):

    if numRows <= 1:
        return s

    res = ""
    for i in range(numRows):
        for j in range(i, len(s), 2*(numRows-1)):
            res += s[j]

            tmp = j + 2*(numRows-1) - 2*i

            if i != 0 and i != numRows-1 and tmp < len(s):
                res += s[tmp]
    return res
    


# 7 Reverse Integer
def reverse(x):
    val = abs(x)
    res = 0

    while val:
        res = res * 10 + val % 10
        val //= 10

    res = res if x >= 0 else -res
    if res > 2147483647 or res < -2147483648:
        return 0
    return res


# 8 String to integer
def myAtoi(strs):
    s = strs.strip()
    res = ""
    ans = 0
    
    for i in range(len(s)):
        if i == 0:
            if s[i] in "+-" or s[i].isdigit():
                res += s[i]
            else:
                break
        else:
            if s[i].isdigit():
                res += s[i]
            else:
                break
    
    if not res or res in "+-":
        ans = 0
    elif res[0] == "+":
        ans = int(res[1:])
    else:
        ans = int(res)

    ans = -2147483648 if ans < -2147483648 else ans
    ans = 2147483647 if ans > 2147483647 else ans
    return ans


# 9 Palindrome Number
def isPalindrome(x):
    if x < 0:
        return False

    res = 0 
    val = x

    while val:
        res = res * 10 + val % 10
        val //= 10

    return res == x


# 10 Regular Expression Matching
def isMatch(s, p):
    if not p:   
        return True if not s else False

    if len(p) == 1: 
        return (len(s) == 1) and (s[0] == p[0] or p[0] == ".")

    if p[1] != "*":
        if not s:
            return False
        else:
            return (s[0] == p[0] or p[0] == ".") and isMatch(s[1:], p[1:])

    while s and (s[0] == p[0] or p[0] == "."):
        if isMatch(s, p[2:]):
            return True
        s = s[1:]

    return isMatch(s, p[2:])



# 11 Container with the most water
def maxAreas(height):
    res, i, j = 0, 0, len(height) - 1

    while i < j:
        res = max(res, min(height[i], height[j]) * (j - i))
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return res


# 13 Roman to Integer
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


# 14 Longest Common Prefix
def longestCommonPrefix(strs):
    n = len(strs)
    res = ""
    if n == 0:
        return res
    m = len(strs[0])
    
    for i in range(m):
        c = strs[0][i]
        for j in range(n):
            if i >= len(strs[j]) or c != strs[j][i]:
                return res
        res += c
    return res


# 15 3Sum


# 16 3Sum closest
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

# 17 Letter Combinations of a Phone Number
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


# 19 Remove Nth Node From End of List
def removeNthFromEnd(head, n):
    pre = ListNode(0)
    cur, pre.next, count, m = pre, head, 0, {}
    
    while cur:
        m[count] = cur
        count += 1
        cur = cur.next
    
    dest = m[count - n - 1]
    dest.next = dest.next.next
    return pre.next


# 20 Valid Parentheses
def isvalid(s):
    m = []
    for c in s:
        if c in "([{":
            m.append(c)
        else:
            if len(m) == 0:
                return False
            elif c == ")" and m[-1] != "(":
                return False
            elif c == "]" and m[-1] != "[":
                return False
            elif c == "}" and m[-1] != "{":
                return False
            else:
                m.pop()
    return len(m) == 0


# 21 Merge Two Sorted Lists
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


# 22 Generate Parentheses
def generateParenthesis(n):
    res = []

    def dfs(left, right, out):
        if left > right:
            return

        if left == 0 and right == 0:
            res.append(out)

        if left > 0:
            dfs(left - 1, right, out + "(")

        if right > 0:
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


# 24 Swap Nodes in Pairs
def swapPairs(head):
    dumpy = ListNode(0)
    if head is None or head.next is None:
        return head
    
    cur, cur1, cur2 = dumpy, head, head.next
    while cur1 and cur2:
        tmp1, tmp2 = cur2, cur1
        cur1 = cur2.next
        cur2 = cur1.next if cur1 else None
        cur.next, cur.next.next = tmp1, tmp2
        cur = tmp2
    cur.next = cur1 if cur1 else None
    return dumpy.next


# 25 Reverse Nodes in k-Group
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


# 27 Remove elements, 快慢指针法
def removeElement(nums, val):
    n = len(nums)
    i = -1
    for j in range(n):
        if nums[j] != val:
            i += 1
            nums[i] = nums[j]
    return i+1


# 28 Implement strStr()
def strStr(haystack, needle):
    m = len(haystack)
    n = len(needle)

    if n == 0:
        return 0

    for i in range(m-n+1):
        if haystack[i:i+n] == needle:
            return i
    return -1

# 30 Substring with Concatenation of All Words
def findSubstring(s, words):
    if not words:
        return False

    m, r, c, res = {}, len(words), len(words[0]), []

    for word in words:
        m[word] = m.get(word, 0) + 1

    for i in range(len(s) - r * c + 1):
        n = {}
        for k in range(r):
            candi = s[i + (k-1) * c : i + k * c]
            if candi not in m:
                break
            n[candi] = n.get(candi, 0) + 1

        if m == n:
            res.append(i)
    return res

# 31 Next Permutation


# 32 Longest Valid Parentheses, 通项公式dp[i] 表示以 s[i-1] 结尾的最长长度
def longestValidParentheses(s):
    




# 33 Search in Rotated Sorted Array
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


# 34 Find First and Last Position of Element in Sored Array
def searchRange(nums, target):
    res = [-1, -1]
    if not nums:
        return res

    lo, hi = 0, len(nums)-1

    # 查找最左侧值
    while lo < hi:
        mid = (lo + hi)// 2
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


# 35 Search Insert Position
def solution(nums, target):
    lo, hi = 0, len(nums)-1
    while lo < hi:
        mid = (lo + hi + 1) //2
        if nums[mid] <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo if nums[lo] >= target else lo+1


# 39 Combination Sum
def combinationSum(candidates, target):
    res, ans =[], []
    dfs(candidates, 0, target, ans, res)
    return res

def dfs(candidates, i, target, ans, res):
    if targrt == 0:
        res.append(ans[:])
        return
    elif taget < 0:
        return
    else:
        for start in range(i, len(candidates)):
            ans.append(en)
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
    elif target:
        return
    for i in range(start, len(nums)):
        if i == start or nums[i] != nums[i-1]:
            out.append(nums[i])
            dfs(nums, i+1, target-nums[i], ans, res)
            out.pop()

# 42 Trapping Rain Water, 递减栈，遇到高于栈顶的元素触发计算
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

# 46 Permutations
def permute(nums):
    out, res, visited = [], [], [0] * len(nums)
    dfs(nums, 0, out, res, visited)
    return res

def dfs(nums, start, out, res, visited):
    if len(out) == len(nums):
        res.append(out[:])
    for i in range(start, len(nums)):
        if not visited[i]:
            visited[i] = 1
            out.append(nums[i])
            dfs(nums, start+1, out, res, visited)
            out.pop()
            visited[i] = 0

# 47 Permutations II
def permuteUnique(nums):
    res = set()

    def dfs(i):
        if i >= len(nums):
            res.add(tuple(nums))

        for j in range(i, len(nums)):
            if j == i or nums[j] != nums[i]:
                nums[i], nums[j] = nums[j], nums[i]
                dfs(i+1)
                nums[i], nums[j] = nums[j], nums[i]

    dfs(0)
    return map(list, res)


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


# 49 Group Anagrams
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