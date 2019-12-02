# 153 Find Minimum in Rotated Sorted Array, 首先判断是否旋转过
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

# 更简单的方法
def findMin(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi)//2

        if nums[mid] > nums[hi]:
            lo = mid + 1
        else:
            hi = mid
    return nums[lo]


# 154 Find Minimum in Rotated Sorted Array II
def findMin(nums):
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        if lo == hi - 1:
            break

        # 判断是否旋转过
        if nums[lo] < nums[hi]:
            return nums[lo]

        mid = (lo + hi) // 2

        # 断点位于左半部分
        if nums[lo] > nums[mid]:
            hi = mid

        # 断点位于右半部分
        elif nums[mid] > nums[hi]:
            lo = mid

        # 无法判断的时候，从左往右遍历
        else:
            while lo < mid:
                # 一直等就一直遍历
                if nums[low] == nums[mid]:
                    lo += 1

                # 断点找到
                elif nums[low] < nums[mid]:
                    return nums[low]

                # 最终一直等于就回到递归的时候
                else:
                    hi = mid
                    break


# 160 Intersection of Two Linked Lists
def getIntersectionNode(headA, headB):
    if not headA or not headB:
        return None

    a, b = headA, headB

    while a != b:
        a = a.next if a else headA
        b = b.next if b else headB
    return a


# 162 Find Peak Element, 要能看出二分的条件判断
def findPeakElement(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] < nums[mid + 1]:
            lef = mid + 1
        else:
            right = mid
    return right


# 168  Excel Sheet Column Title
def convertToTitle(n):
    ans = ""
    while n:
        res = n % 26

        n = n // 26 if res else (n-26)//26
        res = res if res else 26

        ans += chr(res + 64)

    return ans[::-1]

# 169  Majority Element, 每次删除两个不同的数，如果不一定存在的话还需要重新循环检验
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


# 171 Excel Sheet Column Number
def titleToNumber(s):
    res = 0
    for c in s:
        res = res * 26 + ord(c) - 64
    return res



# 191 Number of 1 Bits,n每次和n-1做与运算，都会将最右边的1变为0
def hammingWeight(n):
    res = 0
    while n:
        res += 1
        n = (n-1) & n
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