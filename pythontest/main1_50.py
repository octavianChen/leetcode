# !/usr/bin/python
# -*- coding=UTF-8 -*-


import sys
import queue
sys.path.append("D:\\Leetcode\\function")
#from ans1_50 import *
import random

def binarysearch(nums, target):
    lo, hi = 0, len(nums)-1
    count = 0
    while lo < hi:
        mid = (lo + hi)//2
        print("***********************step{}***********************".format(count))
        print("lo: ", lo, "nums[lo]: ", nums[lo])
        print("mid: ", mid, "nums[mid]: ", nums[mid])
        print("hi: ", hi, "nums[hi]: ", nums[hi])
        print("******************************************************************")
        if nums[mid] >= target:
            hi = mid
        else:
            lo = mid + 1
        count += 1
    print(lo, hi)
    return -1 if nums[lo] != target else lo


def threeSumClosest(nums, target):
    nums = sorted(nums)
    closest = nums[0] + nums[1] + nums[2]
    diff = abs(closest-target)

    print(closest, diff)

    for i in range(len(nums)-2):
        lo, hi = i+1, len(nums)-1

        while lo < hi:
            ans = nums[i] + nums[lo] + nums[hi]
            newdiff = abs(ans-target)
            if newdiff < diff:
                diff, closest = newdiff, ans
                print("After min", diff, closest)
            if ans < target:
                lo += 1
            else:
                hi -= 1
    return closest


if __name__ == "__main__":
    nums = [-1, 2, 1, -4]
    b = sorted(list(set(nums)))
    print(b)
    print(nums)


def isInterleave(s1, s2, s3):
    m, n = len(s1), len(s2)

    if m + n != len(s3):
        return False

    dp = [[False for _ in range(n+1)] for _ in range(m+1)]
    dp[0][0] = True

    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]

    for i in range(1, n+1):
        dp[0][i] = dp[0][i-1] and s2[i-1] == s3[i-1]

    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or (dp[i][j-1] amd s2[j-1] == s3[i+j-1])

    return dp[m][n]  












    




