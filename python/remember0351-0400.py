# 368 Largest Divisible Subset, 排序后dp[i]表示到nums[i]位置的最长可整除子集合的长度
def largestDivisibleSubset(nums):
	nums = sorted(nums)
	dp, p, res, mx, mx_idx = [0 for _ in range(len(nums))], [0 for _ in range(len(nums))], [], 0, 0

	for i in range(len(nums)-1, -1, -1):
		for j in range(i, len(nums)):
			if nums[j] % nums[i] == 0 and dp[i] < dp[j] + 1:
				dp[i], p[i] = dp[j] + 1, j

			if mx < dp[i]:
				mx, mx_idx = dp[i], i

	for _ in range(mx):
		res.append(nums[mx_idx])
		mx_idx = p[mx_idx]
	return res


# 374 Wiggle Subsequence
def wiggleSubsequence(nums):
	

# 378 Kth Smallest Element in a Sorted Matrix, 从矩阵左下角到右上角的阶梯型遍历
def kthSmallest(matrix, k):
	lval, hval = matrix[0][0], matrix[-1][-1] # 矩阵左上角和右下角的元素

	while lval <= hval:
		mval = (lval + hval) // 2

		cnt = countLower(matrix, mval)

		if cnt >= k:
			hval = mval - 1

		else:
			lval = mval + 1

	return lval

def countLower(matrix, num):
	i, j, cnt = len(matrix) - 1, 0, 0

	while i >= 0 and j < len(matrix[0]):
		if matrix[i][j] <= num:
			cnt, j = cnt + i + 1, j + 1

		else:
			i -= 1

	return cnt


# 392 Is Subsequence, 很巧妙，不要用 dp，可以做，但耗时
def isSubsequence(s, t):
	m, n, i = len(s), len(t), 0 # s 是否是 t 的子序列

	for j in range(n):
		if i < m and s[i] == t[j]:
			i += 1
	return i == m

# 394 Decode String
def decodeString(s):
	curnum, curstr, st = 0, "", []

	for char in s:
		if char == "[":
			st.append(curstr)
			st.append(curnum)
			curstr, curnum = "", 0

		elif char == "]":
			prenum = st.pop()
			prestr = st.pop()
			curstr = prestr + prenum * curstr

		elif char.isdigit():
			curnum = curnum * 10 + int(char)

		else:
			curstr += char

	return curstr