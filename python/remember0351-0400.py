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