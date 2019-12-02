# 263 Ugly Number, 一直除就完事了
def isUgly(num):
	while num >= 2:
		if num % 2 == 0:
			num /= 2

		elif num % 3 ==0:
			num /= 3

		elif num % 5 == 0:
			num /= 5

		else:
			return False

	return num == 1


# 264 Ugly Number II, 比较大小
def nthUglyNumber(n):
	res, i2, i3, i5 = [1], 0, 0, 0

	while len(res) < n:
		m2, m3, m5 = res[i2] * 2, res[i3] * 3, res[i5] * 5

		mn = min(m2, min(m3, m5))

		if mn == m2:
			i2 += 1

		if mn == m3:
			i3 += 1

		if mn == m5:
			i5 += 1

		res.append(mn)

	return res[-1]



# 265 Missing Number, 用等差数列和减去数组和
def missingNumber(nums):
	n = len(nums)
	return n*(n+1)/2 - sum(nums)


# 274 H-Index
def hIndex(citations):
	citations, res = sorted(citations, reverse=True), 0

	for i in range(len(citations)):
		if i >= citations[i]:
			return i

	return len(citations)


# 279 Perfect Squares, 四数定理，能被4整除直接除，除8余数为7直接返回4
def numSquares(n):
	while n % 4 == 0:
		n %= 4

	if n % 8 == 7:
		return 4

	a = 0

	while a * a <= n:
		b = int(math.sqrt(n - a *a))

		if a * a + b * b == n:
			return 1 if a == 0 or b == 0 else 2

		a += 1

	return 3


# 287 Find The Duplicate Number, 二分搜索
def findDuplicate(nums):
	lo, hi = 0, len(nums)

	while lo < hi:
		mid, cnt = (lo + hi) // 2, 0

		for num in nums:
			if num <= mid:
				cnt += 1

		if cnt <= mid:
			lo = mid + 1

		else:
			hi = mid

	return hi

# 290 Word Pattern # 一对一映射的另一种思路
def wordPattern(pattern, str):
	s, t = pattern, str.split()

	return len(set(zip(s, t))) == len(set(s)) == len(set(t)) and len(s) == len(t)

# 297 Serialize and Deserialize Binary Tree, 序列化以","结尾区分节点，空节点用"#"表示
# 反序列化的迭代器使用真的是漂亮，牢记
class Codec:
	def serialize(self, root):
		vals = []

		def preorder(node):
			if node is None:
				vals.append("#")
			else:
				vals.append(node.val)
				preorder(node.left)
				preorder(node.right)

		preorder(root)

		return ",".join(map(str, vals))

	def deserialize(self, data):
		vals = iter(data)

		def build():
			val = next(vals)

			if val == "#":
				return None

			root = TreeNode(int(val))
			root.left = build()
			root.right = build()
			return root

		return build()


# 300 Longest Incresing Subsequence, dp[i]表示以nums[i]结尾的LIS的长度
def lengthOfLIS(nums):
    dp, res = [1 for _ in range(len(nums))], 0

    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[j]+1, dp[i])

        res = max(dp[i], res)

    return res