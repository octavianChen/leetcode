# 257 Binary Tree Path
def binaryTreePaths(root):
	out, res = [], []
	dfs(root, out, res)
	return res

def dfs(p, out, res):
	if not p:
		return

	out.append(str(p.val))

	if not p.left and not p.right:
		res.append("->".join(out[:]))
	
	dfs(p.left, out, res)
	
	dfs(p.right, out, res)

	out.pop()


# 263 Ugly Number
def isUgly(num):
	if num <= 0:
		return False
    while(2 * (num/2) == num):
        num = num/2
    while(3 * (num/3) == num):
        num = num/3
    while(5 * (num/5) == num):
        num = num/5
    return num == 1

# 264 Ugly Number II
def nthUglyNumber(n):
	

# 265 Missing Number, 用等差数列和减去数组和
def missingNumber(nums):
	n = len(nums)
	return n*(n+1)/2 - sum(nums)


# 274 H-Index
def hIndex(citations):
	citations = sorted(citations, reverse=True)

	for i in range(len(citations)):
		if i >= citations[i]:
			return i

	return len(citations)

# 275 H-Index II
def hIndex(citations):
	lo, hi = 0, len(citations) - 1

	while lo <= hi:
		mid = (lo + hi) // 2

		if citations[mid] == len(citations) - mid:
			return len(citations) - mid

		elif citations[mid] > len(citations) - mid:
			hi = mid - 1

		else:
			lo = mid + 1
	return len(citations) - mid

# 279 Perfect Squares
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

# 283 Move Zeros
def moveZeros(nums):
	i = -1

	for j in range(len(nums)):
		if nums[j] != 0:
			nums[i+1], nums[j] = nums[j], nums[i+1]
			i += 1


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

# 292 Nim Game
def canWinNim(n):
	return n % 4

# 297 Serialize and Deserialize Binary Tree
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
		vals = iter(data.split(","))

		def build():
			val = next(vals)

			if val == "#":
				return None

			root = TreeNode(int(val))
			root.left = build()
			root.right = build()
			return root

		return build()

# 290 Word Pattern # 一对一映射的另一种思路
def wordPattern(pattern, str):
	s, t = pattern, str.split()

	return len(set(zip(s, t))) == len(set(s)) == len(set(t)) and len(s) == len(t)

# 300 Longest Incresing Subsequence
def lengthOfLIS(nums):
    dp, res = [1 for _ in range(len(nums))], 0

    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[j]+1, dp[i])

        res = max(dp[i], res)

    return res