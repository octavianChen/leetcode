# 652 Find Duplicate Subtrees, 序列化的操作是非常漂亮的，另外字典值等于1防止重复
def findDuplicateSubtrees(root):
	res, m = [], {}

	def help(node):
		if node is None:
			return "#"
		path = str(node.val) + ","
		path += help(node.left) + ","
		path += help(node.right)

		if m.get(path, 0) == 1:
			res.append(node)
		m[path] = m.get(path, 0) + 1
		return path

	help(root)
	return res

# 653 Two Sum IV - Input is a BST, 2Sum 的题用一定记得用集合来做
def findTarget(root, k):
	m =set()
	return help(root, k, m)

def help(root, k, m):
	if not root:
		return False

	if k - root.val in m:
		return True

	m.add(root.val)

	return help(root.left, k, m) or help(root.right, k, m)

# 655 Print Binary Tree, 两个队列，注意分析， 用递归也可以
def printTree(root):
	import math, Queue
	h, curH = getHeight(root), -1
	res = [["" for _ in range(int(math.pow(2, h)-1))] for _ in range(h)]
	q1, q2 = Queue.Queue(), Queue.Queue()
	q1.put(root)
	q2.put((0, len(res[0])-1))

	while not q1.empty():
		m, curH = q1.qsize(), curH + 1

		for _ in range(m):
			node = q1.get()
			left, right = q2.get()

			mid = (left + right) // 2

			res[curH][mid] = str(node.val)

			if node.left:
				q1.put(node.left)
				q2.put((left, mid - 1))

			if node.right:
				q1.put(node.right)
				q2.put((mid+1, right))
	return res

def getHeight(root):
	if root is None:
		return 0
	lh = getHeight(root.left)
	rh = getHeight(root.right)
	return max(lh, rh) + 1

# 665 Non-decreasing Array, 只有一次修改机会，注意如何修改
def checkPossibility(nums):
	cnt = 1

	for i in range(1, len(nums)):

		if nums[i] < nums[i-1]:
			if cnt == 0:
				return False

			if i == 1 or nums[i] > nums[i-2]:
				nums[i-1] = nums[i]

			else:
				nums[i] = nums[i-1]

			cnt -= 1
	return True

# 673 Number of Longest Increasing Subsequence, 用另一个数组记录LIS的个数,
# 注意状态转移方程
def findNumberOfLIS(nums):
	res, mx = 0, 0
	dp, cnt = [1 for _ in range(len(nums))], [1 for _ in range(len(nums))]

	for i in range(len(nums)):
		for j in range(i):
			if nums[j] < nums[i]:
				if dp[i] == dp[j] + 1:
					cnt[i] += cnt[j]

				elif dp[i] < dp[j] + 1:
					dp[i], cnt[i] = dp[j] + 1, cnt[j]

		if mx == dp[i]:
			res += cnt[i]

		elif mx < dp[i]:
			mx, res = dp[i], cnt[i]

	return res

# 680 Valid Palindrome II, 遇到有问题的各自删除就可以，然后检测剩下的是不是回文
def validPalindrome(s):
	lo, hi = 0, len(s) - 1
	while lo < hi:
		if s[lo] != s[hi]:
			return help(s[lo:hi]) or help(s[lo+1:hi+1])
		lo, hi = lo + 1, hi - 1
	return True

def help(s):
	return s == s[::-1]

# 684 Redundant Connection, 并查集的神操作，注意使用 Union
def findRedundantConnection(edges):
	root = [-1 for _ in range(2001)]

	for edge in edges:
		x, y = find(root, edge[0]), find(root, edge[1])

		if x == y:
			return edge
		root[x] = y
	return []

def find(root, i):
	while root[i] != -1:
		i = root[i]
	return i

# 685 Redundant Connection II, 三种情况， 入度为2的节点，有环的边，注意如何找出有环的边
# 以及并查集的各种神操作
def findRedundantDirectedConnection(edges):
	root, parent = [i for i in range(len(edges)+1)], [0 for _ in range(len(edges)+1)]
	first, second = [], []
	for edge in edges:
		if root[edge[1]] == edge[1]:
			root[edge[1]] = edge[0]
		else: # 节点的入度为2
			first.append(root[edge[:][1]])
			first.append(edge[:][1])
			second = [edge[:][0], edge[:][1]]
			edge[1] = 0

	root = [i for i in range(len(edges)+1)]

	for edge in edges:
		if edge[1] == 0:
			continue

		x, y = find(root, edge[0]), find(root, edge[1])

		if x == y:
			return first if first else edge
		root[x] = y
	return second

# 686 Repeated String Match, 重复+3, 2会报错，不需要重复太多
def repeatedStringMatch(A, B):
	a, b = len(A), len(B)

	times = b // a + 3

	for i in range(1, times):
		if B in A * i:
			return i
	return -1

