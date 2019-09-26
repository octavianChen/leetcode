# 652 Find Duplicate Subtrees,序列化的操作真的漂亮
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


# 653 Two Sum IV - Input is a BST
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

# 654 Maximum Binary Tree
def constructMaximumBinaryTree(nums):
	return help(nums, 0, len(nums)-1)

def help(nums, left, right):
	if left < right:
		return None

	pivot, index = -float("inf"), 0

	for i in range(left, right+1):
		if nums[i] > pivot:
			pivot, index = nums[i], i
	root = TreeNode(pivot)
	root.left = help(nums, left, index-1)
	root.right = help(nums, index+1, right)
	return root

# 655 Print Binary Tree
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

# 657 Robot Return to Origin
def judgeCircle(moves):
	m = {}

	for c in moves:
		m[c] = m.get(c, 0) + 1

	return m["L"] == m["R"] and m["U"] == m["D"]


# 661 Image Smoother
def imageSmoother(M):
	m, n = len(M), len(M[0])

	res = [[0 for _ in range(n)] for _ in range(m)]

	for i in range(m):
		for j in range(n):
			val, cnt = 0, 0
			for dx, dy in zip([-1, -1, -1, 0, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 0, 1, -1, 0, 1]):
				if 0 <= i + dx < m and 0 <= j + dy < n:
					val, cnt = val + M[i+dx][j+dy], cnt + 1

			res[i][j] = val // cnt

	return res

# 662 Maximum Width of Binary Tree
def widthOfBinaryTree(root):
	cnt, queue = 0, []
	if not root:
		return cnt

	queue.append((root, 0))

	while queue:
		new_queue = []
		cnt = max(queue[-1][1] - queue[0][1] + 1, cnt)
		for tup in queue:
			node, pos = tup

			if node.left:
				new_queue.append((node.left, 2*pos))

			if node.right:
				new_queue.append((node.right, 2*pos+1))
		queue = new_queue
	return cnt

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

# 669 Trim a Binary Search Tree
def trimBST(root, L, R):
	if root is None:
		return None

	elif root.val < L:
		return trimBST(root.right, L, R)
	elif root.val > R:
		return trimBST(root.left, L, R)
	else:
		root.left = trimBST(root.left, L, R)
		root.right = trimBST(root.right, L, R)
		return root


# 671 Second Minimum Node In a Binary Tree
def findSecondMinimumValue(root):
	f = s = float("inf")
	help(root, f, s)

	if s[0] == float("inf") or s[0] == f[0]:
		return -1
	return s[0]

def help(root, f, s):
	if not root:
		return

	if root.val < f:
		f, s = root.val, f
	elif f < root.val < s:
		s = root.val
	help(root.left, f, s)
	help(root.right, f, s)


# 674 Longest Continuous Increasing Subsequence
def findLengthOfLCIS(nums):
	if not nums:
		return 0
		
	dp = [1 for _ in range(len(nums))]

	for i in range(1, len(nums)):
		if nums[i] <= nums[i-1]:
			dp[i] = 1

		else:
			dp[i] = dp[i-1] + 1

	return max(dp)
 
# 676 Implement Magic Dictionary
class MagicDictionary(object):
	def __init__(self):
		self.m = set()

	def buildDict(self, dict):
		for word in dict:
			self.m.add(word)

	def search(self, word):
		word = list(word)

		for i in range(len(word)):
			for j in range(97, 123):
				if word[i] == chr(j):
					continue
				word[i] = chr(j)
			chg = "".join(word)

			if chg in self.m:
				return True
		return False

# 677 Map Sum Pairs
class MapSum(object):
	def __init__(self):
		self.root = {}

	def insert(self, key, val):
		p = self.root

		for char in key:
			if char not in p:
				p[char] = {}

			p = p[char]

		p["END"] = val

	def sum(self, prefix):
		self.res, p = 0, self.root

		for char in prefix:
			if char not in p:
				return self.res

			p = p[char]

		def dfs(t):
			if "END" in t:
				self.res += t["END"]

			for key in t:
				if key != "END":
					dfs(t[key])

		dfs(p)

		return self.res


# 678 Valid Parenthesis String
def 


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

# 684 Redundant Connection
def findRedundantConnection(edges):
	root = [i for i in range(2001)]

	for edge in edges:
		x, y = find(root, edge[0]), find(root, edge[1])

		if x == y:
			return edge
		root[x] = y
	return []

def find(root, i):
	while root[i] != i:
		i = root[i]
	return i

# 685 Redundant Connection II
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
			


def find(root, i):
	while root[i] != i:
		i = root[i]
	return i


# 686 Repeated String Match, 重复+3, 2会报错
def repeatedStringMatch(A, B):
	a, b = len(A), len(B)

	times = b // a + 3

	for i in range(1, times):
		if B in A * i:
			return i
	return -1

# 687 Longest Univalue Path
def longestUnivaluePath(root):

	self.res = 0

	def dfs(node):
		if node is None:
			return None, -1

		lval, llen = dfs(node.left)
		rval, rlen = dfs(node.right)

		if lval != node.val and rval != node.val:
			self.res = max(self.res, 0)
			return node.val, 0

		elif lval != node.val:
			self.res = max(self.res, rlen + 1)
			return node.val, rlen + 1
		elif rval != node.val:
			self.res = max(self.res, llen + 1)
			return node.val, llen + 1
		else:
			self.res = max(self.res, llen + rlen + 2)
			return node.val, max(llen, rlen) + 1

	dfs(root)

	return self.res


# 690 Employee Importance


# 692 Top K Frequent Words
def topKFrequent(words, k):
	m, res, candi = {}, [], 0

	for word in words:
		m[word] = m.get(word, 1) + 1

	vec = sorted(m.items(), key=lambda x:(-x[1], x[0]))

	for i in range(k):
		res.append(vec[i][0])

	return res

# 695 Max Area of Island
class Solution(object):
	def maxAreaOfIsland(self, grid):
		self.cnt = 0

		if not grid or not grid[0]:
			return self.cnt

		row, col = len(grid), len(grid[0])

		def dfs(i, j):
			direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]
			grid[i][j], self.num = -1, self.sum + 1
			for dirs in direction:
				r, c = i + dirs[0], j + dirs[1]

				if 0 <= r < row and 0 <= c < col and grid[r][c] == 1:
					dfs(r, c)

		for i in range(row):
			for j in range(col):
				if 0 <= i < row and 0 <= j < col and grid[i][j] == 1:
					self.num = 0
					dfs(i, j)
					self.cnt = max(self.cnt, self.num)

		return self.cnt

# 697 Degree of an Array
def findShortestSubArray(nums):
	m, vec = {}, []

	for i in range(len(nums)):
		if nums[i] not in m:
			m[nums[i]] = [i, i, 1]
		else:
			m[nums[i]][1] = i
			m[nums[i]][2] += 1 

	for key, val in m:
		vec.append((key, val[1] - val[0], val[2]))

	vec = sorted(vec, key=lamda )

# 700 Search in a Binary Search Tree
def searchBST(root, val):

	if not root:
		return None

	if root.val == val:
		return root

	return searchBST(root.left, val) if root.val > val else searchBST(root.right, val)
