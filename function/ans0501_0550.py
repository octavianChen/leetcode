class TreeNode(object):
    def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

# 501  Find Mode in Binary Search Tree
def findMode(root):
	res, p, candi, m = [], root, 0, {}

	if not root:
		return res

	while st or p:
		while p:
			st.append(p)
			p = p.left

		p = st.pop()

		m[p.val] = m.get(p.val, 0) + 1
		candi = max(candi, m[p.val])
		p = p.right

	for key, val in m.items():
		if val == candi:
			res.append(key)
	return res


# 504 Base 7
def convertToBase7(num):
	if num == 0:
		return "0"
	res, nums = [], abs(num)

	while nums:
		res.append(nums % 7)
		nums = nums // 7

	if num < 0:
		res.append("-")

	return "".join(str(x) for x in res[::-1]) 


# 508 Most Frequent Subtree Sum
def findFrequentTreeSum(self, root):
    m, res, max_val= {}, [], -float("inf")
    if not root:
    	return res

    help(root, m)
    
    max_val = max(m[key] for key in m)
    
    for key, val in m.items():
        if val == max_val:
            res.append(key)
    return res

def help(root, m):
	if not root:
		return 0

	val1 = help(root.left, m)
	val2 = help(root.right, m)

	total = val1 + val2 + root.val

	m[total] = m.get(total, 0) + 1
	return total


# 509 Fibonacci Number
def fib(N):
	if N == 0 or N == 1:
		return N

	pre, now = 0, 1
	for i in range(2, N):
		pre, now = now, pre + now

	return now


# 513 Find Bottom Left Tree Value
def findBottomLeftValue(root):
	res, queue = 0, []
	queue.append(root)
	while queue:
		res = [x.val for x in queue][0]
		new_queue = []

		for p in queue:
			if p.left:
				new_queue.append(p.left)
			if p.right:
				new_queue.append(p.right)
		queue = new_queue
	return res


# 515 Find Largest Value in Each Tree Row
def largestValues(root):
	res, q = [], [] 

	if root:
		q.append(root)

	while q:
		res.append(max(p.val for p in q))

		new_queue = []

		for p in q:
			if p.left:
				new_queue.append(p.left)
			if p.right:
				new_queue.append(p.right)
		q = new_queue
	return res

# 516 Longest Palindromic Subsequence
def longestPalindromeSubseq(s):
	n = len(s)

	dp = [[0 for _ in range(n)] for _ in range(n)]

	for i in range(n-1, -1, -1):
		dp[i][i] = 1

		for j in range(i+1, n):
			if s[i] == s[j]:
				dp[i][j] = dp[i+1][j-1] + 2
			else:
				dp[i][j] = max(dp[i+1][j], dp[i][j-1])
	return dp[0][n-1]

# 520 Detect Capital
def detectCapitalUse(word):
	if len(word) <= 1:
		return True

	if word[0].isupper() and word[1].isupper():
		for i in range(2, len(word)):
			if not word[i].isupper():
				return False
		return True

	else:
		for i in range(1, len(word)):
			if not word[i].islower():
				return False
		return True

# 521 Longest Uncommon Subsequence I
def findLUSlength(a, b):
	return -1 if a == b else max(len(a), len(b))


# 523 Contiguous Subarray Sum
def checkSubarraySum(nums, k):
	m, val = {0:-1}, 0

	for i in range(len(nums)):
		val += nums[i]

		if k == 0:
			if val in m:
				if i - m[val] > 1:
					return True

			else:
				m[val] = i

		else:
			if val % k in m:
				if i - m[val % k] > 1:
					return True
			else:
				m[val % k] = i
	return False


# 530 Minimum Absolute Difference in BST
def getMinimumDifference(root):
	m, ans = [], float("inf")
	inorder(root, m)

	for i in range(len(m)):
		if i < len(m) - 1:
			ans = min(ans, abs(m[i+1]-m[i]))
	return ans

def inorder(root, m):

	if not root:
		return

	inorder(root.left)
	m.append(root.val)
	inorder(root.right)


# 532 K-diff Pairs in an Array
def findPairs(nums, k):
	

# 537 Complex Number Multiplication
def complexNumberMultiply(a, b):
	import re
	alist, blist = re.split(r"[+i]", a), re.split("[+i]", b)

	ar, br, ac, bc = int(alist[0]), int(blist[0]), int(alist[1]), int(blist[1])

	r, c = ar * br - ac * bc, ar * bc + br * ac

	return str(r) + "+" + str(c) + "i"


# 538 Convert BST to Greater Tree
class Solution(object):
    def convertBST(self, root):
    	self.sum = 0

    	def help(root):
    		if not root:
    			return

    		help(root.right)
    		root.val += self.sum
    		self.sum = root.val
    		help(root.left)

    	help(root)
    	return root

# 541 Reverse String II
def reverseStr(s, k):
	cnt, r = len(s) // (2*k), len(s) % (2*k)

	ans = ""

	for i in range(cnt):
		ans = ans + s[2*i*k:2*i*k + k][::-1] + s[2*i*k + k:2*i*k+2*k]

	if r < k:
		ans += s[len(s)- r : len(s)][::-1]
	else:
		ans += s[len(s)-r:len(s)-(r-k)][::-1] + s[len(s)-(r-k):len(s)]
	return ans

# 542 01 Matrix, BFS, 起点为都是 0 的点，距离场 BFS是首选
def updateMatrix(matrix):
	import Queue
	row, col, q = len(matrix), len(matrix[0]), Queue.Queue()

	for i in range(row):
		for j in range(col):
			if matrix[i][j] == 0:
				q.put((i, j))
			else:
				matrix[i][j] = float("inf")

	while not q.empty():
		i, j = q.get()
		direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

		for dirs in direction:
			x, y = i + dirs[0], j + dirs[1]

			if 0 <= x < row and 0 <= y < col and matrix[x][y] > matrix[i][j] + 1:
				matrix[x][y] = matrix[i][j] + 1
				q.push((x, y))
	return matrix

    		
# 543 Diameter of Binary Tree
def diameterOfBinaryTree(self, root):
	self.res = -float("inf")

	def help(node):
		if node is None:
			return 0
		lv = help(node.left)
		rv = help(node.right)

		self.res = max(self.res, lv + rv)

		return max(lv, rv) + 1

	help(root)
	return self.res

# 547 Friend Circles
def findCircleNum(self, M):
	res, row = 0, len(M)
        
    visited = [0 for _ in range(row)]
    
    def dfs(i):
        visited[i] = 1
        
        for j in range(row):
            if M[i][j] == 1 and visited[j] == 0:
                dfs(j)
            
    
    for i in range(row):
        if visited[i] == 0:
            dfs(i)
            res += 1
    return res

def findCircleNum(M): # 并查集
    res, n, root = len(M), len(M), [i for i in range(len(M))]

    for i in range(n):
        for j in range(i+1, n):
            if M[i][j] == 1:
                p1, p2 = find(root, i), find(root, j)

                if p1 != p2:
                    res -= 1
                    root[p1] = p2
    return res

def find(root, i):
    while root[i] != i:
        i = root[i]
    return i