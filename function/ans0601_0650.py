# 605 Can Place Flowers
def canPlaceFlower(flowerbed, n):
	

# 606 Construct String from Binary Tree
def tree2str(t):
	if not t:
		return ""
	res = [""]
	help(t, res)
	return res[0][1:-1]

def help(t, res):
	if not t:
		return

	res[0] += "(" + str(t.val)

	if not t.left and t.right:
		res += "()"
	help(t.left, res)
	
	help(t.right, res)
	res[0] += ")"


# 611 Valid Triangle Number
def triangleNumber(nums):



# 617 Merge Two Binary Trees
def mergeTrees(t1, t2):
	if not t1 and not t2:
		return None
	if not t1 or not t2:
		return t1 if t1 else t2
	root = TreeNode(t1.val + t2.val)
	root.left = mergeTrees(t1.left, t2.left)
	root.right = mergeTrees(t1.right, t2.right)
	return root


# 621 Task Scheduler
def leastInterval(tasks, n):
	

# 623 Add One Row to Tree
def addOneRow(root, v, d):
	if d == 1:
		t = TreeNode(v)
		t.left = root
		return t

	queue, cnt = [], 1
	queue.append(root)

	while queue:
		new_queue = []
		for node in queue:
			if d - 1 == cnt:
				tmp = node.left
				node.left = TreeNode(v)
				node.left.left = tmp
				tmp = node.right
				node.right = TreeNode(v)
				node.right.right = tmp
			else:
				if node.left:
					new_queue.append(node.left)
				if node.right:
					new_queue.append(node.right)
		queue, cnt = new_queue, cnt + 1
	return root

# 633 Sum of Square Numbers
def judgeSquareSum(c):
	import math
	m = set()

	for i in range(int(math.sqrt(c)) + 2):
		m.add(i * i)
		if c - i * i in m:
			return True
	return False

# 637 Average of Levels in Binary Tree
def averageOfLevels(root):
	res, queue = [], []

	if root:
		queue.append(root)

	while queue:
		res.append(sum(x.val for x in queue)/float(len(queue)))

		new_queue = []

		for p in queue:
			if p.left:
				new_queue.append(p.left)

			if p.right:
				new_queue.append(p.right)
		queue = new_queue
	return res


# 639 Decode Ways II
def numDecodings(s):
	

# 643 Maximum Average Subarray I
def findMaxAverage(nums, k):
	left, val, res = 0, 0.0, -float("inf")

    for i in range(len(nums)):

        if i - left + 1 < k:
            val += nums[i]

        elif i - left + 1 == k:
            val += nums[i]
            res = max(res, val / k)
            val, left = val - nums[left], left + 1

    return res


# 647 Palindromic Substrings
def countSubstrings(s):
	res, dp = 0, [[None for _ in range(len(s))] for _ in range(len(s))]

	for k in range(1, len(s) + 1):
		for i in range(len(s)-k+1):
			j = i + k - 1

			if k <= 2:
				dp[i][j] = (s[i] == s[j])

			else:
				dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])

			if dp[i][j]:
				res += 1
	return res


# 648 Replace words, 字典树操作很经典
class Trie(object):
	def __init__(self):
		self.root = {}
		self.res = []

	def insert(self, prefix):
		p = self.root

		for char in prefix:
			if char not in p:
				p[char] = {}
			p = p[char]
		p["END"] = True

	def search(self, word):
		p, cur = self.root, ""

		for i in range(len(word)):
			if word[i] not in p: # 不在前缀树中就break掉
				break

			cur += word[i]

			p = p[word[i]]

			if "END" in p: # 遇到 END 就结尾返回
				return cur
		return word


def replaceWords(dicts, sentence):
	t, res = Trie(), []
	for prefix in dicts:
		t.insert(prefix)

	wordlist = sentence.strip().split(" ")

	for word in wordlist:
		res.append(t.search(word))

	return " ".join(res)


