class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 551 Student Attendance Record I
def checkRecord(s):
	conuntA, j = 0, False

	for i in range(len(s)):
		if s[i] == "A":
			conuntA += 1

		if s[i] == "L":
			j = i
			while j < len(s) and s[j] == "L":
				j += 1

		 	if j - i >= 2:
		 		return False

	return conuntA == 0


# 557 Reverse Words in a String III
def reverseWords(s):
	words = [word[::-1] for word in s.strip().split()]
	return " ".join(words)

# 559 Maximum Depth of N-ary Tree
def maxDepth(root):
	if not root:
		return 0

	res = 1
	for child in root.children:
		res = max(self.maxDepth(child) + 1, res)
	return res

# 561 Array Partition I
def arrayPairSum(nums):
	nums, res = sorted(nums), 0

	for i in range(0, len(nums), 2):
		res += nums[i]

	return res

# 563 Binary Tree Tilt
def findTilt(root):
	if not root:
		return 0

	val1 = findTilt(root.left)
	val2 = findTilt(root.right)

	val = abs(sumNode(root.left) - sumNode(root.right))

	return val1 + val2 + val

def sumNode(root):
	if not root:
		return 0

	res = 0

	res += sumNode(root.left)
	res += sumNode(root.right)
	res += root.val
	return res
		
# 566 Reshape the Matrix
def matrixReshape(nums, r, c):
	m, n, res = len(nums), len(nums[0]), [[0 for _ in range(c)] for _ in range(r)]

	if m * n != r * c:
		return nums

	for i in range(m):
		for j in range(n):
			pos = i * n + j
			kr, kc = pos // c, pos % c
			res[kr][kc] = nums[i][j]

	return res


# 572 Subtree of Another Tree
def isSubtree(s, t):
	if t is None:
		return True

	if s is None:
		return False

	if s.val == t.val and sameTree(s, t):
		return True

	return isSubtree(s.left, t) or isSubtree(s.right, t)
	

def sameTree(s, t):
	if not s and not t:
		return True

	if not s or not t:
		return False

	if s.val != t.val:
		return False

	return sameTree(s.left, t.left) and sameTree(s.right, t.right)


# 581 Shortest Unsorted Continuous Subarray
def findUnsortedSubarray(nums):
	nums1 = sorted(nums)

	for i in range(len(nums)):
		if nums[i] != nums1[i]:
			break

	for j in range(len(nums) - 1, -1, -1):
		if nums[j] != nums1[j]:
			break

	return j - i + 1


# 589 N-ary Tree Preorder Traversal
def preorder(root):
	res, st = [], []

	if root:
		st.append(root)

	while st:
		p = st.pop()
		res.append(p.val)

		for i in range(len(p.children)-1, -1, -1):
			st.append(p.children[i])
	return res

# 590 N-ary Tree Postorder Traversal
def postorder(root):
	m = []
	help(root, m)
	return m

def help(root, m):
	if not root:
		return m

	for node in root.children:
		help(node, m)

	m.append(root.val)
