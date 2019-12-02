class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 929 Unique Email Addresses
def numUniqueEmails(emails):
	m = set()

	for email in emails:
		p1, p2 = email.strip().split("@")

		p1 = p1.replace(".", "").split("+")[0]

		m.add(p1 + "@" + p2)

	return len(m)


# 958 Check Completeness of a Binary Tree
def isCompleteTree(root):
	if not root:
		return True
	queue. leaf = [], False
	queue.append(root)

	while queue:
		new_queue = []

		for p in queue:
			p = queue.pop()

			if (not p.left and p.right) or (leaf and (p.left and not p.right)):
				return False
			if p.left:
				new_queue.append(p.left)
			if p.right:
				new_queue.append(p.right)
			else:
				leaf = True 
		queue = new_queue
	return True

# 965 Univalued Binary Tree
def isUnivalTree(root):
	if not root:
		return True

	st, candi, p = [], root.val, root

	while st or p:
		while p:
			st.append(p)
			p = p.left

		p = st.pop()

		if p.val != candi:
			return False

		p = p.right
	return True


# 974 Subarray Sums Divisible by K
def subarraysDivByK(A, K):
	m, val, cnt = {0:1}, 0, 0

	for num in A:
		val += num

		cnt += m.get(val % K, 0)

		m[val % K] = m.get(val % K, 0) + 1

	return cnt

# 977 Squares of a Sorted Array
def sortedSquare(A):
	return sorted(x*x for x in A)

# 978 Longest Turbulent Subarray
def maxTurbulenceSize(A):
	

# 993 Cousins in Binary Tree
def isCousins(root, x, y):
	queue = []

	if not root or root.val == x or root.val == y:
		return False

	queue.append((root, None))

	while queue:
		new_queue, pre1, pre2 = [], None, None

		for node, pre in queue:

			if node.val == x:
				pre1 = pre

			if node.val == y:
				pre2 = pre

			if node.left:
				new_queue.append((node.left, node))

			if node.right:
				new_queue.append((node.right, node))

		if pre1 and pre2:
			return pre1 != pre2
		elif pre1 or pre2:
			return False

		queue = new_queue
	return False

# 987 Vertical Order Traversal of a Binary Tree
def verticalTraversal(root):
	m, x, y = {}, 0, 0

	dfs(root, x, y, m)

	vec = [x[1] for x in sorted(m.items(), key=lambda x:x[0])]

	return [[y[1] for y in x]for x in [sorted(x, key=lambda x:(-x[0], x[1])) for x in vec]]

def dfs(root, x, y, m):
	if root is None:
		return

	if x not in m:
		m[x] = []
	m[x].append((y, root.val))

	dfs(root.left, x-1, y-1, m):
	dfs(root.right, x+1, y-1, m)


# 989 Add to Array-Form of Integer
def addToArrayForm(A, K):
	dig = 0

	for num in A:
		dig = dig * 10 + num

	dig = dig + K

	return map(int, list(str(dig))) 

# 998 Maximum Binary Tree II
def insertIntoMaxTree(root, val):
	if not root or root.val < val:
		cur = TreeNode(val)
		cur.left = root
		return cur

	root.right = insertIntoMaxTree(root.right, val)
	return root





