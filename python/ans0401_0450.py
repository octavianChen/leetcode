class Node(object):
	def __init__(self, val, prev, next, child):
		self.val = val
		self.prev = prev
		self.next = next
		self.child = child

# 404 Sum of Left Leaves
def sumOfLeftLeaves(root):
	total = [0]
	help(root, total)
	return total[0]

def help(root, total):
	if not root:
		return

	if root.left and not root.left.left and not root.left.right:
		total += root.left.val

	help(root.left, total)
	help(root.right, total)
	

# 409 Longest Palindrome
def longestPalindrome(s):
	m, cnt = {}, 0

	for c in s:
		m[c] = m.get(c, 0) + 1

	for key, val in m.items():
		if val % 2 == 1:
			cnt += 1
	return len(s) - cnt + 1 if cnt else len(s)


# 414 Third Maximum Number
def thirdMax(nums):
	f = s = t = -float("inf")

	for num in nums:
		if num > f:
			s , t, f = f, s, num

		elif s < num < f:
			t, s = s, num

		elif t < num < s:
			t = num

	return t if t > -float("inf") else f


# 415 Add Strings
def addStrings(num1, num2):
	return help(num1[::-1], num2[::-1])

def help(a, b):
	m, mode, res = max(len(a), len(b)), 0, ""

	for i in range(m):
		val1 = int(a[i]) if i < len(a) else 0
		val2 = int(b[i]) if i < len(b) else 0

		res += str((val + val2 + mode) % 10)

		mode = (val1 + val2 + mode) // 10

	if mode:
		res += str(mode)
	return res[::-1]

# 416 Partition Equal Subset Sum
def canPartition(nums):
	val = sum(nums)

	if val % 2 == 1:
		return False

	dp = [False for _ in range(val//2 + 1)]

	dp[0] = True

	for num in nums:
		for i in range(val // 2, num - 1, -1):
			dp[i] = dp[i] or dp[i-num]

	return dp[val//2]

# 417 Pacific Atlantic Water Flow
def pacificAtlantic(matrix):
	self.res = []

	if not matrix or not matrix[0]:
		return self.res

	row, col = len(matrix), len(matrix[0])

	pac, alt = [[False for _ in range(col)] for _ in range(row)], [[False for _ in range(col)] for _ in range(row)]

	def dfs(i, j, visited, pre):

		if matrix[i][j] < pre:
			return

		direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

		visited[i][j] = True

		for dirs in direction:
			r, c = i + dirs[0], j + dirs[1]

			if 0 <= r < row and 0 <= c < col and not visited[r][c]:
				dfs(r, c, visited, matrix[i][j])

	for i in range(row):
		dfs(i, 0, pac, -float("inf"))
		dfs(i, col-1, alt, -float("inf"))

	for i in range(col):
		dfs(0, i, pac, -float("inf"))
		dfs(row-1, i, alt, -float("inf"))

	for i in range(row):
		for j in range(col):
			if pac[i][j] and alt[i][j]:
				self.res.append([i, j])

	return self.res


# 429 N-ary Tree Level Order Traversal
def levelOrder(root):
	import Queue
	q, out, res = Queue.Queue(), [], []
	if root:
		q.put(root)

	while not q.empty():
		lev = []

		for _ in range(q.qsize()):
			p = q.get()
			lev.append(p.val)

			if p.children:
				q.put(p1 for p1 in p.children)

		res.append(lev)
	return res 


# 430 Flatten a Multilevel Doubly Linkded List
def flatten(head):
	cur = head

	while cur:
		if cur.child:
			tmp = cur.next
			cur.child = flatten(cur.child)
			cur1 = cur.child

			while cur1.next:
				cur1 = cur1.next

			cur.next = cur.child
			cur.next.prev = cur
			cur.child = None
			cur1.next = tmp

			if tmp:
				tmp.prev = cur1
		cur = cur.next
	return head

def flatten(head):
	if not head:
		return head

	cur = dumpy = Node(-100, None, None, None)
	stk = [head]

	while stk:
		cand = stk.pop()
		cur.next = cand
		cand.prev = cur

		while cur.next or cur.child:
			if cur.child:
				if cur.next:
					stk.append(cur.next)
				cur.next = cur.child
				cur.next.prev = cur
				cur.child = None
			cur = cur.next
	head.prev = None
	return head


# 437 Path Sum III
def pathSum(root, sumval):
	self.num_path, out = 0, []

	def dfs(node, vals, cursum, out):
		if node is None:
			return

		out.append(node)
		cursum += node.val

		if cursum == vals:
			self.res += 1

		t = cursum

		for i in range(len(out)-1):
			t -= out[i].val

			if t == vals:
				self.res += 1

		dfs(node.left, vals, cursum, out)
		dfs(node.right, vals, cursum, out)
		out.pop()

	dfs(root, sumval, 0, out)
	return self.res


# 438 Find All Anagrams in a String
def findAnagrams(s, p):
    m, left, right, n, res = {}, 0, 0, {}, []
    for c in p:
        m[c] = m.get(c, 0) + 1

    for i in range(len(s)):
        n[s[i]] = n.get(s[i], 0) + 1

        if i - left + 1 == len(p):
            if n == m:
                res.append(left)

            n[s[left]] -= 1

            if n[s[left]] == 0:
                n.pop(s[left])

            left += 1
    return res


# 442 Find All Duplicates in an Array
def findDuplicates(nums):
	res = []

	for i in range(len(nums)):
		while nums[i] and nums[i] != i + 1: # 需要将 nums[i] 移动到 nums[nums[i] - 1]
			n = nums[i] # 一定要把该值取出来，不要在列表上随便修改
			if nums[i] == nums[n - 1]: # 此时位置上有人
				res.append(n)
				nums[i] = 0 # 跳出循环
			else: # 否则互换位置
				nums[i], nums[n - 1] = nums[n - 1], nums[i]
	return res

def findDuplicates(nums):
	res, i = [], 0

	while i < len(nums):
		n = nums[i]

		if n != nums[n - 1]:
			nums[i], nums[n-1] = nums[n-1], nums[i]
			i - = 1

		i += 1

	for i in range(len(nums)):
		if nums[i] != i + 1:
			res.append(nums[i])
	return res


# 443 String Compression
def compress(chars):
	i, j, cnt, k = 0, 0, 0, 0

	chars.append("NED")

	while  j < len(chars):
		if chars[i] != chars[j]:
			chars[k] = chars[i]
			k += 1
			if cnt >= 2:
				chars[k: k + len(str(cnt))] = str(cnt)
				k = k + len(str(cnt))

			i, cnt = j, 0

		else:
			cnt, j = cnt + 1, j + 1

	return k
		

# 445 Add Two Numbers II
def addTwoNumbers(l1, l2):
	s1, s2, dumpy, mode = [], [], ListNode(0), 0

	while l1 or l2:
		if l1:
			s1.append(l1.val)
			l1 = l1.next

		if l2:
			s2.append(l2.val)
			l2 = l2.next

	while s1 or s2:
		val1 = s1[-1] if s1 else 0
		if s1:
			s1.pop()
		val2 = s2[-1] if s2 else 0
		if s2:
			s2.pop()

		val, mode = (val1 + val2 + mode) % 10, (val1 + val2 + mode) // 10

		tmp = dumpy.next
		dumpy.next = ListNode(val)
		dumpy.next.next = tmp

	if mode:
		tmp = dumpy.next
		dumpy.next = ListNode(mode)
		dumpy.next.next = tmp

	return dumpy.next

# 448 Find All Numbers Disappeared in an Array
def findDisappearedNumbers(nums):
	res, i = [], 0

	while i < len(nums):
		n = nums[i]

		if n != nums[n-1]:
			nums[i], nums[n-1] = nums[n-1], nums[i]
			i -= 1
		i += 1

	for i in range(len(nums)):
		if nums[i] != i + 1:
			res.append(i+1)
	return res

# 449 Serialize and Deseriallize BST
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

		return vals

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

# 450 Delete Node in BST
def deleteNode(root, key):
	if not root:
		return root

	if root.val == key:
		if not root.left or not root.right:
			return root.left if root.left else root.right

		cur = root.right

		while cur.left:
			cur = cur.left

		root.val, cur.val = cur.val, root.val

		root.right = deleteNode(root.right, key)

	elif root.val < key:
		root.right = deleteNode(root.right, key)
	else:
		root.left = deleteNode(root.left, key)

	return root








	


