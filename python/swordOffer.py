# 3 二维数组的查找
def searchMatrix(matrix, target):
	if not matrix or not matrix[0]:
		return False

	m, n, i = len(matrix), len(matrix[0]), 0
	j = n-1

	while i < m and j >= 0:
		mid = matrix[i][j]

		if mid == target:
			return True

		elif mid > target:
			j -= 1

		else:
			i += 1
	return False

# 6 重建二叉树
def 


# 7 用两个栈实现一个队列
class MyQueue(object):
	def __init__(self):
		self.st1 = []
		self.st2 = []

	def push(self, x):
		self.st1.append(x)

	def pop(self):
		self.peek()
		return self.st2.pop()


	def peek(self):
		if self.st2:
			return self.st2[-1]

		for _ in len(self.st1):
			self.st2.append(self.st1.pop())

		return self.st2[-1]

	def empty(self):
		return self.st1 or self.st2

# 8 旋转数组的最小数字


# 10 二进制中1的个数
def numberOfOne(n):
	res = 0

	while n:
		res += 1
		n = n & (n-1)
	return res


# 16 反转链表
def reverseLinkedList(head):
	dumpy, cur = ListNode(0), head

	while cur:
		tmp, t = cur.next, dumpy.next
		dumpy.next = cur
		dumpy.next.next = t
		cur = tmp
	return dumpy.next


# 17 合并两个排序的链表
def mergeTwoLists(l1, l2):
	dumpy = ListNode(0)
	cur = dumpy

	while l1 and l2:
		if l1.val < l2.val:
			cur.next = l1
			l1 = l1.next

		else:
			cur.next = l2
			l2 = l2.next

		cur = cur.next

	cur.next = l1 if l1 else l2

	return dumpy.next


# 20 顺时针打印矩阵
def spiralMatrix(A):
	res = []

	if not A or not A[0]:
		return res

	tx, ty, bx, by = 0, 0, len(A)-1, len(A[0])-1

	while tx <= bx and ty <= by:
		if tx == bx:
			for i in range(ty, by+1):
				res.append(A[tx][i])
		elif ty == by:
			for i in range(tx, bx + 1):
				res.append(A[i][ty])

		else:
			for i in range(ty, by):
				res.append(A[tx][i])

			for i in range(tx, bx):
				res.append(A[i][by])

			for i in range(by, ty, -1):
				res.append(A[bx][i])

			for i in range(bx, tx, -1):
				res.append(A[i][ty])

		tx, ty, bx, by = tx + 1, ty + 1, bx - 1, by -1

	return res


# 23 从上到下打印二叉树
def levelOrderTraversal(root):
	res, st = [], []

	if root is None:
		return res

	st.append(root)

	while st:
		new_st = []

		for node in st:
			res.append(node.val)

			if node.left:
				new_st.append(node.left)

			if node.right:
				new_st.append(node.right)
		st = new_st

	return res


# 25 二叉树中和为某一值的路径
def pathSum(root, target):
	res, out = [], []

	def dfs(p, target):
		if p is None:
			return 

		out.append(p.val)

		if p.left is None and p.right is None and p.val == target:
			res.append(out[:])

		dfs(p.left, target-p.val)
		dfs(p.right, target-p.val)

		out.pop()

	dfs(root, target)

	return res



# 26 复杂链表的复制
def copyRandomList(head):
	dumpy = Node(0, None, None)
	cur, cur1, m = head, dumpy, {}

	while cur:
		cur1.next = Node(cur.val, None, None)
		m[cur] = cur1.next
		cur, cur1 = cur.next, cur1.next

	cur, cur1 = head, dumpy.next

	while cur:
		if cur.random in m:
			cur1.random = m[cur.random]
		cur, cur1 = cur.next, cur1.next

	return dumpy.next

# 无向图的复制
def cloneGraph(node):
	m = {}

	def dfs(t):
		if t is None:
			return None

		if t in m:
			return m[t]

		clone = Node(t.val, [])
		m[t] = clone

		for node in t.neighbors:
			clone.neighbors.append(dfs(node))
		return clone

	return dfs(node)


# 29 数组中出现次数超过一半的数字
def majorityElement(nums):
	candi, times = 0, 0

	for num in nums:
		if num == candi:
			times += 1

		else:
			if times > 0:
				times -= 1
			else:
				candi, times = num, times + 1
	return candi


# 31 连续子数组最大和
def maxSubarray(nums):
	res, total = -float("inf"), 0

	for num in nums:
		total = max(total + num, num)
		res = max(res, total)

	return res


# 33 把数组排成最小的数
def (numbers):
 	res = "".join(sorted(list(map(str, numbers)), cmp=lambda x, y:cmp(x+y, y+x))).lstrip("0")
 	return res if res else "0"

# 34 丑数
def nthUglyNumber(n):
	res, i2, i3, i5 = [1], 0, 0, 0

	while len(res) < n:
		m2, m3, m5 = res[i2] * 2, res[i3] * 3, res[i5] * 5

		candi = min(m2, min(m3, m5))

		if candi == m2:
			i2 += 1

		if candi == m3:
			i3 += 1

		if candi == m5:
			i5 += 1

		res.append(candi)

	return res[-1]

# 35 第一个只出现一次的字符
def firstUniqChar(s):
	m = {}

	for char in s:
		m[char] = m.get(char, 0) + 1

	for i in range(len(s)):
		if m[s[i]] == 1:
			return i
	return -1

# 37 两个链表的第一个公共节点
def getIntersectionNode(headA, headB):
	if headA is None or headB is None:
		return None

	a, b = headA, headB

	while a != b:
		a = a.next if a else headA
		b = b.next if b else headB

	return a

# 38 数字在排序数组中出现的次数
def 


# 39 二叉树的深度
def getHeight(root):
	if root is None:
		return 0

	lh = getHeight(root.left)
	lr = getHeight(root.right)

	return max(lh, lr) + 1


# 52 构建乘积数组
def productExceptSelf(nums):
	res, right = [1 for _ in range(len(nums))], 1

	for i in range(1, len(nums)):
		res[i] = res[i-1] * nums[i-1]

	for i in range(len(nums)-1, 0, -1):
		res[i] = res[i] * right
		right = right * nums[i]

	return res


# 53 正则表达式匹配, 初始条件与匹配零次以上的条件最复杂
def isMatch(s, p):
	dp = [[False for _ in range(len(p) + 1)] for _ in range(len(s)+1)]
	dp[0][0] = True

	for i in range(2, len(p) + 1):
		if p[i-1] == "*":
			dp[0][i] = dp[0][i-2]

	for i in range(1, len(s)+1):
		for j in range(1, len(p) + 1):
			if s[i-1] == p[j-1]:
				dp[i][j] = dp[i-1][j-1]

			elif p[j-1] == ".":
				dp[i][j] = dp[i-1][j-1]

			elif p[j-1] == "*": # dp[i][j-1]匹配一次dp[i][j-2] 匹配零次, 
				dp[i][j] = dp[i][j-1] or dp[i][j-2] or (s[i-1] == p[j-2] or p[j-2] == ".")

	return dp[len(s)][len(p)]


# 56 链表中环的入口结点
def detectCycle(head):
	slow, fast = head, head

	while fast and fast.next:
		slow, fast = slow.next, fast.next.next

		if slow == fast:
			break

	if fast is None or fast.next is None:
		return None

	slow = head

	while slow != fast:
		slow, fast = slow.next, fast.next

	return slow


# 57 删除链表中重复的节点
def deleteDuplicates(head):
	if head is None:
		return None

	dumpy = ListNode(0)

	dumpy.next = head

	cur1, cur2 = dumpy.next, head.next

	while cur2:
		if cur2.val != cur1.val:
			cur1.next = cur2
			cur1 = cur1.next
		cur2 = cur2.next

	cur1.next = None
	return dumpy.next


# 60 把二叉树打印成多行
def levelOrder(root):
	res, queue = [], []

	if root is None:
		return res

	queue.append(root)

	while queue:
		out, new_queue = [], []

		for node in queue:
			out.append(node.val)

			if node.left:
				new_queue.append(node.left)

			if node.right:
				new_queue.append(node.right)

			res.append(out)
		queue = new_queue

	return res

# 61 按之字形顺序打印二叉树
def zigZag(root):
	res, queue, level = [], [], 0

	if root is None:
		return res

	queue.append(root)

	while queue:
		new_queue, out = [], []

		for node in queue:
			out.append(node.val)

			if node.left:
				new_queue.append(node.left)

			if node.right:
				new_queue.append(node.right)

		res.append(out[::-1]) if level % 2 else res.append(out)
		queue = new_queue

	return res


# 62 序列化二叉树
class Codec:
	def serialize(self, root):
		res = []

		if root is None:
			res.append("#")
			return res

		res.append(str(root.val))
		self.serialize(root.left)
		self.serialize(root.right)

		return ",".join(res)

	def deserialize(self, data):
		data = iter(data.split(","))

		def build():
			val = next(data)

			if val == "#":
				return None

			root = TreeNode(val)

			root.left = self.deserialize(data)

			root.right = self.deserialize(data)

			return root

		return build()

# 63 二叉搜索树的第 k 个节点
def kthSmallest(root):
	st, p, cnt = [], root, 1

	while st or p:
		while p:
			st.append(p)
			p = p.left

		p = st.pop()

		if cnt == k:
			return p.val

		p, cnt = p.right, cnt + 1


# Kmeans 算法
def kmeans(data, k):
	m = len(data) # 数据点个数
	n = len(data[0]) # 数据维度
	cluster_center = np.zeros((k, n)) # 每行表示一个聚类中心

	init_list = np.random.randomint(low=0, high=m, size=k)

	for index, j in enumerate(init_list):
		cluster_center[index] = data[j][:] # 随机选取 k 个点做聚类中心

	# 聚类
	cluster = np.zeros(m, dtype=np.int) - 1

	cc = np.zeros((k, n)) # 下一轮聚类中心

	c_number = np.zeros(k) # 每个聚类中心上样本的数目


	for times in range(1000):
		for i in range(m):
			c = nearest(data[i], cluster_center) # 计算每个数据点和所有聚类中心的距离，返回属于哪个聚类中心
			cluster[i] = c # 第 i 个点属于 第 c 个聚类中心
			c_number[c] += 1 # 第 c 个聚类中心的个数加一
			cc[c] += data[i]

		for i in range(k):
			cluster_center[i] = cc[c] / c_number[i]

		cc.flat, c_number.flat = 0, 0

	return cluster


def nearest(data, cluster_center):
	nearest_center_index = 0
	distance = float("inf")

	for index, cluster_center_one in enumerate(cluster_center):
		dis = np.sum((data - center) ** 2)

		if dis < distance:
			nearest_center_index = index
			distance = dis

	return nearest_center_index


# 两个有序数组找中位数
def topK(A, s1, e1, B, s2, e2, k):
	l1, l2 = e1- s1 + 1, e2 - s2 + 1
	i, j = s1 + min(k//2, l1) - 1, s2 + min(k//2, l2) - 1

	if l1 == 0 and l2 > 0:
		return B[s2 + k - 1]

	if l2 == 0 and l1 > 0:
		return A[s1 + k - 1]

	if k == 1:
		return min(A[s1], B[s2])

	if A[i] > B[j]:
		return topK(A, s1, e1, B, j+1, e2, k-(j-s2+1))
	else:
		return topK(A, i+1, e1, B, s2, e2, k-(i-s1+1))


def findMedianSortedArrays(nums1, nums2):
	if len(num1) > len(nums2):
		nums1, nums2 = nums2, nums1
	m, n = len(nums1), len(nums2)

	if (m + n) % 2 == 0:
		return (topK(nums1, 0, m-1, nums2, 0, n-1, (m+n)/2) + topK(nums1, 0, m-1, nums2, 0, n-1, (m+n)/2+1))/2.0
	else:
		return topK(nums1, 0, m-1, nums2, 0, n-1, (m+n+1)/2)

# KNN tensorflow实现
import tensorflow as tf
import numpy as np

# Build Graph
tr = tf.placeholder(tf.float32, [None, 784])
te = tf.placeholder(tf.float32, [784])

distance = tf.reduce_sum(tf.square(tf.substract(te, tr)), axis=1)
pred = tf.nn.top_k(distance, k)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for i in range(len(testdata)):
		nn_index = sess.run(pred, feed_dict={tr:traindata, te:testdata[i, :]})


# 链表的快排
def sortList(head):
	if head is None or head.next is None:
		return head

	small, large, cur = ListNode(0), ListNode(0), head.next
	sp, lp = small, large

	while cur:
		if cur.val <= head.val:
			sp.next = cur
			sp = sp.next
		else:
			lp.next = cur
			lp = lp.next
		cur = cur.next

	sp.next, lp.next = None, None

	small, large = self.sortList(small.next), self.sortList(large.next)

	sp = small

	if sp:
		while sp.next:
			sp = sp.next
		sp.next = head
		head.next = large
		return small
	else:
		head.next = large
		return head

def maxPooling(nums, k):
	from collections import deque
	q = deque()

	for i, x in enumerate(nums):
		if q and i - q[0] >= k:
			q.popleft()

		while q and nums[q[-1]] <= x:
			q.pop()

		q.append(i)

		if i >= k - 1:
			res.append(nums[q[0]])
	return res
	