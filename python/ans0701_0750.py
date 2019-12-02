class ListNode(object):
	def __init__(self, val):
		self.val = val
		self.next = None

# 701 Insert into a Binary Search Tree
def insertIntoBST(root, val):
	if not root:
		return TreeNode(val)

	if root.val > val:
		root.left = insertIntoBST(root.left, val)
	else:
		root.right = insertIntoBST(root.right, val)
	return root


# 704 Binary Search
def search(nums, target):
	lo, hi = 0, len(nums) - 1

	while lo <= hi:
		mid = (lo + hi) // 2

		if nums[mid] == target:
			return mid

		elif nums[mid] > target:
			hi = mid - 1
		else:
			lo = mid + 1
	return -1


# 707 Design Linked List
class MyLinkedList(object):
	def __init__(self):
		self.head = None
		self.size = 0

	def get(self, index):
		if index < 0 or index >= self.size:
			return -1
		cur = self.head

		for _ in range(index):
			cur = cur.next
		return cur.val

	def addAtHead(self, val):
		node = ListNode(val)
		node.next = self.head
		self.head = node
		self.size += 1

	def addAtTail(self, val):
		cur = self.head

		if not cur:
			self.head = ListNode(val)
		else:
			while cur.next:
				cur = cur.next
			cur.next = ListNode(val)
		self.size += 1

	def addAtIndex(self, index, val):
		if index > self.size:
			return
		elif index == 0:
			self.addAtHead(val)
		else:
			cur = self.head
			for _ in range(index-1):
				cur = cur.next
			tmp = cur.next
			cur.next = ListNode(val)
			cur.next.next = tmp
		self.size += 1

	def deleteAtIndex(self, index):
		if index < 0 or index >= self.size:
			return
		elif index == 0:
			self.head = self.head.next
		else:
			cur = self.head
			for _ in range(index-1):
				cur = cur.next

			tmp = cur.next.next
			cur.next = tmp
		self.size -= 1


# 709 To Lower Case
def toLowerCase(str):
	words = list(str)

	for i in range(len(words)):
		if 65 <= ord(words[i]) <= 90:
			words[i] = chr(ord(words[i]) + 32)
	return "".join(words)


# 718 Maximum Length of Repeated Subarray
def findLength(A, B):
	m, n, res = len(A), len(B), 0

	dp = [[0 for _ in range(n)] for _ in range(m)]

	for i in range(1, m):
		for j in range(1, n):
			if A[i-1] == B[j-1]:
				dp[i][j] = dp[i-1][j-1] + 1

	return max(max(row) for row in dp)

# 724 Find Pivot Index
def pivotIndex(nums):
	if not nums:
        return -1

    lval, rval = 0, sum(nums) - nums[0]
    
    if lval == rval:
        return 0

    for i in range(1, len(nums)):

        lval, rval = lval + nums[i-1], rval - nums[i]
        if lval == rval:
            return i

    return -1


# 725 Split Linked List in Parts
def splitListToParts(root, k):
	cnt, cur, ans = 0, root, []

	while cur:
		cnt += 1
		cur = cur.next

	val, res, cur = cnt // k, cnt % k, root

	for i in range(k):
		dumpy = ListNode(0)
		cur1 = dumpy
		val = val if i < res else val + 1
		for _ in range(val):
			cur1.next = cur
			cur1 = cur1.next
			cur = cur.next

		if cur1:
			cur1.next = None
		ans.append(dumpy.next)
		return ans

# 733 Flood Fill
def floddFill(image, sr, sc, newColor):
	if image[sr][sc] == newColor:
            return image

    def dfs(sr, sc, val):
        image[sr][sc] = newColor

        for dirs in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            r, c = sr + dirs[0], sc + dirs[1]

            if 0 <= r < len(image) and 0 <= c < len(image[0]) and image[r][c] == val:
                dfs(r, c, val)

    dfs(sr, sc, image[sr][sc])
    return image

# 739 Daily Temperature, 单调栈
def dailyTemperature(T):
	st, res = [], [0 for _ in range(len(T))]

	for i in range(len(T)):
		while st and T[i] > T[st[-1]]:
			t = st.pop()
			res[t] = i - t
		st.append(i)
	return res


# 743 Network Delay Time
def networkDelayTime(times, N, K):
	import Queue
	graph = [[] for _ in range(N+1)]

	for u, v, w in times:
		graph[u].append((v, w))

	res, m = -float("inf"), [float("inf") for _ in range(N+1)]
	q, m[K] = Queue.Queue(), 0
	q.put(K)

	while not q.empty():
		i = q.get()

		for j, d in graph[i]:
			if m[i] + d < m[j]:
				m[j] = m[i] + d
				q.put(j)

	for i in range(1, N+1):
		if m[i] == float("inf"):
			return -1
		res = max(res, m[i])
	return res

# 744 Find Smallest Letter Greater Than Target
def nextGreatestLetter(letters, target):
	lo, hi = 0, len(letters) - 1

    while lo < hi:
        mid = (lo+ hi + 1) // 2

        if letters[mid] <= target:
            lo = mid

        else:
            hi = mid - 1
    
    if letters[lo] > target:
        return letters[lo]

    return letters[(lo + 1) % (len(letters))]
    

# 745 Prefix and Suffix Search
def 


# 746 Min Cost Climbing Stairs
def mincostClimbingStairs(cost):
	cost.append(0)
	p2, p1 = cost[0], cost[1]

	for i in range(2, len(cost)):
		ans = min(p2, p1) + cost[i]
		p2, p1 = p1, ans

	return ans

# 747 Largest Number At Least Twice of Others
def dominantIndex(nums):
	f, s, fi, si = -float("inf"), -float("inf"), -1, -1

	for i in range(len(nums)):
		if nums[i] > f:
			s, f, si, fi = f, nums[i], fi, i

		elif s < nums[i] < f:
			s, si = nums[i], i

	if si == -1:
		return fi

	return fi if 2 * s <= f else -1