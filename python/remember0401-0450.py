# 410 Split Array Largest Sum, dp[i][j]将数组中前 j 个数字分成 i 组所能得到的子数组
# 最大值，要算出子数组和。同时注意k的取值为什么是这样的
def splitArray(nums, m):
	 n = len(nums)
	 sums, dp = [0 for _ in range(n+1)], [[float("inf") for _ in range(n+1)] for _ in range(m+1)]
	 dp[0][0] = 0

	 for i in range(1, n+1):
	 	sums[i] = sums[i-1] + nums[i-1]

	 for i in range(1, m+1): # i个子数组, 最多划分成m个
	 	for j in range(1, n + 1): # 前 j 个数
	 		for k in range(i-1, j): # 前 k 个数
	 			val = max(dp[i-1][k], sums[j] - sums[k])
	 			dp[i][j] = min(dp[i][j], val)

	 return dp[m][n]


# 416 Partition Equal Subset Sum, 注意状态转移方程
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


# 417 Pacific Atlantic Water Flow, 能不能流到边上，就从边上开始搜，两边最后取集合
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

# 437 Path Sum III, 先序遍历二叉树，用一个cursum记录一条根节点到当前节点的路径的和
# 看是否相等；不等查看子路径，每次查看一个节点
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

		# 除去root节点的子路径的和
		for i in range(len(out)-1):
			t -= out[i].val

			if t == vals:
				self.res += 1

		dfs(node.left, vals, cursum, out)
		dfs(node.right, vals, cursum, out)
		out.pop()

	dfs(root, sumval, 0, out)
	return self.res


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

def findDuplicates(nums): # 这种方法也很好
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

# 445 Add Two Numbers II, 用栈将数字翻转过来
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