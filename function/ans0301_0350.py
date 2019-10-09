# 322 Coin Change
def coinChange(coins, amount):
	dp = [i for i in range(amount + 1)]

	for i in range(1, amount + 1):
		for j in range(len(coins)):
			if coins[j] <= i:
				dp[i] = min(dp[i], dp[i - coins[j]] + 1)

	return -a if dp[amount] > amount else dp[amount]


# 326 Power of Three, 范围3的19次方
def isPowerOfThree(n):
	return n > 0 and 1162261467 % n == 0

def isPowerOfThree(n):
	if n <= 0:
		return False

	while n % 3 == 0:
		n /= 3

	return True if n == 1 else False

# 328 Odd Even Linkded List
def oddEvenList(head):
	dumpy1, dumpy2 = ListNode(0), ListNode(0)
	cur1, cur2, cur, cnt = dumpy1, dumpy2, head, 1

	while cur:
		if cnt % 2:
			cur1.next = cur
			cur1 = cur1.next
		else:
			cur2.next = cur
			cur2 = cur2.next
		cur = cur.next
		cnt += 1

	cur1.next = dumpy2.next
	cur2.next = None
	return dumpy1.next

# 329 Longest Increasing Path in a Matrix, 深度优先搜索加动态规划, dp[i][j] 记录的
# 是从 i, j 出发的最长路径的长度
def longestIncreasingPath(matrix):
	if not matrix or not matrix[0]:
		return 0

	m, n = len(matrix), len(matrix[0])

	def dfs(i, j):
		for dx, dy in zip([-1, 0, 0, 1], [0, 1, -1, 0]):
			r, c = i + dx, j + dy

			if 0 <= r < m and 0 <= c < n and matrix[r][c] > matrix[i][j]:
				if not dp[r][c]:
					dp[r][c] = dfs(r, c)

				dp[i][j] = max(dp[i][j], dp[r][c] + 1)
		dp[i][j] = max(dp[i][j], 1)
		return dp[i][j]

	dp = [[0 for _ in range(n)]for _ in range(m)]

	for i in range(m):
		for j in range(n):
			if not dp[i][j]:
				dp[i][j] = dfs(i, j)

	return max([max(x) for x in dp])


# 332 Reconstruct Itinerary
def findItinerary(tickets):
	
	

# 337 House Robber III
def rob(root):
	m = {}

	def dfs(node):
		if node is None:
			return 0

		if node in m:
			return m[node]

		val = 0

		if node.left:
			val += dfs(node.left.left) + dfs(node.left.right)

		if node.right:
			val += dfs(node.right.left) + dfs(node.right.right)

		val = max(val + node.val, dfs(node.left) + dfs(node.right))
		
		m[node] = val

		return val

	return dfs(root)


# 338 Counting Bits, 二进制中1的个数的巧妙运用
def countBits(num):
	res = [0 for _ in range(num+1)]

	for i in range(1, num+1):
		res[i] = res[i & (i-1)] + 1

	return res

# 342 Power of Four, 首先判断是否是2次方，再判断奇数位上是否是1
def isPowerOfFour(num):
	return num > 0 and not (n & (n-1)) and (n & 0x55555555)

# 344 Reverse String
def reverseString(s):
    if not s:
        return

    n = len(s)
    for i in range(n//2):
        s[i], s[n-1-i] = s[n-1-i], s[i]


# 345 Reverse Vowels of a String
def reverseVowels(s):
	i, j, m, s = 0, len(s)-1, set("aeiouAEIOU"), list(s)

	while i < j:
		if s[i] in m and s[j] in m:
			s[i], s[j] = s[j], s[i]
			i, j = i+1, j-1

		elif s[i] in m:
			j -= 1
		elif s[j] in m:
			i += 1
		else:
			i, j = i + 1, j-1
	return "".join(s)


# 347 Top K Frequent Elements
def topKFrequent(nums, k):
	m = {}

	for num in nums:
		m[num] = m.get(num, 0) + 1

	vec = sorted(m.items(), key=lambda x:x[1], reverse=True)

	return [x[0] for x in vec[:k]]

# 349 Intersection of Two Arrays
def intersection(nums1, nums2):
	nums1, nums2 = set(nums1), set(nums2)

	return nums1 & nums2

# 350 Intersection of Two Arrays II
def intersect(nums1, nums2):
	nums1, nums2 = sorted(nums1), sorted(nums2)

	i, j, res = len(nums1)-1, len(nums2)-1, []

	while i >= 0 and j >= 0:
		if nums1[i] == nums2[j]:
			res.append(nums1[i])
			i, j = i-1, j-1

		elif nums1[i] > nums2[j]:
			i -= 1

		else:
			j -= 1
	return res
