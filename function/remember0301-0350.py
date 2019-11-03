# 313 Super Ugly Number, 延续 264 Ugly Number II 的方针
def nthSuperUglyNumber(n, primes):
    res, index = [1], [1 for _ in range(len(primes))]

    while len(res) < n:
        candidates = [res[index[i]] * primes[i] for i in range(len(primes))]

        minval = min(candidates)
        for i in range(len(candidates)):
            if candidates[i] == minval:
                index[i] += 1

        res.append(minval)

    return res[-1]


# 322 Coin Change
def coinChange(coins, amount):
    dp = [amount + 1 for i in range(amount + 1)]
    dp[0] = 0

    for i in range(1, amount + 1):
        for j in range(len(coins)):
            if coins[j] <= i:
                dp[i] = min(dp[i], dp[i - coins[j]] + 1)

    return -a if dp[amount] > amount else dp[amount]


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

# 337 House Robber III, 一定要注意返回值究竟代表什么意义
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