class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 516 Longest Palindromic Subsequence
def longestPalindromeSubseq(s):
    if s == s[::-1]:
        return len(s)
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


# 518 Coin Change 2 # dp[i][j] 表示 coins 中前 i 个硬币组成 j 块钱的方法, 分不用当前
# 硬币和用当前硬币两种，注意通项公式
def change(amount, coins):
    dp = [[0 for _ in range(amount+1)] for _ in range(len(coins)+1)]
    dp[0][0] = 1

    for i in range(1, len(coins)+1):
        dp[i][0] = 1
        for j in range(1, amount+1):
            cnt = dp[i][j-coins[i-1]] if j >= coins[i-1] else 0
            dp[i][j] = dp[i-1][j] + cnt

    return dp[len(coins)][amount]

# 519 Random Flip the Matrix


# 521 Longest Uncommon Subsequence I, 字符串若相等，必定没有LUS，不等，自然是最长的那个
def findLUSlength(a, b):
    return -1 if a == b else max(len(a), len(b))

# 522 Longest Uncommon Subsequence II


# 530 Minimum Absolute Difference in BST, 用闭包的函数解，非常有用
def getMinimumDifference(root):
	self.ans, self.last = float("inf"), None

	def help(root):
		if not root:
			return

		help(root.left)
		if self.last:
			self.ans = min(self.ans, root.val - self.last)
		
		self.last = root
		help(root.right)

	help(root)
	return self.ans

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

# 543 Diameter of Binary Tree, 计算时候顺便计算一下树高
def diameterOfBinaryTree(self, root):
    self.res = 0

    def help(node):
        if node is None:
            return 0
        lv = help(node.left)
        rv = help(node.right)

        self.res = max(self.res, lv + rv)

        return max(lv, rv) + 1

    help(root)
    return self.res

# 547 Friend Circles, 注意一定要有节点访问过的标志，并且注意有边的使用, 还有并查集的使用
def findCircleNum(M):
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

def findCircleNum(M):
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

