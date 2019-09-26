class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 1002 Find Common Charaters
def comonChars(A):
    

# 1008 Construct Binary Search Tree from Preorder Traversal
def bstFromPreorder(preorder):
	inorder = sorted(preorder)
    return help(preorder, 0, len(preorder)-1, inorder, 0, len(inorder)-1)

def help(preorder, pl, pr, inorder, il, ir):
	if pl > pr:
		return None

	root, index = TreeNode(preorder[pl]), 0

	for i in range(il, ir + 1):
		if inorder[i] == preorder[pl]:
			index = i
			break
	root.left = help(preorder, pl + 1, index + pl - il, inorder, il, index - 1)
	root.right = help(preorder, pr+index+1-ir, pr, inorder, index + 1, ir)
	return root

# 1020 Number of Enclaves
def numEnclaves(self, A):
    if not A or not A[0]:
        return 0
    
    row, col, res = len(A), len(A[0]), 0
    
    def dfs(i, j):
        A[i][j] = 0
        
        direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        
        for dirs in direction:
            r, c = i + dirs[0], j + dirs[1]
            
            if 0 <= r < row and 0 <= c < col and A[r][c] == 1:
                dfs(r, c)
    
    for i in range(row):
        for j in range(col):
            if (i == 0 or i == row - 1 or j == 0 or j == col - 1) and A[i][j] == 1:
                dfs(i, j)
    
    for i in range(1, row-1):
        for j in range(1, col-1):
            if A[i][j] == 1:
                res += 1
    return res

# 1022 Sum of Root To Leaf Binary Numbers
class Solution(object):
    def sumRootToLeaf(self, root):
        self.res = 0

        def help(root, ans):
        	if not root:
        		return

        	curr = (ans << 1) + root.val

        	if not root.left and not root.right:
        		self.res += curr
        		return

        	help(root.left, curr)
        	help(root.right, curr)

        help(root, 0)
        return self.res

# 1026 Maximum Difference Between Node and Ancestor
def maxAncestorDiff(root):
    self.res = -float("inf")

    def dfs(node):
        if node is None:
            return float("inf"), -float("inf")

        ll, lr = dfs(node.left)
        rl, rr = dfs(node.right)

        l, r = min(ll, rl), max(lr, rr)

        if l == float("inf") and r == -float("inf"):
            return node.val, node.val

        elif l == float("inf"):
            self.res = max(self.res, abs(node.val - r))
            return min(node.val, r), max(node.val, r)

        elif r == float("inf"):
            self.res = max(self.res, abs(node.val - l))
            return min(node.val, l), max(node.val, l)

        else:
            self.res = max(self.res, abs(node.val - l), abs(node.val - r))
            return min(node.val, l), max(node.val, r) 


    dfs(root)
    return self.res


# 1028 Recover a Tree From Preorder Traversal
def recoverFeomPreorder(S):
    stack, i = [], 0
    while i < len(S):
        level, val = 0, ""
        while i < len(S) and S[i] == '-':
            level, i = level + 1, i + 1

        while i < len(S) and S[i] != '-':
            val, i = val + S[i], i + 1

        while len(stack) > level:
            stack.pop()

        node = TreeNode(val)

        if stack and stack[-1].left is None:
            stack[-1].left = node
        elif stack:
            stack[-1].right = node

        stack.append(node)
    return stack[0]

# 1030 Matrix Cells in Distance Order
def allCellsDistOrder(R, C, r0, c0):
    vec = []

    for i in range(R):
        for j in range(C):
            dis = abs(i-r0) + abs(j-c0)
            vec.append(([i, j], dis))

    vec = sorted(vec, key=lambda x:x[1])
    return [x[0] for x in vec]

# 1143 Longest Common Subsequence
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)

    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
