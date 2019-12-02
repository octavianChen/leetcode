# 756 


# 766 Toeplitz Matrix
def isToeplitzMatrix(matrix):
	m, n = len(matrix), len(matrix[0])

	for j in range(n):
		candi, r, c = matrix[0][j], 0, j

		while r < m and c < n:
			if matrix[r][c] != candi:
				return False

			r, c = r + 1, c + 1

	for i in range(1, m):
		candi, r, c = matrix[i][0], i , 0

		while r < m and c < n:
			if matrix[r][c] != candi:
				return False

			r, c = r + 1, c + 1

	return True

# 767 Reorganize String
def reorganizeString(S):
	

# 783 Minimum Distance Between BST Nodes
def minDiffInBST(self, root):
	self.res, self.last = float("inf"), None

	def help(root):
		if root is None:
			return

		help(root.left)

		if self.last:
			self.res = min(self.res, abs(root.val - self.last.val))
		self.last = root
		help(root.right)

	help(root)
	return self.res