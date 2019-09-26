# 987 Vertical Order Traversal of a Binary Tree, 记住排序的时候key加符号会有不同的惊喜
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