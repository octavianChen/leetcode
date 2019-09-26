# 856 Score of Parenthesis,栈的操作
def scoreOfParentheses(S):
    m, res = {}, 0
    for char in S:
        if char == "(":
            m.append(res)
            res = 0

        else:
            res = max(2 * res, 1) + st.pop()
    return res


# 861 Score After Flipping Matrix
def matrixScore(A):


# 863 All Nodes Distance K in Binary Tree
def distanceK(root, target, K):
    if root is None:
        return []

    res, parent, visited = [], {}, set()
    findparent(parent, root)
    dfs(target, K, parent, visited)
    return res

def findparent(parent, node):
    if node is None:
        return

    if node.left:
        parent[node.left] = node

    if node.right:
        parent[node.right] = node

    findparent(parent, node.left)
    findparent(parent, node.right)

def dfs(node, K, parent, visited):
    if node in visited:
        return

    visited.add(node)

    if K == 0:
        res.append(node.val)
        return

    if node.left:
        dfs(node.left, K-1, parent, visited)

    if node.right:
        dfs(node.right, K-1, parent, visited)

    if parent[node]:
        dfs(parent[node], K-1, parent, visited)
   

# 867 Transpose Matrix
def transpose(A):
    m, n = len(A), len(A[0])

    res = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(m):
        for j in range(n):
            res[j][i] = A[i][j]

    return res


# 872 Leaf-Similar Trees
def leafSimilar(root1, root2):
    vec1, vec2 = [], []

    getleaf(root1, vec1)
    getleft(root2, vec2)

    return vec1 == vec2

def getleaf(root, vec):
    if not root:
        return

    if not root.left and not root.right:
        vec.append(root.val)
        return

    getleaf(root.left, vec)
    getleaf(root.right, vec)


# 874 Walking Robot Simulation
def robotSim(commands, obstacles):
    dirs, x, y, obstacles, ans = [0, 1], 0, 0, set(map(tuple, obstacles)), 0

    for val in commands:
        if val == -1:
            dirs[0], dirs[1] = dirs[1], -dirs[0]

        elif val == -2:
            dirs[0], dirs[1] = -dirs[1], dirs[0]

        else:
            for _ in range(val):
                if (x + dirs[0], y + dirs[1]) not in obstacles:
                    x, y = x + dirs[0], y + dirs[1]
                ans = max(ans, x*x+y*y)
    return ans


# 876 Middle of the Linked List
def middleNode(head):
	fast, slow = head, head

	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next

	return slow

# 885 Spiral Matrix III
def spiralMatrixIII(R, C, r0, c0):
    

# 897 Increasing Order Search Tree
def increasingBST(root):
    st, p, root_new = [], root, None

    while st or p:
        while p:
            st.append(p)
            p = p.left

        p = st.pop()

        if root_new is None:
            root_new = TreeNode(p.val)
            cur = root_new

        else:
            cur.right = TreeNode(p.val)
            cur = cur.right

        p = p.right

    return root_new


# 889 Construct Binary Tree from Preorder and Postorder Traversal
def constructFromPrePost(self, pre, post):
    
    if not pre: return None
    root = TreeNode(pre[0])  # 创建当前根节点
    if len(pre) == 1: return root  # 如果只有一个值，直接返回

    i = post.index(pre[1]) + 1  # 获取当前跟节点，在 post 中的位置
    root.left = self.constructFromPrePost(pre[1:i+1], post[:i])  # 递归左子树
    root.right = self.constructFromPrePost(pre[i+1:], post[i:])  # 递归右子树
    return root


# 896 Monotonic Array
def isMonotonic(A):
    cnti, cntd = 0, 0

    for i in range(1, len(A)):
        if A[i] >= A[i-1]:
            cnti += 1

        if A[i] <= A[i-1]:
            cntd += 1

    return True if cnti == len(A) - 1 or cntd == len(A) - 1 else False




