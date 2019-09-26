# 856 Score of Parenthesis, 遇到左括号，压入当前得分，并清零，遇到右括号，内层得分
# 已经计算出来，乘以2并与1比较，再加上并行的值
def scoreOfParentheses(S):
    m, res = {}, 0
    for char in S:
        if char == "(":
            m.append(res)
            res = 0

        else:
            res = max(2 * res, 1) + st.pop()
    return res

# 863 All Nodes Distance K in Binary Tree, 所有的父节点可以用字典保存
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

# 889 Construct Binary Tree from Preorder and Postorder Traversal, 获取index的位置函数
# 然后用列表的切片
def constructFromPrePost(self, pre, post):
    
    if not pre: return None
    root = TreeNode(pre[0])  # 创建当前根节点
    if len(pre) == 1: return root  # 如果只有一个值，直接返回

    i = post.index(pre[1]) + 1  # 获取当前跟节点，在 post 中的位置
    root.left = self.constructFromPrePost(pre[1:i+1], post[:i])  # 递归左子树
    root.right = self.constructFromPrePost(pre[i+1:], post[i:])  # 递归右子树
    return root


# 894 All Possible Full Binary Trees
def allPossibleFBT(N):
	


