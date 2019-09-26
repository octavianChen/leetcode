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






