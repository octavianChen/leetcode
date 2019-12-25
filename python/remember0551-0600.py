# 559 Maximum Depth of N-ary Tree, 注意 map 的使用
def maxDepth(self, root):
	if not root:
		return 0

	res = 1
	for child in root.children:
		res = max(self.maxDepth(child) + 1, res)
	return res

def maxDepth(root):
	if root is None:
		return 0

	return max(map(self.maxDepth, root.children)) + 1


# 560 SubArray Sum Equals K
def subarraySum(nums, k):
	m, cnt, val = {0:1}, 0, 0

    for num in nums:
        val += num

        cnt += m.get(val - k, 0)

        m[val] = m.get(val, 0) + 1

    return cnt

# 563 Binary Tree Tilt
def findTilt(root):
	nt, ns, ts = help(root)

	return ts

def help(root):
	if not root:
		return 0, 0, 0

	lt, ls, lts = help(root.left)
	rt, rs, rts = help(root.right)

	titlt = abs(ns - ls)

	return titlt, ls + rs + root.val, titlt + lts + rts

# 590 N-ary Tree Postorder Traversal
def postorder(root):
	stack, ans = [], []

	if root:
		stack.append(root)

	while stack:
		node = stack.pop()
		ans.append(node.val)
		stack.extend(node.children)
	return ans[::-1]