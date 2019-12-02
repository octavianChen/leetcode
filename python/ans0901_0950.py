class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 912 Sort an Array
def sortArray(nums):
	
	def help(nums, lo, hi):
		if hi < lo:
			return
		i = partition(nums, lo, hi)

		help(nums, lo, i-1)
		help(nums, i+1, hi)

	help(nums, 0, len(nums) - 1)
	return nums

def partition(nums, lo, hi):
	pivot = nums[hi]

	i = lo -1

	for j in range(lo, hi):
		if nums[j] <= pivot:
			i = i+1
			nums[i], nums[j] = nums[j], nums[i]

	i = i + 1
	nums[i], nums[hi] = nums[hi], nums[i]

	return i

# 917 Reverse Only Letters
def reverseOnlyLetters(S):
	s = list(S)
	i, j = 0, len(s) - 1

	while i < j:
		if not s[i].isalpha():
			i += 1
		elif not s[j].isalpha():
			j -= 1
		else:
			s[i], s[j] = s[j], s[i]
			i, j = i + 1, j - 1

	return "".join(s)


# 922 Sort Array By Parity II
def 

# 938 Range Sum of BST
def rangeSumBST(root, L, R):
	if root is None:
		return 0

	if L > R:
		return 0

	total = 0

	if root.val < L:
		total += self.rangeSumBST(root.right, L, R)

	elif root.val > R:
		totoal += self.rangeSumBST(root.left, L, R)

	else:
		total += root.val
		total += self.rangeSumBST(root.left, L, R)
		total += self.rangeSumBST(root.right, L, R)

	return total

