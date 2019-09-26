# 1104 Path In Zigzag Labelled Binary Tree
def pathInZigZagPath(label):
	parent = {}


# 1122 Relative Sort Array
def relativeSortArray(arr1, arr2):
	

# 1137 N-th Tribonacci Number
def tribonacci(n):
	if n <= 1:
		return n

	if n == 2:
		return 1

	f, s, t = 0, 1, 1
	for _ in range(3, n+1):
		f, s, t = s, t, f + s + t

	return t