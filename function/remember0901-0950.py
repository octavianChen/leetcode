# 905 Sort Array By Parity
def sortArrayByParity(A):
    i = -1

    for j in range(len(A)):
        if A[j] % 2 == 0:
            i = i + 1
            A[i], A[j] = A[j], A[i]

    return A


# 938 Range Sum of BST, 注意 total 的值
def rangeSumBST(root, L, R):
	if root is None:
        return 0

    if L > R:
        return 0

    total = 0

    if root.val < L:
        total += rangeSumBST(root.right, L, R)

    elif root.val > R:
        total += rangeSumBST(root.left, L, R)

    else:
        total += root.val
        total += rangeSumBST(root.left, L, R)
        total += rangeSumBST(root.right, L, R)

    return total