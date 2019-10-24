class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 101 Symmetric Tree
def isSymmetric(root):
    if not root:
        return True

    return help(root.left, root.right)

def help(p1, p2):
    if not p1 and not p2:
        return True

    elif p1 and p2 and p1.val == p2.val:
        return help(p1.left, p2.right) and help(p1.right, p2.left)
    
    else:
        return False


# 102 Binary Tree Level Order Traversal
def levelOrder(root):
    import Queue
    res, q = [], Queue.Queue()

    if root:
        q.put(root)

    while not q.empty():
        lev, m = [], q.qsize()

        for _ in range(m):
            node = q.get()
            lev.append(node.val)
            if node.left:
                q.put(node.left)
            if node.right:
                q.put(node.right)
        res.append(lev)
    return res


# 103 Binary Tree Zigzag Level Order Traversal
def zigzagLevelOrder(root):
    import Queue
    res, q = [], Queue.Queue()
    if root:
        q.put(root)
    count = 1
    while not q.empty():
        lev, m = [], q.qsize()
        for _ in range(m):
            node = q.get()
            lev.append(node.val)
            if node.left:
                q.put(node.left)
            if node.right:
                q.put(node.right)
        res.append(lev) if count % 2 else res.append(lev[::-1])
        count += 1
    return res


# 104 Maximum Depth of Binary Tree
def maxDepth(root):
    if not root:
        return 0
    return max(maxDepth(root.left), maxDepth(root.right)) + 1


# 105 Construct Binary Tree from Preorder and Inorder Traversal
def buildTree(inorder, preorder):
    return help(preorder, 0, len(preorder)-1, inorder, 0, len(inorder)-1)

def help(preorder, pl, pr, inorder, il, ir):
    if pl > pr or il > ir:
        return None

    for i in range(il, ir+1):
        if inorder[i] == preorder[pl]:
            break

    cur = TreeNode(preorder[pl])
    cur.left = help(preorder, pl+1, pl+i-il, inorder, il, i-1)
    cur.right = help(preorder, pl+i-il+1, pr, inorder, i+1, ir)
    return cur


# 106 Construct Binary Tree from Inorder and Postorder Traversal
def buildTree(inorder, postorder):
    return help(inorder, 0, len(inorder)-1, postorder, 0, len(postorder)-1)

def help(inorder, il, ir, postorder, pl, pr):
    if il > ir or pl > pr:
        return None
    for i in range(il, ir+1):
        if inorder[i] == postorder[pr]:
            break

    cur = TreeNode(postorder[pr])
    cur.left = help(inorder, pl, i-1, postorder, il, il+i-pl-1)
    cur.right = help(inorder, i+1, pr, postorder, ir-pr+i+1, ir)
    return cur


# 107 Binary Tree Level Order Traversal II
def levelOrderBottom(root):
    import Queue
    res, q = [], Queue.Queue()

    if root:
        q.put(root)

    while not q.empty():
        lev, m = [], q.qsize()

        for _ in range(m):
            node = q.get()
            lev.append(node.val)
            if node.left:
                q.put(node.left)
            if node.right:
                q.put(node.right)
        res.append(lev)
    return res[::-1]

# 108 Convert Sorted Array to Binary Search Tree
def sortedArrayToBst(nums):
    return help(nums, 0, len(nums)-1)

def help(nums, lo, hi):
    if lo > hi:
        return None

    mid = (lo + hi + 1) // 2

    cur = TreeNode(nums[mid])
    cur.left = help(nums, lo, mid-1)
    cur.right = help(nums, mid + 1, hi)
    return cur

# 109 Convert Sorted List to Binary Search Tree
def sortedListToBST(head):
    if not head:
        return None
    if not head.next:
        return TreeNode(head.val)

    last = slow = fast = head

    while fast and fast.next:
        last, slow, fast = slow, slow.next, fast.next.next

    last.next, fast, cur = None, slow.next, TreeNode(slow.val)
    cur.left = sortedListToBST(head)
    cur.right = sortedListToBST(fast)
    return cur

# 110 Balanced Binary Tree
def isBalanced(root):
    return False if help(root) == -1 else True

def help(root):
    if not root:
        return 0

    left = help(root.left)
    if left == -1:
        return -1

    right = help(root.right)
    if right == -1:
        return -1

    return -1 if abs(left-right) > 1 else 1+max(left, right)

# 111 Minimum Depth of Binary Tree
def minDepth(root):
    if not root:
        return 0
    elif root.left and root.right:
        return min(minDepth(root.left), minDepth(root.right)) + 1
    else:
        return minDepth(root.right) + 1 if not root.left else minDepth(root.left) + 1


# 112 Path Sum
def hasPathSum(root, sum):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == sum
    return hasPathSum(root.left, sum-root.val) or hasPathSum(root.right, sum-root.val)


# 113 Path Sum II
def pathSum(root):
    out, res = [], []
    dfs(root, sum, out, res)
    return res

def dfs(p, sum, out, res):
    if not p:
        return
    
    out.append(p.val)

    if not p.left and not p.right and p.val == sum:
        res.append(out[:])
    
    dfs(p.left, sum-p.val, out, res)
    dfs(p.right, sum-p.val, out, res)
    out.pop()

# 114 Flatten Binary Tree to Linked List
def flatten(root):
    help(root)

def help(root):
    if not root:
        return root

    left = help(root.left)
    right = help(root.right)
    root.left = None
    root.right = left
    cur = root
    while cur.right:
        cur = cur.right
    cur.right = right
    return root

# 115 Distinct SubSequences
def numDistinct(s, t):
    dp = [[0 for _ in range(len(t) + 1)] for _ in range(len(s) + 1)]
    for i in range(len(s)+1):
        dp[i][0] = 1

    for i in range(1, len(s)+1):
        for j in range(1, len(t)+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j]

    return dp[len(s)][len(t)]


# 116 Populating Next Right Pointers in Each Node
def connect(root):
	if not root:
		return root

	if root.left:
		root.left.next = root.right

	if root.right:
		root.right.next = root.next.left if root.next else None

	connect(root.left)
	connect(root.right)
	return root

# 117 Populating Next Right Pointers in Each Node II
def connect(root):
	import Queue
	q, pre = Queue.Queue(), None

	if root:
		q.put(root)

	while not q.empty():
		m = q.qsize()

		for i in range(m):
			if i == 0:
				t = q.get()
				pre = t
			else:
				t = q.get()
				pre.next = t
				pre = pre.next

			if t.left:
				q.put(t.left)
			if t.right:
				q.put(t.right)
	return root


# 118 Pascal's Triangle
def generate(numRows):
    res = []
    
    for i in range(numRows):
        out = [1 for _ in range(i+1)]
        if i == 0 or i == 1:
            res.append(out[:])
        else:
            candi = res[i-1][:]
            for j in range(len(candi)-1):
                out[j+1] = candi[j] + candi[j+1]
            res.append(out[:])
    return res


# 119 Pascal's Triangle II
def getRow(rowIndex):
    out = [1 for _ in range(rowIndex+1)]

    if rowIndex == 0 or rowIndex == 1:
        return out[:]

    candi = getRow(rowIndex - 1)
    for i in range(len(candi)-1):
        out[i+1] = candi[i] + candi[i+1]
    return out[:]


# 120 Triangle
def minimumTotal(triangle):
    for i in range(1, len(triangle)):
        for j in range(len(triangle[i])):
            if j == 0:
                triangle[i][j] += triangle[i-1][j]
            elif j == len(triangle[i]) - 1:
                triangle[i][j] += triangle[i-1][j-1]
            else:
                triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])

    res = float("inf")
    for num in triangle[len(triangle)-1]:
        res = min(res, num)
    return res


# 121 Best Time to Buy and Sell Stock
def maxProfit(prices):
    low, res = float("inf"), 0

    for price in prices:
        low = min(low, price)
        res = max(res, price - low)
    return res


# 122 Best Time to Buy and Sell Stock
def maxProfit(prices):
    res = 0
    for i in range(len(nums)-1):
        if prices[i] < prices[i+1]:
            res += prices[i+1]-prices[i]
    return res

# 124 Binary Tree Maximum Path Sum
class Solution(object):
    def maxPathSum(self, root):
        self.res = -float("inf")


        def help(node):
            if node is None:
                return 0

            lv = max(help(node.left), 0)
            rv = max(help(node.right),0)
            self.res = max(self.res, lv + rv + node.val)
            return max(lv, rv) + node.val

        help(root)
        return self.res

# 125 Valid Palindrome
def isPalindrome(s):
    res = "".join([c.lower() for c in s if c.isalnum()])
    return res == res[::-1]

# 126 Word Ladder II
def findLadders(beginWord, endWord, wordList):
    


# 127 Word Ladder
def ladderLength(beginWord, endWord, wordList):
    m = set(wordList)

    import string, Queue

    q = Queue.Queue()
    q.put((beginWord, 1))

    while not q.empty():
        word, lens = q.get()

        if word == endWord:
            return lens

        for i in range(len(word)):
            for c in string.lowercase:
                neword = word[0:i] + c + word[i+1:]

                if neword != word and neword in m:
                    m.remove(neword)
                    q.put((neword, lens+1))
    return 0

# 129 Sum Root to Leaf Numbers
def sumNumbers(root):
    ans, res = [0], [0]
    dfs(root, ans, res)
    return res


def dfs(p, ans, res):
    if not p:
        return

    if not p.left and not p.right:
        ans[0] = ans[0] * 10 + p.val
        res[0] += ans[0]
        return

    ans[0] = ans[0] * 10 + p.val
    if p.left:
        dfs(p.left, ans, res)
        ans[0] //= 10
    if p.right:
        dfs(p.right, ans, res)
        ans[0] //= 10


# 130 Surrounded Regions
def solve(board):
    if not board or not board[0]:
        return board

    row, col = len(board), len(board[0])

    def dfs(i, j):
        direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        board[i][j] = "c"

        for dirs in direction:
            r, c = i+dirs[0], j+dirs[1]

            if 0 <= r < row and 0 <= c < col and board[r][c] == "O":
                dfs(r, c)

    for i in range(row):
        for j in range(col):
            if (i == 0 or i== row-1 or j == 0 or j == col-1) and board[i][j] == "O":
                dfs(i, j)

    for i in range(row):
        for j in range(col):
            if board[i][j] == "c":
                board[i][j] = "O"
            elif board[i][j] == "O":
                board[i][j] = "X"

# 133 Clone Graph, dfs的操作要明白
def cloneGraph(node):
    m = {}

    def dfs(t):
        if t is None:
            return None

        if t in m:
            return m[t]

        clone = Node(t.val, [])
        m[t] = clone

        for node in t.neighbors:
            clone.neighbors.append(dfs(node))

        return clone

    return dfs(node)


# 134 Gas Station
def canCompleteCircuit(gas, cost):
     

# 136 Single Number,异或位操作，两个相同的数字异或为0,0与任何数字异或还是原来的数字
def singleNumber(nums):
    res = 0
    for num in nums:
        res ^= num
    return res


# 137 Single Number II, 把32位的二进制数进行遍历，统计每个数字的每一位出现的次数和
def singleNumber(nums):
    res = 0

    for i in range(32):
        cnt = 0

        mask = 1 << i

        for num in nums:
            if num & mask:
                cnt += 1

        if cnt % 3 == 1:
            res |= mask

    if res >= 2 ** 31:
        res -= 2 ** 32

    return res

# 138 Copy List with Random Pointer, 每复制一个节点，该节点和复制的节点存到字典里
def copyRandomList(head):
    if head is None:
        return None

    h = Node(head.val, None, None)

    t, cur, m = h, head.next, {}
    m[head] = t
    while cur:
        node = Node(cur.val, None, None)
        m[cur] = node
        t.next = node
        t, cur = t.next, cur.next

    cur, t = head, h

    while cur:
        if cur.random:
            t.random = m[cur.random]
        t, cur = t.next, cur.next

    return h


# 141 Linked List Cycle, 快慢双指针
def hasCycle(head):
    slow, fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True
    return False


# 142 Linked List Cycle II
def detectCycle(head):
    slow, fast. res = head, head, -1

    while fast and fast.next:
        slow, fast.next = slow.next, fast.next.next

        if slow == fast:
            break

    if not fast or not fast.next:
        return None

    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow


# 143 Reorder List
def reorderList(head):
    if not head or not head.next:
        return
    last, slow, fast, dumpy = head, head, head, ListNode(0)
    while fast and fast.next:
        last, slow, fast = slow, slow.next, fast.next.next
    last.next = None

    cur1, cur0, cur = reverse(slow), head, dumpy
    while cur0:
        cur.next, cur0 = cur0, cur0.next
        cur = cur.next
        cur.next, cur1 = cur1, cur1.next
        cur = cur.next
    cur.next = cur1
    return dumpy.next

def reverse(head):
    dumpy, cur = ListNode(0), head
    while cur:
        tmp = dumpy.next
        dumpy.next, cur = cur, cur.next
        dumpy.next.next = tmp
    return dumpy.next


# 144 Binary Tree Preorder Traversal
def preorderTraversal(root):
    m, res, p = [], [], root

    while m or p:
        while p:
            res.append(p.val)
            m.append(p)
            p = p.left

        if m:
            p = m.pop()
            p = p.right
    return res


# 145 Binary Tree Postorder Traversal
def postorderTraversal(root):
    m, res, pcur, plast = [], [], root, None

    while pcur:
        m.append(pcur)
        pcur = pcur.left

    while m:
        pcur = m.pop()

        if not pcur.right or pcur.right == plast:
            res.append(pcur.val)
            plast = pcur
        else:
            m.append(pcur)
            pcur = pcur.right

            while pcur:
                m.append(pcur)
                pcur = pcur.left

    return res


# 146 LRU Cache, 字典 + 双向链表
class ListNode(object):
    def __init__(self, k, x):
        self.key = k
        self.val = x
        self.prev = None
        self.next = None

class DoubleLinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None

    def isEmpty(self):
        return not self.tail

    def removeLast(self):
        self.remove(self.tail)

    def remove(self, node):
        if self.head == self.tail:
            self.head, self.tail = None, None
            return

        if node == self.head:
            node.next.prev = None
            self.head = node.next
            return

        if node == self.tail:
            node.prev.next = None
            self.tail = node.prev
            return

        node.prev.next = node.next
        node.next.prev = node.prev

    def addFirst(self, node):
        if not self.head:
            self.head = self.tail = node
            node.prev = node.next = None
            return

        node.next = self.head
        self.head.prev = node
        self.head = node
        node.prev = None

class LRU(object):
    def __init__(self, capacity):
        self.cap = capacity
        self.size = 0
        self.m = {}
        self.cache = DoubleLinkedList()

    def get(self, key):
        if key in self.m: # 删除对应节点并将该值加在链表头部
            self.cache.remove(self.m[key])
            self.cache.addFirst(self.m[key])
            return self.m[key].val

        else:
            return -1


    def set(self, key, val):
        if key in self.m:
            self.m[key].val = val
            self.cache.remove(self.m[key])
            self.cache.addFirst(self.m[key])

        else:
            node = ListNode(key, val)
            self.m[key] = node
            self.cache.addFirst(node)
            self.size += 1

            if self.size > self.cap: # 超出容量则删除最后一个节点
                self.size -= 1
                del self.m[self.cache.tail.key]
                self.cache.removeLast()


# 147 Insertation Sort List
def insertionSortList(head):
    dumpy = ListNode(-1)
    cur = dumpy

    while head:
        t, cur = head.next, dumpy
        while cur.next and cur.next.val <= head.val:
            cur = cur.next

        head.next = cur.next
        cur.next = head
        head = t
    return dumpy.next


# 148 Sort List
def sortList(head):
    if not head or not head.next:
        return head

    pre, slow, fast = None, head, head
    while fast and fast.next:
        pre = slow
        slow = slow.next
        fast = fast.next.next

    pre.next = None

    left, right = sortList(head), sortList(slow)
    return mergeTwoLists(left, right)

def mergeTwoLists(l1, l2):
    dumpy = ListNode(0)
    cur = dumpy
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 if l1 else l2
    return dumpy.next


def sortList(head):
    if not head or not head:
        return head

    cur, small, large = head.next, ListNode(0), ListNode(0)
    sp, lp = small, large

    while cur:
        if cur.val < head.val:
            sp.next = cur
            sp = sp.next
        else:
            lp.next = cur
            lp = lp.next

        cur = cur.next
    sp.next, lp.next = None, None
    small, large = sortList(small.next), sortList(large.next)
    cur = small

    if cur:
        while cur.next:
            cur = cur.next
        cur.next = head
        head.next = large
        return small
    else:
        head.next = large
        return head

# 150 Evaluate Reverse Polish Notation
def evalRPN(tokens):
    if len(tokens) == 1:
        return int(tokens[0])

    st = []

    for c in tokens:
        if c not in "+-*/":
            st.append(int(c))
        else:
            a = int(st.pop())
            b = int(st.pop())

            if c == "+":
                st.append(b+a)
            elif c == "-":
                st.append(b-a)
            elif c == "*":
                st.append(b*a)
            else:
                st.append(int(float(b)/a))
    return st.pop()


