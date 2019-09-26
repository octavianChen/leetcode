class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

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

# 109 Convert Sorted List to Binary Search Tree,快慢指针找中间节点
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


# 110 Balanced Binary Tree, 后续遍历的结果
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


# 117 Populating Next Right Pointers in Each Node II, 队列的操作，注意递归的用法
# 需要平行地扫描父节点如果父节点是叶节点的话
def connect(root):
    queue = [root] if root else []

    while queue:
        new_queue = []
        for i in range(len(queue)):
            if i != len(queue) - 1:
                queue[i].next = queue[i+1]
            else:
                queue[i].next = None
            
            left, right = queue[i].left, queue[i].right

            if left:
                new_queue.append(left)

            if right:
                new_queue.append(right)
        queue = new_queue
    return root


def connect(root):
    if not root:
        return root

    p = root.next

    while p:
        if p.left:
            p = p.left
            break
        if p.right:
            p = p.right
            break
        p = p.next
    if root.right:
        root.right.next = p
    if root.left:
        root.left.next = root.right if root.right else p
    connect(root.left)
    connect(root.right)
    return root

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


# 122 Best Time to Buy and Sell Stock II,有差值就加到结果里
def maxProfit(prices):
    res = 0
    for i in range(len(nums)-1):
        if prices[i] < prices[i+1]:
            res += prices[i+1]-prices[i]
    return res

# 124 Binary Tree Maximum Path Sum, 注意递归的调用以及只加加正数的操作
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


# 133 Clone Graph, 本质上还是用字典保存复制的节点，然后 dfs
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


# 136 Single Number,异或位操作，两个相同的数字异或为0,0与任何数字异或还是原来的数字
def singleNumber(nums):
    res = 0
    for num in nums:
        res ^= num
    return res


# 137 Single Number II, 把32位的二进制数进行遍历，统计每个数字的每一位出现的次数和
# 如果某一位出现的次数不是3，那么该位置一定是因为那个只出现1次的数字导致的。用来保存
# 结果的res是0，使用或操作，就能把该位置数字变成1
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

# 142 Linked List Cycle II, 先找到相遇的位置，然后慢指针回头，快慢都单步走
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


# 145 Binary Tree Postorder Traversal, 后续遍历难度在于先要判断上次访问的节点是位于左子树
# 右子树。若是左子树，需要跳过根节点，进入右子树；若是右子树，直接访问根节点
def postorderTraversal(root):
    m, res, pcur, plast = [], [], root, None

    # 先走到最左侧节点
    while pcur:
        m.append(pcur)
        pcur = pcur.left

    while m:
        pcur = m.pop()

        # 没有右子树或者已经访问过右子树，直接上根节点
        if not pcur.right or pcur.right == plast:
            res.append(pcur.val)
            plast = pcur

        # 否则跳过根节点，进入右子树，跳过就是重新入栈
        else:
            m.append(pcur)
            pcur = pcur.right

            while pcur:
                m.append(pcur)
                pcur = pcur.left

    return res

# 145 Binary Tree Postorder Traversal
def postorderTraversal(root):
    st, res = [], []

    if root:
        return res
    st.append(root)

    while st:
        t = st.pop()
        res.append(t.val)
        if t.left:
            st.append(t.left)
        if t.right:
            st.append(t.right)
    return res[::-1]



# 147 Insertation Sort List, 找到位置最关键
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

# 另外一种方法是用有序字典来实现
from collections import OrderedDict
class LRUCache(object):
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if self.cache.has_key(key):
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

        else:
            return -1

    def set(self, key, value):
        if self.cache.has_key(key):
            self.cache.pop(key)
            self.cache[key] = value

        else:
            if len(self.cache) == self.cap:
                self.cache.popitem(False)
                self.cache[key] = value
            else:
                self.cache[key] = value


# 148 Sort List, 链表的快排，合并排序见另一个文件
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

# 150 Evaluate Reverse Polish Notation,逆波兰表达式遇到不是符号的直接压到栈里
# 如果遇到符号，取出top1, top2，然后top2 计算符号 top1，将值压入栈里
# 这里python向0取整的方法记住
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