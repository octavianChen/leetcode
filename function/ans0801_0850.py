# 814 Binary Tree Pruning
def pruneTree(root):
	if not root:
		return root

	root.left = pruneTree(root.left)
	root.right = pruneTree(root.right)

	if root.left or root.right:
		return root
	else:
		return None if root.val == 0 else root


# 817 Linkded List Components
def numComponents(head, G):
	G, cur, res = set(G), head, 0

	while cur:
		if cur.val in G:
			while cur.next and cur.next.val in G:
				cur = cur.next
			res += 1
		cur = cur.next if cur else None
	return res


# 819 Most Common Word
def mostCommonWord(paragraph, banned):
	import re
	wordlist = re.split(r"[!?',;. ]", paragraph)
	wordlist = [word.lower() for word in wordlist if word]
	m, n = {}, set([word.lower() for word in banned])

	for word in wordlist:
		m[word] = m.get(word, 0) + 1

	m = sorted(m.items(), key=lambda x:x[1], reverse=True)

	for key, val in m:
		if key not in n:
			return key

# 830 Positions of Large Groups
def largeGroupPositions(S):
	left, i, res, S = 0, 0, [], S + " "

	while i < len(S):
		if S[i] == S[left]:
			i += 1

		else: 
			if i - left >= 3:
				res.append([left, i -1])

			left = i
	
	return res


# 832 Flipping an Image
def flipAndInvertImage(A):
	for i in range(len(A)):
		A[i] = A[i][::-1]

		for j in range(len(A[i])):
			A[i][j] = 1- A[i][j]

	return A

# 834 Sum of Distances in Tree, 会 TLE
def sumOfDistancesInTree(N, edges):
	import Queue
	graph, res = [[] for _ in range(N)], [0 for _ in range(N)]

	for u, v in edges:
		graph[u].append(v)
		graph[v].append(u)

	for i in range(N):
		distance, visited = [float("inf") for _ in range(N)], [-1 for _ in range(N)]
		distance[i], visited[i] = 0, 0
		q = Queue.Queue()
		q.put(i)

		while not q.empty():
			i = q.get()

			for j in graph[i]:
				if visited[j] == -1:
					visited[j] = 0
					q.put(j)
					distance[j] = distance[i] + 1
			visited[i] = 1
		res[i] = sum(distance)
	return res

# 836 Rectangle Overlap
def isRectangleOverlap(self, rec1, rec2):
	if rec1[0] >= rec2[2] or rec1[2] <= rec2[0] or rec1[1] >= rec2[3] or rec1[3] <= rec2[1]:
		return False
	return True
