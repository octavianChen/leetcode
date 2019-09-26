# 606 Construct String from Binary Tree, 注意加的位置
def tree2str(t):
	if not t:
		return ""
	res = [""]
	help(t, res)
	return res[0][1:-1]

def help(t, res):
	if not t:
		return

	res[0] += "(" + str(t.val)

	if not t.left and t.right:
		res += "()"
	help(t.left, res)
	
	help(t.right, res)
	res[0] += ")"

# 633 Sum of Square Numbers
def judgeSquareSum(c):
	import math
	m = set()

	for i in range(int(math.sqrt(c)) + 2):
		m.add(i * i)
		if c - i * i in m:
			return True
	return False

# 648 Replace words,字典树, 逻辑要理清
class Trie(object):
	def __init__(self):
		self.root = {}
		self.res = []

	def insert(self, prefix):
		p = self.root

		for char in prefix:
			if char not in p:
				p[char] = {}
			p = p[char]
		p["END"] = True

	def search(self, word):
		p, cur = self.root, ""

		for i in range(len(word)):
			if word[i] not in p: # 不在前缀树中就break掉
				break

			cur += word[i]

			p = p[word[i]]

			if "END" in p: # 遇到 END 就结尾返回
				return cur
		return word

def replaceWords(dicts, sentence):
	t, res = Trie(), []
	for prefix in dicts:
		t.insert(prefix)

	wordlist = sentence.strip().split(" ")

	for word in wordlist:
		res.append(t.search(word))

	return " ".join(res)

