# 1160 Find Words That Can Be Formed by Characters
def countChracters(words, chars):
	m, cnt = {}, 0

	for char in chars:
		m[char] = m.get(char, 0) + 1

	