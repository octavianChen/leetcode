#451 Sort Characters By Frequency
def frequencySort(s):
	m, vec = {}, []

	for char in s:
		m[char] = m.get(char, 0) + 1

	vec1 = sorted(m.items(), key=lambda x:x[1], reverse=True)

	for char, val in vec1:
		vec += [char] * val

	return "".join(vec)

# 459 Repeated Substring Pattern
def repeatedSubstringPattern(s):
	if not s:
		return True

	m = len(s)

	for i in range(1, m//2+1):
		if m % i:
			continue
		sub = s[:i]

		if sub * (m // i) == s:
			return True

	return False

# 485 Max Consective One
def findMaxConsectiveOnes(self, nums):
	pre, now, res = 0, 0, 0

	for num in nums:
		now = pre + 1 if num == 1 else 0
		res = max(res, now)

		pre = now

	return res

# 491 Increasing Subsequences
def findSubsequences(nums):
	res, out = set(), []

	def dfs(start):
		if len(out) >= 2:
			res.add(tuple(out))

		for i in range(start, len(nums)):
			if not out or nums[i] >= path[-1]:
				out.append(nums[i])
				dfs(i+1)
				out.pop()


	dfs(0)
	return map(list, res)


# 494 Target Sum
def findTargetSumWays(nums, S):
	

