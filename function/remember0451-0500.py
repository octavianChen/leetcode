# 491 Increasing Subsequences, 记住条件，每次这个dfs都不会
def findSubsequences(nums):
	res, out = set(), []

	def dfs(start):
		if len(out) >= 2:
			res.add(tuple(out))

		for i in range(start, len(nums)):
			if not out or nums[i] >= out[-1]:
				out.append(nums[i])
				dfs(i+1)
				out.pop()


	dfs(0)
	return map(list, res)