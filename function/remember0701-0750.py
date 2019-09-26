# 739 Daily Temperature, 单调栈
def dailyTemperature(T):
	st, res = [], [0 for _ in range(len(T))]

	for i in range(len(T)):
		while st and T[i] > T[st[-1]]:
			t = st.pop()
			res[t] = i - t
		st.append(i)
	return res