import random
i = 0
for _ in range(10000):
	random.seed(i)
	print random.randint(0,99999)
	i += 1