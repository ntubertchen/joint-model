with open('../All/Data/intent','r') as f:
	d = dict()
	for line in f:
		line = line.strip().split()
		for l in line:
			if l in d:
				continue
			else:
				d[l] = 1
with open('../All/Data/intent_list','w') as f:
	for key,value in d.iteritems():
		f.write(key+'\n')
with open('../All/Data/seq.out','r') as f:
	d = dict()
	for line in f:
		word = line.strip().split()
		for c in word:
			if c in d:
				continue
			else:
				d[c] = 1
with open('../All/Data/slot_list','w') as f:
	for key,value in d.iteritems():
		f.write(key+'\n')
