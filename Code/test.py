from w2v import DataPrepare

path = ["../GloVe/glove.6B.200d.txt" , "../All/Data/seq.in" , "../All/Data/seq.out" , "../All/Data/intent"]
slotpath = '../All/Data/slot_list'
intentpath = '../All/Data/intent_list'
d = DataPrepare(path,slotpath,intentpath)
seq_in,seq_out,intent = d.get_all()
for i in range(len(seq_in)):
	for j in range(len(seq_in[i])):
		if len(seq_in[i][j]) != d.maxlength or len(seq_out[i][j]) != d.maxlength:
			print seq_in[i][j],seq_out[i][j],d.maxlength