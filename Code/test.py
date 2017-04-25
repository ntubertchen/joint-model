from w2v import DataPrepare

path = ["../GloVe/glove.6B.200d.txt" , "../Guide/Data/seq.in" , "../Guide/Data/seq.out" , "../Guide/Data/intent"]
slotpath = '../Guide/Data/slot_list'
intentpath = '../Guide/Data/intent_list'
d = DataPrepare(path,slotpath,intentpath)
x,y,z = d.get_trainbatch()