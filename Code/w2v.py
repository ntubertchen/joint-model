import numpy as np
import random

class DataPrepare(object):
  def __init__(self,path,slotpath,intentpath):
    self.seed = 0
    self.maxlength = 0
    self.slotdict,self.rev_slotdict,self.slot_dim = self.get_slots(slotpath)
    self.intentdict,self.rev_intentdict,self.intent_dim = self.get_slots(intentpath)
    self.worddict,self.dict_dim = self.get_glove(path[0])
    
    self.encoded,self.reverse = self.set_word2vec(path[1])
    self.slotvalue = self.get_labelvalue(path[2])
    self.intentvalue = self.get_labelvalue(path[3])

  def get_glove(self,GloVe):
    d = {}
    dict_dim = 0
    with open(GloVe,'r') as f:
        for l in f:
            tmp = l.strip().split()
            dict_dim = len(tmp)-1
            d[tmp[0]] = np.asarray(tmp[1:])
    n = np.zeros(dict_dim)
    n[dict_dim-1] = 999
    d['<unk>'] = n
    return d,dict_dim

  def get_slots(self,slotpath):
    d = {}
    rev_d = {}
    count = 0
    with open(slotpath,'r') as f:
      for line in f:
       line = line.strip()
       d[line] = count
       rev_d[count] = line
       count += 1
    return d,rev_d,count

  def set_word2vec(self,seq_in):
    """
     for ***next*** only
    """
    unkbook = open('nuk','w')
    with open(seq_in,'r') as f:
     training_set = []
     rev_set = []
     for line in f:
      batch = []
      #print line
      line = line.strip().split('***next***')
      rev = []
      for i in range(len(line)-1):
        l = line[i].lower()
        rev.append(l)
        l = l.strip().split()
        if len(l) > self.maxlength:
          self.maxlength = len(l)
        sen = []
        for word in l:
          if word in self.worddict:
            sen.append(self.worddict[word])
          else:
            sen.append(self.worddict['<unk>'])
            unkbook.write(word+'\n')
        batch.append(sen)
      rev_set.append(rev)
      training_set.append(batch)
    return training_set, rev_set

  def get_labelvalue(self,label_p):
    training_set = []
    error = open('error','w')
    with open(label_p,'r') as f:
      for line in f:
        batch = []
        l = line.strip().split()
        for word in l:
          word = word.strip()
          if word in self.slotdict:
            vec = np.zeros(self.slot_dim)
            vec[self.slotdict[word]] = 1
            batch.append(vec)
          elif word in self.intentdict:
            vec = np.zeros(self.intent_dim)
            vec[self.intentdict[word]] = 1
            batch.append(vec)
          else:
            batch.append(self.worddict['<unk>'])
            error.write(word+'\n')
            print ("error")
        training_set.append(batch)
    return training_set

  def get_trainbatch(self):
   random.seed(self.seed)
   if self.seed > 99999999:
     self.seed = random.random()
   self.seed += 100
   index = random.randint(0,len(self.encoded) - 1)
   sentence = self.encoded[index]
   slot = self.slotvalue[index]
   for _ in range(self.maxlength - len(self.encoded[index][3])):
    slot.append(np.zeros(self.slot_dim))
   for i in range(len(sentence)):
    for _ in range(self.maxlength - len(sentence[i])):
      sentence[i].append(np.zeros(200))
   return sentence,slot,self.intentvalue[index]
