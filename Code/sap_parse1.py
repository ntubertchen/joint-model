import sys
import json
import re
import os
import random
import argparse

def parse_tagging(sentences):
	taggings = []
	types = ['FROM-TO', 'REL', 'CAT']
	n = []
	length = []
	final_lines = ""
	pattern = '<[^\<\>]*>'
	WTF = ['%','% ','%uh', '%Uh', '%Uh ', '%um', '%um ', '%Um', '%Um ', '%oh ', '%Oh ','%hm ','%uh ','%hm ','%Hm','%Hm ','%Ah','%Ah ','%ah','%ah ','%eh','%eh ','%oh','%oh ','%h','%h ','%eh','%eh ','%Eh','%Eh ','%er','%er ','%Er','%Er ','%un','%un ']
	for line in sentences:
		a = re.findall(pattern, line)
		line = line.replace(',', '').replace('?', '').replace('.', '').replace('\r', '')
		for wtf in WTF:
			line = line.replace(wtf, '')
		tmp_line = line
		if (len(a) > 0):
			for s in a:
				tmp_line = tmp_line.replace(s, '')
		tmp_line = tmp_line.strip().replace('  ', ' ')
		if (tmp_line == ''):
			continue
		else:
			length.append(len(tmp_line.split(' ')))
		final_lines = tmp_line.lower().replace('-','')
		if ('<' in line):
			tmp = line.split('<')
			count = (int)((len(tmp) - 1) / 2)
			tmp_n = []
			tmp_taggings = []
			offset = 0
			for i in range(count):
				offset += len(tmp[i*2].split(' ')) - 1
				tmp_n.append(offset)
				tmp_str = tmp[i*2+1].split('>')[0].split(' ')
				tmp_tag = tmp_str[0]
				for j in range(len(types)):
					find = 0
					for k in range(len(tmp_str)):
						if (types[j] in tmp_str[k]):
							tmp_tag += '-' + tmp_str[k].split('\"')[1]
							find = 1
							break
					if (find == 0):
						tmp_tag += '-NONE'
				tmp_taggings.append('B-' + tmp_tag)
				tagwords_length = len(tmp[i*2+1].split('>')[1].strip().split(' '))
				for j in range(tagwords_length-1):
					tmp_n.append(offset+j+1)
					tmp_taggings.append('I-' + tmp_tag)
			taggings.append(tmp_taggings)
			n.append(tmp_n)
		else:
			taggings.append([])
			n.append([-1])

	final_tags = ""
	for i in range(len(taggings)):
		c = 0
		tmp_tags = ''
		for j in range(length[i]):
			if (j in n[i]):
				tmp_tags += taggings[i][c] + ' '
				c += 1
			else:
				tmp_tags += 'O '
		final_tags = tmp_tags

	return final_lines, final_tags

def parse_one_json(json_dir,speaker_list,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,intype):
	with open(json_dir + '/label.json', 'r') as jsonfile:
		data = json.load(jsonfile)
	sentences = []
	counter = 0
	step_pref_guide = []
	step_pref_tourist = []
	i_pref_guide = []
	i_pref_tourist = []
	s_pref_guide = []
	s_pref_tourist = []
	empty = "Empty"
	for i in range(3):
		
		step_pref_tourist.append(3-i)
		
		i_pref_tourist.append("None-none")
		s_pref_tourist.append('O')
	for i in range(4):
		step_pref_guide.append(4-i)
		i_pref_guide.append("None-none")
		s_pref_guide.append('O')
	for line in data["utterances"]:
		speaker_info = speaker_list[counter]
		counter += 1
		for i in range(len(line["semantic_tagged"])):
			"""
			SAP solo training, full information for SAP
			sentence = "guide_act-"+ speaker_info["guide_act"] + " "
			sentence += "initiativity-" + speaker_info["initiativity"] + " "
			sentence += "target_bio-" + speaker_info["target_bio"] + " "
			sentence += "topic-" + speaker_info["topic"] + " "
			sentence += "tourist_act-" + speaker_info["tourist_act"] + " "
			sentence += "speaker-" + speaker_info["speaker"] + " "
			"""
			sentence = line["semantic_tagged"][i].encode('utf-8')
			final_line, final_tag = parse_tagging([sentence])

			# sentence += line["speech_act"][i]["act"] + " "
			# for attr in line["speech_act"][i]["attributes"]:
			# 	if attr == "":
			# 		attr = "MAIN"
			# 	sentence += attr + " "
			intent = ""
			if line["speech_act"][i]["act"] == "":
				intent = "None"
			else:
				intent = line["speech_act"][i]["act"].strip()
			for attr in line["speech_act"][i]["attributes"]:
				attr = attr.strip()
				# if attr == "HOW_TO ":
				# 	print "ass"
				if attr == "":
					attr = 'none'
				intent += "-" + attr
			

			if speaker_info["speaker"].strip() == "Guide":
				i_pref_guide = i_pref_guide[1:]
				s_pref_guide = s_pref_guide[1:]
				step_pref_guide = step_pref_guide[1:]
				i_pref_guide.append(intent)
				s_pref_guide.append(final_tag)
				step_pref_guide.append(0)
			elif speaker_info["speaker"].strip() == "Tourist":
				i_pref_tourist = i_pref_tourist[1:]
				s_pref_tourist = s_pref_tourist[1:]
				step_pref_tourist = step_pref_tourist[1:]
				i_pref_tourist.append(intent)
				s_pref_tourist.append(final_tag)
				step_pref_tourist.append(0)
			else:
				print ("speaker error",speaker_info["speaker"].strip())

			for i in range(3):
				step_pref_tourist[i] += 1
			for i in range(4):
				step_pref_guide[i] += 1
			if speaker_info["speaker"].strip() == "Guide":
				if intype == "train":
					for s in s_pref_tourist:
						f5.write(s + " ***next*** ")
					for s in s_pref_guide:
						f5.write(s + " ***next*** ")
					for s in i_pref_tourist:
						f9.write(s + " ***next*** ")
					for s in i_pref_guide:
						f9.write(s + " ***next*** ")
					for s in step_pref_tourist:
						f13.write(str(s) + " ***next*** ")
					for s in step_pref_guide:
						f13.write(str(s) + " ***next*** ")
					f4.write('Guide\n')
					f5.write('\n')
					f9.write('\n')
					f13.write('\n')
				elif intype == "test":
					for s in s_pref_tourist:
						f6.write(s + " ***next*** ")
					for s in s_pref_guide:
						f6.write(s + " ***next*** ")
					for s in i_pref_tourist:
						f10.write(s + " ***next*** ")
					for s in i_pref_guide:
						f10.write(s + " ***next*** ")
					for s in step_pref_tourist:
						f14.write(str(s) + " ***next*** ")
					for s in step_pref_guide:
						f14.write(str(s) + " ***next*** ")
					f8.write("Guide\n")
					f6.write('\n')
					f10.write('\n')
					f14.write('\n')

def sent_2_speaker(json_dir):
	with open(json_dir + '/log.json', 'r') as jsonfile:
		data = json.load(jsonfile)
	speaker = []
	for line in data["utterances"]:
		info = dict()
		if "guide_act" in line["segment_info"]:
			info["guide_act"] = line["segment_info"]["guide_act"]
		else:
			info["guide_act"] = ""
		if "initiativity" in line["segment_info"]:
			info["initiativity"] = line["segment_info"]["initiativity"]
		else:
			info["initiativity"] = ""
		if "target_bio" in line["segment_info"]:
			info["target_bio"] = line["segment_info"]["target_bio"]
		else:
			info["target_bio"] = ""
		if "topic" in line["segment_info"]:
			info["topic"] = line["segment_info"]["topic"]
		else:
			info["topic"] = ""
		if "tourist_act" in line["segment_info"]:
			info["tourist_act"] = line["segment_info"]["tourist_act"]
		else:
			info["tourist_act"] = ""
		info["speaker"] = line["speaker"]
		speaker.append(info)
	return speaker

# parser = argparse.ArgumentParser()
# group = parser.add_mutually_exclusive_group()
# group.add_argument("--guide",action="store_true",help="guide conversation only")
# group.add_argument("--tourist",action="store_true",help="tourist conversation only")
# group.add_argument("--all",action="store_true",help="all conversation")
# args = parser.parse_args()
FinalLines = []
FinalTags = []
finalintent = []

f1 = open('../All/Sap/train/seq.in','w')
f2 = open('../All/Sap/test/seq.in','w')
f3 = open('../All/Sap/valid/seq.in','w')
f4 = open('../All/Sap/train/talker','w')
f5 = open('../All/Sap/train/seq.out','w')
f6 = open('../All/Sap/test/seq.out','w')
f7 = open('../All/Sap/valid/seq.out','w')
f8 = open('../All/Sap/test/talker','w')
f9 = open('../All/Sap/train/intent','w')
f10 = open('../All/Sap/test/intent','w')
f11 = open('../All/Sap/valid/intent','w')
f12 = open('../All/Sap/intent','w')
f13 = open('../All/Sap/train/info','w')
f14 = open('../All/Sap/test/info','w')
f15 = open('../All/Sap/valid/info','w')

#train_dir = ['001','002','003','004','006','007','008','009','010','011','012','013','016','017','019','020','021','022','023','024','025','026','028','030','031','032','033','035','039','040','041','047','048','052','053']
#train_4 = ['002']
train_4 = ['001','002','003','004','006','007','008','009','010','012','013','017','019','022']
test_4 = ['021','023','024','030','033','035','041','047','048']
dev_4 =  ['011','016','020','025','026','028']
for i in range(len(test_4)):
	json_dir = '../dstc5/'
	if (not os.path.exists( json_dir +test_4[i] +'/label.json')):
		print ('dir error')
	else:
		speaker_list = sent_2_speaker(json_dir + test_4[i])
		parse_one_json(json_dir + test_4[i],speaker_list,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,"test")
for i in range(len(train_4)):
	json_dir = '../dstc5/'
	if (not os.path.exists( json_dir +train_4[i] +'/label.json')):
		print ('dir error')
	else:
		speaker_list = sent_2_speaker(json_dir + train_4[i])
		parse_one_json(json_dir + train_4[i],speaker_list,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,"train")
for i in range(len(dev_4)):
	json_dir = '../dstc5/'
	if (not os.path.exists( json_dir +dev_4[i] +'/label.json')):
		print ('dir error')
	else:
		speaker_list = sent_2_speaker(json_dir + dev_4[i])
		parse_one_json(json_dir + dev_4[i],speaker_list,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,"dev")