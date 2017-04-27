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
			#print tmp_taggings
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
		"""if (n[i] != [-1]):
			print i
			print taggings[i]
			print final_lines[i]
			print tmp_tags"""
		final_tags = tmp_tags

	return final_lines, final_tags

def parse_one_json(json_dir,speaker_list,args,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12):
	with open(json_dir + '/label.json', 'r') as jsonfile:
		data = json.load(jsonfile)
	sentences = []
	counter = 0
	pref = []
	pref_tag = []
	pref_int = []
	empty = "Empty"
	for _ in range(7):
		pref.append(empty)
		pref_tag.append("O")
		pref_int.append("None-None")
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
					attr = 'None'
				intent += "-" + attr

			# sentence += line["speech_act"][i]["act"] + " "
			# intent = ""
			# for attr in line["speech_act"][i]["attributes"]:
			# 	if attr == "":
			# 		attr = "MAIN"
			# 	intent += "-" + attr
			# sentence += intent
			#print (sentence,intent)
			pref = pref[1:]
			pref_tag = pref_tag[1:]
			pref_int = pref_int[1:]
			pref.append(final_line)
			pref_tag.append(final_tag)
			pref_int.append(intent)
			if speaker_info["speaker"] == "Tourist" and args.tourist != True and args.all != True:
				continue
			elif speaker_info["speaker"] == "Guide" and args.guide != True and args.all != True:
				continue
			if random.random() < 0.7:
				for s in pref:
					f1.write(s + " ***next*** ")
				for s in pref_tag:
					f5.write(s + " ***next*** ")
				for s in pref_int:
					f9.write(s + " ***next*** ")
				f1.write(speaker_info["speaker"])
				f1.write('\n')
				f5.write('\n')
				f9.write('\n')
			elif random.random() < 0.8:
				for s in pref:
					f2.write(s + " ***next*** ")
				for s in pref_tag:
					f6.write(s + " ***next*** ")
				for s in pref_int:
					f10.write(s + " ***next*** ")
				f2.write(speaker_info["speaker"])
				f2.write('\n')
				f6.write('\n')
				f10.write('\n')
			else:
				for s in pref:
					f3.write(s + " ***next*** ")
				for s in pref_tag:
					f7.write(s + " ***next*** ")
				for s in pref_int:
					f11.write(s + " ***next*** ")
				f3.write(speaker_info["speaker"])
				f3.write('\n')
				f7.write('\n')
				f11.write('\n')
			for s in pref:
					f4.write(s + " ***next*** ")
			for s in pref_tag:
					f8.write(s + " ***next*** ")
			for s in pref_int:
					f12.write(s + " ***next*** ")
			f4.write(speaker_info["speaker"])					
			f4.write('\n')
			f8.write('\n')
			f12.write('\n')

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

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--guide",action="store_true",help="guide conversation only")
group.add_argument("--tourist",action="store_true",help="tourist conversation only")
group.add_argument("--all",action="store_true",help="all conversation")
args = parser.parse_args()
FinalLines = []
FinalTags = []
finalintent = []
if args.guide == True:
	f1 = open('../Guide/Data/train/seq.in','w')
	f2 = open('../Guide/Data/test/seq.in','w')
	f3 = open('../Guide/Data/valid/seq.in','w')
	f4 = open('../Guide/Data/seq.in','w')
	f5 = open('../Guide/Data/train/seq.out','w')
	f6 = open('../Guide/Data/test/seq.out','w')
	f7 = open('../Guide/Data/valid/seq.out','w')
	f8 = open('../Guide/Data/seq.out','w')
	f9 = open('../Guide/Data/train/intent','w')
	f10 = open('../Guide/Data/test/intent','w')
	f11 = open('../Guide/Data/valid/intent','w')
	f12 = open('../Guide/Data/intent','w')
elif args.tourist == True:
	f1 = open('../Tourist/Data/train/seq.in','w')
	f2 = open('../Tourist/Data/test/seq.in','w')
	f3 = open('../Tourist/Data/valid/seq.in','w')
	f4 = open('../Tourist/Data/seq.in','w')
	f5 = open('../Tourist/Data/train/seq.out','w')
	f6 = open('../Tourist/Data/test/seq.out','w')
	f7 = open('../Tourist/Data/valid/seq.out','w')
	f8 = open('../Tourist/Data/seq.out','w')
	f9 = open('../Tourist/Data/train/intent','w')
	f10 = open('../Tourist/Data/test/intent','w')
	f11 = open('../Tourist/Data/valid/intent','w')
	f12 = open('../Tourist/Data/intent','w')
else:
	f1 = open('../All/Data/train/seq.in','w')
	f2 = open('../All/Data/test/seq.in','w')
	f3 = open('../All/Data/valid/seq.in','w')
	f4 = open('../All/Data/seq.in','w')
	f5 = open('../All/Data/train/seq.out','w')
	f6 = open('../All/Data/test/seq.out','w')
	f7 = open('../All/Data/valid/seq.out','w')
	f8 = open('../All/Data/seq.out','w')
	f9 = open('../All/Data/train/intent','w')
	f10 = open('../All/Data/test/intent','w')
	f11 = open('../All/Data/valid/intent','w')
	f12 = open('../All/Data/intent','w')

for i in range(54):
	json_dir = str(i+1)
	if (len(json_dir) == 1):
		json_dir = '00' + json_dir
	elif (len(json_dir) == 2):
		json_dir = '0' + json_dir
	json_dir = '../dstc5/' + json_dir
	if (not os.path.exists('../dstc5/' + json_dir + '/label.json')):
		continue
	speaker_list = sent_2_speaker(json_dir)
	parse_one_json(json_dir,speaker_list,args,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12)

#ninfile = open('Data/train/train.seq.in', 'w')
#noutfile = open('Data/train/train.seq.out', 'w')
#nlabelfile = open('Data/train/train.label','w')
#vinfile = open('Data/valid/valid.seq.in', 'w')
#voutfile = open('Data/valid/valid.seq.out', 'w')
#vlabelfile = open('Data/valid/valid.label','w')
#tinfile = open('Data/test/test.seq.in', 'w')
#toutfile = open('Data/test/test.seq.out', 'w')
#tlabelfile = open('Data/test/test.label','w')
# for i in range(len(FinalLines)):
# 	random.seed(i)
# 	r = random.randint(1, 10)
# 	if r > 9:
# 		tinfile.write(FinalLines[i] + '\n')
# 		toutfile.write(FinalTags[i] + '\n')
# 		tlabelfile.write(finalintent[i] + '\n')
# 	elif r > 8:
# 		vinfile.write(FinalLines[i] + '\n')
# 		voutfile.write(FinalTags[i] + '\n')
# 		vlabelfile.write(finalintent[i] + '\n')
# 	else:
# 		ninfile.write(FinalLines[i] + '\n')
# 		noutfile.write(FinalTags[i] + '\n')
# 		nlabelfile.write(finalintent[i] + '\n')