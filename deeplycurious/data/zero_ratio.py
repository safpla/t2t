# coding:utf-8
input_file = '../../t2t_datagen_event/word_subword_train'
input_stream = open(input_file, 'r')
count_all = 0
count_long = 0
length_sum = 0
for line in input_stream.readlines():
    count_all += 1
    line = line.strip().split()
    if len(line) > 256:
        count_long += 1
    length_sum += len(line)

print(count_all)
print(count_long)
print(length_sum / count_all)
exit()



input_file = '../../t2t_datagen_event/ner'
input_stream = open(input_file,'r')
count_all = 0
count_o = 0
for line in input_stream.readlines():
    line = line.strip()
    count_all += len(line)
    count_o += len([c for c in line if c == 'O'])

print(float(count_o) / float(count_all))

