import pickle as pkl

vocab_stream = open('/home/xuhaowen/GitHub/t2t/t2t_data/vocab.chn.7', 'rb')
vocab_data = pkl.load(vocab_stream)
print(vocab_data[0])
exit()
for key in vocab_data[0]:
    vocab_data[0][key] -= 2
vocab_stream.close()
vocab_stream = open('/home/xuhaowen/GitHub/t2t/t2t_data/vocab.chn', 'wb')
pkl.dump(vocab_data, vocab_stream, 2)
vocab_stream.close()
