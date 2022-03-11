
state_length = 100
data_dir_read='./data/delay_test_150ms.txt'
data_dir_write='./data/Delay_processed_test_150ms_100.txt'
# data_dir_read='./data/loss_test_150ms.txt'
# data_dir_write='./data/Trace_processed_test_150ms_100.txt'
f_read = open(data_dir_read,'r')
value = []
for line in f_read:
    value.append(float(line))
f_read.close()

f_write = open(data_dir_write, 'w')
for i in range(0, len(value), state_length):
    f_write.write(str(value[i:i+state_length]).strip('[]'))
    f_write.write('\n')

f_write.close()