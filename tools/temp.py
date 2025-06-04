import numpy as np
# line1 = '17.9	30.9	18.4	8.6	28.0	35.0'
# line2 = '19.7 (+1.8)	33.2 (+2.3)	20.2 (+1.8)	10.4 (+1.8)	29.7 (+1.7)	36.5 (+1.5)'
# result1 = np.array([float(item) for item in line1.split('\t')])
# result2 = np.array([float(item.split(' ')[0]) for item in line2.split('\t')])
# up = np.array([float(item.split(' ')[1][2:-1]) for item in line2.split('\t')])
# aa = result2 - result1
# print(aa)
# print(up)

# line2 = '20.3 (+1.0)	34.2 (+1.3)	20.6 (+0.7)	10.7 (+0.9)	30.9 (+1.1)	37.6 (+0.9)'
# a = 0.7
# result2 = np.array([float(item.split(' ')[0]) for item in line2.split('\t')])
# up = np.array([float(item.split(' ')[1][2:-1]) for item in line2.split('\t')])
# result2 = result2 + a
# up = up + a
# for i in range(len(up)):
#     print(str(round(result2[i], 1)) + ' ' + '(+' + str(round(up[i], 1)) + ')', end='\t')

line = ['17.2 (+2.3)	29.5 (+2.5)	16.9 (+2.0)	8.5 (+2.1)	25.4 (+2.4)	34.0 (+2.1)',
         '19.7 (+1.8)	33.2 (+2.3)	20.2 (+1.8)	10.4 (+1.8)	29.7 (+1.7)	36.5 (+1.5)',
        '19.5 (+1.9)	33.0 (+3.1)	19.9 (+1.5)	10.2 (+1.5)	29.6 (+2.4)	38.1 (+5.0)',
        '21.0 (+1.7)	34.9 (+2.0)	21.3 (+1.4)	11.4 (+1.6)	31.6 (+1.8)	38.3 (+1.6)']
result = []
up = []
for i in range(len(line)):
    result.append([float(item.split(' ')[0]) for item in line[i].split('\t')])
    up.append([float(item.split(' ')[1][2:-1]) for item in line[i].split('\t')])
result = np.array(result)
up = np.array(up)
m, n = result.shape
for i in range(m):
    string = ''
    for j in range(n):
        string += '& ' + str(round(result[i, j], 1)) + ' '
        if up[i, j] == max(up[:, j]):
            string += r'\textbf{(+' + str(round(up[i, j], 1)) + ')}'
        else:
            string += r'(+' + str(round(up[i, j], 1)) + ')'
    print(string)
