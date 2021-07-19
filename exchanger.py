
f = open('ex.txt', 'w')
a = open('./data/entail_evaluation_set.txt', 'r')

text = a.read()
new_text = text.split(sep=' ')
for i in new_text:
    f.write(f'{i}\n')
    

f.close()
a.close()