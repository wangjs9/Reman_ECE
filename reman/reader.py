import pickle as pk

embedding = pk.load(open('x.txt', 'rb'))
# embedding = pk.load(open('C:/Users/csjwang/Documents/RTHN/data/x.txt', 'rb'))
print(embedding)