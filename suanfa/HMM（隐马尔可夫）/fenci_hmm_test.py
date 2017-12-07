# -*- coding: utf-8 -*-
import pickle

def load_model(f_name):
    ifp = open(f_name, 'rb')
    return pickle.load(ifp)  


prob_start = load_model("trainHMM\prob_start.pkl")
prob_trans = load_model("trainHMM\prob_trans.pkl")
prob_emit = load_model("trainHMM\prob_emit.pkl")

#维特比算法
def viterbi(obs, states, start_p, trans_p, emit_p):  
    V = [{}]
    path = {}
    for y in states:   #初始值
    #在位置0，以y状态为末尾的状态序列的最大概率
        V[0][y] = start_p[y] * emit_p[y].get(obs[0],0)   
        path[y] = [y]
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}
        for y in states:      #从y0 -> y状态的递归
            (prob, state) = max([(V[t-1][y0] * trans_p[y0].get(y,0) * \
            emit_p[y].get(obs[t],0) ,y0) for y0 in states if V[t-1][y0]>0])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath  #记录状态序列
        #在最后一个位置，以y状态为末尾的状态序列的最大概率
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])  
    return (prob, path[state])  #返回概率和状态序列


def cut(sentence):
    prob, pos_list =  viterbi(sentence,('B','M','E','S'), \
    prob_start, prob_trans, prob_emit)
    return (prob,pos_list)


if __name__ == "__main__":
    test_str = u"中新网电综合报道"
    prob,pos_list = cut(test_str)
    print (test_str)
    #print (pos_list)
    for i in range(len(test_str)):
        print(test_str[i], end="") 
        if pos_list[i] == "E" or pos_list[i] == "S":
            print(end ="/") 	
		
