# -*- coding: utf-8 -*-

# 二元隐马尔科夫模型（Bigram HMMs）
# 'trainCorpus.txt_utf8'为人民日报已经人工分词的预料，29万多条句子

import sys
import pickle
#state_M = 4
#word_N = 0
A_dic = {} #状态转移
B_dic = {} #发射状态
Count_dic = {}
Pi_dic = {} #初始状态概率
word_set = set()
state_list = ['B','M','E','S']
line_num = -1

INPUT_DATA = "trainCorpus.txt_utf8"
PROB_START = "trainHMM\prob_start.pkl"   #初始状态概率
PROB_EMIT = "trainHMM\prob_emit.pkl"     #发射概率
PROB_TRANS = "trainHMM\prob_trans.pkl"   #转移概率


def init():  #初始化字典
    #global state_M
    #global word_N
    for state in state_list:
        A_dic[state] = {}
        for state1 in state_list:
            A_dic[state][state1] = 0.0
    for state in state_list:
        Pi_dic[state] = 0.0
        B_dic[state] = {}
        Count_dic[state] = 0


def getList(input_str):  #输入词语，输出状态
    outpout_str = []
    if len(input_str) == 1:
        outpout_str.append('S')
    elif len(input_str) == 2:
        outpout_str = ['B','E']
    else:
        M_num = len(input_str) -2
        M_list = ['M'] * M_num
        outpout_str.append('B')
        outpout_str.extend(M_list)  #把M_list中的'M'分别添加进去
        outpout_str.append('E')
    return outpout_str


def Output():   #输出模型的三个参数：初始概率+转移概率+发射概率
    start_fp = open(PROB_START,'wb')
    emit_fp = open(PROB_EMIT,'wb')
    trans_fp = open(PROB_TRANS,'wb')
    print ("len(word_set) = %s " % (len(word_set)))

    for key in Pi_dic:           #状态的初始概率
        Pi_dic[key] = Pi_dic[key] * 1.0 / line_num
    pickle.dump(Pi_dic, start_fp)

    for key in A_dic:            #状态转移概率
        for key1 in A_dic[key]:
            A_dic[key][key1] = A_dic[key][key1] / Count_dic[key]
    pickle.dump(A_dic, trans_fp)

    for key in B_dic:            #发射概率(状态->词语的条件概率)
        for word in B_dic[key]:
            B_dic[key][word] = B_dic[key][word] / Count_dic[key]
    pickle.dump(B_dic, emit_fp)

    start_fp.close()
    emit_fp.close()
    trans_fp.close()


def main():

    ifp = open(INPUT_DATA, encoding="utf-8")
    init()
    global word_set   #初始是set()
    global line_num   #初始是-1
    for line in ifp:
        line_num += 1
        if line_num % 10000 == 0:
            print (line_num)

        line = line.strip()
        if not line:continue
        #line = line.decode("utf-8","ignore")  #设置为ignore，会忽略非法字符


        word_list = []
        for i in range(len(line)):
            if line[i] == " ":continue
            word_list.append(line[i])
        word_set = word_set | set(word_list)   #训练预料库中所有字的集合,并集


        lineArr = line.split(" ")
        line_state = []
        for item in lineArr:
        #一句话对应一行连续的状态，一句话的状态值
            line_state.extend(getList(item))   
        if len(word_list) != len(line_state):
            print  (sys.stderr,"[line_num = %d][line = %s]" % (line_num, line.endoce("utf-8",'ignore')))
        else:
            for i in range(len(line_state)):
                if i == 0:
                #Pi_dic记录句子第一个字的状态，用于计算初始状态概率
                    Pi_dic[line_state[0]] += 1  
                  #记录每一个状态的出现次数 
                    Count_dic[line_state[0]] += 1   
                else:
                #用于计算转移概率
                    A_dic[line_state[i-1]][line_state[i]] += 1    
                    Count_dic[line_state[i]] += 1
                    if word_list[i] not in B_dic[line_state[i]]:
                        B_dic[line_state[i]][word_list[i]] = 0.0
                    else:
                     #用于计算发射概率
                        B_dic[line_state[i]][word_list[i]] += 1  
    Output()
    ifp.close()


if __name__ == "__main__":
    main()