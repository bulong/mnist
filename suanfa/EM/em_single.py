import numpy
from scipy import stats

def em_single(priors,observations):

    """
    EM算法的单次迭代
    Arguments
    ------------
    priors:[theta_A,theta_B]
    observation:[m X n matrix]

    Returns
    ---------------
    new_priors:[new_theta_A,new_theta_B]
    :param priors:
    :param observations:
    :return:
    """
    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
    theta_A = priors[0]
    theta_B = priors[1]
    #E step
    for observation in observations:
        len_observation = len(observation) #总的序列长度，即每次抛硬币的次数
        num_heads = observation.sum() #求正面朝上的个数，即为1的个数
        num_tails = len_observation-num_heads
        #二项分布求解公式
        contribution_A = stats.binom.pmf(num_heads,len_observation,theta_A) #计算抛硬币的概率分布
        contribution_B = stats.binom.pmf(num_heads,len_observation,theta_B)
        
        weight_A = contribution_A / (contribution_A + contribution_B)
        weight_B = contribution_B / (contribution_A + contribution_B)
        #更新在当前参数下A，B硬币产生的正反面次数
        counts['A']['H'] += weight_A * num_heads
        counts['A']['T'] += weight_A * num_tails
        counts['B']['H'] += weight_B * num_heads
        counts['B']['T'] += weight_B * num_tails

    # M step
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
    return [new_theta_A,new_theta_B]
    
def em_run(observations,prior,tol = 1e-6,iterations=10000):
    """
    EM算法
    ：param observations :观测数据
    ：param prior：模型初值
    ：param tol：迭代结束阈值
    ：param iterations：最大迭代次数
    ：return：局部最优的模型参数
    """
    iteration = 0;
    while iteration < iterations:
        new_prior = em_single(prior,observations)
        delta_change = numpy.abs(prior[0]-new_prior[0])
        if delta_change < tol:
            break
        else:
            prior = new_prior
            iteration +=1
    return [new_prior,iteration]