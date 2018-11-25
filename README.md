# MachineLearning
# author : sunjingpeng6@126.com 
HMM.py
隐马尔科夫模型，数学推导见李航《统计学习方法》 第10章
代码实现HMM的3个基本问题步骤
 1. 概率计算问题:前向-后向算法——动态规划
 给定模型λ=(pi, A , B) , 和观测序列 O ,计算模型λ下观测序列O出现的概率P(O| λ)
 2. 学习问题:Baum-Welch算法(状态未知)——EM
 已知观测序列 O  ,估计模型λ = (pi, A , B)的参数,使得在该模型下观测序列P(O|λ)最大
 3. 预测问题:Viterbi算法——动态规划
 解码问题:已知模型λ = (pi, A , B)和观测序列 O
 求给定观测序列条件概率P(I|O, λ )最大的状态序列I
