# author : sunjingpeng6@126.com 隐马尔科夫模型，数学推导见李航《统计学习方法》 第10章
import numpy as np

class HMM():
    """
    HMM的3个基本问题
    1. 概率计算问题:前向-后向算法——动态规划
    给定模型λ=(pi, A , B) , 和观测序列 O ,计算模型λ下观测序列O出现的概率P(O| λ)
    2. 学习问题:Baum-Welch算法(状态未知)——EM
    已知观测序列 O  ,估计模型λ = (pi, A , B)的参数,使得在该模型下观测序列P(O|λ)最大
    3. 预测问题:Viterbi算法——动态规划
    解码问题:已知模型λ = (pi, A , B)和观测序列 O
    求给定观测序列条件概率P(I|O, λ )最大的状态序列I
    """
    def __init__(self, Ann, Bnm, pi, iter_times = 2000, tolerance=1e-8):
        self.A = np.array(Ann)
        self.B = np.array(Bnm)
        self.pi = pi
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]
        self.iter_times = iter_times
        self.parameters = []
        self.tolerance = tolerance

    # 函数名称：Forward *功能：前向算法估计参数
    def _forword(self, O):
        T = (len(O))
        alpha = np.zeros((T, self.N))

        alpha[0] = self.pi * self.B[:, O[0]]
        for t  in range(1, T):
            for i in range(self.N):
                alpha[t][i] = np.inner(alpha[t-1], A[:,i]) * B[i][O[t]]
        p = alpha[-1].sum
        return alpha

    # 函数名称：Backward * 功能:后向算法估计参数
    def _backword(self, O):
        T = (len(O))
        beta = np.zeros((T, self.N))
        beta[-1] = 1.0
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                for j in range(self.N):
                    beta[t][i] += self.A[i][j] * beta[t+1][j] * self.B[j, O[t+1]]
        return beta

    # 1. 概率计算问题:前向-后向算法——动态规划
    # 给定模型λ=(pi, A , B) , 和观测序列 O ,计算模型λ下观测序列O出现的概率P(O| λ)
    def p(self, O):
        alpha = self._forword(O)
        p = alpha[-1].sum()
        # p = 0
        # beta = self._backword(O)
        # for i in range(self.N):
        #     p += self.pi[i] * self.B[i][O[0]] * beta[0][i]
        return p

    # 2. 学习问题:Baum-Welch算法(状态未知)——EM
    # 已知观测序列 O  ,估计模型λ = (pi, A , B)的参数,使得在该模型下观测序列P(O|λ)最大
    def BaumWelch(self, O):
        # 保存每次迭代的参数
        params = {}
        # 随机初始化参数
        self.pi, self.A, self.B = self._init_random_params()
        params['pi'], params['A'], params['B'] = self.pi, self.A, self.B
        self.parameters.append(params)
        # EM 算法
        for _ in range(self.iter_times):
            gamma, sai = self._expectation(O)
            self._maximization(O, gamma, sai)
            if self._converged(): break
        return self.parameters[-1]

    def _converged(self):
        """ Covergence if || A - last_A || < tolerance """
        if len(self.parameters) < 2: return False
        diff = np.linalg.norm(self.parameters[-1]['A'] - self.parameters[-2]['A'])
        return diff <= self.tolerance

    def _expectation(self, O):
        """ Calculate the expectation """
        gamma = self._gamma(O)
        sai = self._sai(O)
        return (gamma, sai)

    def _maximization(self, O, gamma, sai):
        """ Update the parameters """
        self.pi = gamma[0]
        sai = np.sum(sai, axis=0)
        self.A = sai / sai.sum(axis=1, keepdims=True)
        for k in range(self.M):
            idx = O == k
            self.B[:, k] = gamma[idx, :].sum(axis = 0)
        self.B = self.B / self.B.sum(axis=1, keepdims=True)
        params = {}
        params['pi'], params['A'], params['B'] = self.pi, self.A, self.B
        self.parameters.append(params)

    def _init_random_params(self):
        """ Initialize parameters randomly """
        def _generate_probablity_distribution(N):
            prob_dist = np.random.uniform(low=0.2, high=0.8, size=N)
            prob_dist = prob_dist / prob_dist.sum()
            return prob_dist
        pi = _generate_probablity_distribution(self.N)
        A = np.array([ _generate_probablity_distribution(self.N) for _ in range(self.N)])
        B = np.array([ _generate_probablity_distribution(self.M) for _ in range(self.N)])
        return (pi, A, B)

    # 计算gamma : 时刻t时马尔可夫链处于状态Si的概率
    def _gamma(self, O):
        alpha = self._forword(O)
        beta = self._backword(O)
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        return gamma

    # 计算sai(i,j) 为给定训练序列O和模型lambda时：
    # 时刻t是马尔可夫链处于Si状态，二时刻t+1处于Sj状态的概率
    def _sai(self, O):
        T = len(O)
        alpha = self._forword(O)
        beta = self._backword(O)
        sai = np.zeros((T-1, self.N, self.N))
        for t in range(T-1):
            for i in range(self.N):
                for j in range(self.N):
                    sai[t][i][j] = alpha[t][i] * A[i][j] * B[j][O[t+1]] * beta[t+1][j]
            sai[t] = sai[t] / sai[t].sum()
        return sai

    # 预测算法 3.1.近似算法 选择概率最大的i作为最有可能的状态
    def approximate(self, O):
        gamma = self._gamma(O)
        pt = np.max(gamma, axis=1)
        probability = 1
        for t in range(len(O)):
            probability *= pt[t]
        I = np.argmax(gamma, axis=1)
        return (I, probability)

    # 3.2 Viterbi算法 实际是用动态规划解HMM预测问题,
    # 用动态规划求概率最大的路径(最优路径),这是一条路径对应一个状态序列。
    def viterbi(self, O):
        # T 观测的时间长度， I 观测序列的状态
        T = len(O)
        I = np.zeros(T)
        # delta(T, N) 定义变量δt(i):在时刻t状态为i的所有路径中,概率的最大值。
        # fai(T,N) 定义变量为在时刻t状态为i的所有路径中概率的最大值的第t-1个节点的状态。
        delta, fai = self._init_viterbi_params(O)
        for t in range(1, T):
            for i in range(self.N):
                temp = delta[t-1] * A[:, i]
                fai[t][i] = np.argmax(temp)
                temp *= B[i, O[t]]
                delta[t][i] = np.max(temp)
        probability = np.max(delta[-1])
        I[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            I[t] = fai[t+1, int(I[t+1])]
        return (I, probability)
    # 初始化
    def _init_viterbi_params(self, O):
        T = len(O)
        delta = np.zeros((T, self.N))
        delta[0] = pi * B[:, O[0]]
        fai = np.zeros((T, self.N))
        return (delta, fai)

if __name__ == '__main__':
    A = np.array([[0.5, 0.2, 0.3],[0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5,0.5],[0.4,0.6],[0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])
    O = np.array([0, 1, 0])
    model = HMM(A, B, pi)
    I, probability = model.approximate(O)
    print(probability)
    print(I)
