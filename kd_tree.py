import numpy as np

# 最近邻 kd tree

class KdNode():
    def __init__(self, dom_element, left=None, right=None):
        # n维向量节点(n维空间中的一个样本点)
        self.dom_element = dom_element
        # 该结点分割超平面左子空间构成的kd-tree
        self.left = left
        # 该结点分割超平面右子空间构成的kd-tree
        self.right = right

class KdTree():
    """
    最近邻搜索 kd tree
    具体推导过程见李航《统计学习方法》第3章 k近邻法   3.3 k近邻法的实现:kd树
    Parameters：
    -----------------------
    dataset 数据集
    ------------------------
    返回距离测试样本 x 最近的点
    tips:
    kd树是二叉树,表示对k维空间的一个划分,其每个结
    点对应于k维空间划分中的一个超矩形区域。利用kd树可以省去对大部分数据点的搜索,
    从而减少搜索的计算量。
    """
    def __init__(self, dataset):
        self.root = self._create(dataset)

    # 递归的方式创建 kdNode
    def _create(self, dataset, depth=0):
        if len(dataset):
            # m 样本数， n 数据维度
            m, n = np.shape(dataset)
            # index 的中位数
            mid_index = m // 2
            # axis 分割数据集的维度
            axis = depth % n
            # 在axis维上对数据进行排序
            sorted_dataset = self._sort(dataset, axis)
            # 递归的创建节点
            node = KdNode(sorted_dataset[mid_index])
            left_dataset = sorted_dataset[:mid_index]
            right_dataset = sorted_dataset[mid_index+1:]
            node.left = self._create(left_dataset, depth+1)
            node.right = self._create(right_dataset, depth+1)
            return node

    # 在axis维上对数据进行排序
    def _sort(self, dataset, axis):
        m, n = np.shape(dataset)
        index = np.argsort(dataset[:, axis])
        sort_dataset = dataset[index]
        return sort_dataset

    # 递归的打印节点
    def print_tree(self, node=None):
        if not node:
            node = self.root
        print(node.dom_element)
        if node.left:
            self.print_tree(node.left)
        if node.right:
            self.print_tree(node.right)

    # 搜索样本中距离 x 最近的点
    def search(self, x):
        # 初始化
        self.nearest_point = None
        self.nearest_value = float('inf')
        # 搜索
        self._travel(self.root)
        return self.nearest_point

    def _travel(self, node, depth=0):
        if node != None:
            n = len(x)
            axis = depth % n
            # 递归的搜索距离x 最近的叶子节点
            if x[axis] < node.dom_element[axis]:
                self._travel(node.left, depth+1)
            else:
                self._travel(node.right, depth+1)

            # 由叶节点向上回溯， 寻找距离 X 最近的样本
            dist_node_x = self._dist(x, node.dom_element)
            if self.nearest_point is None:
                self.nearest_point = node.dom_element
                self.nearest_value = dist_node_x
            elif (self.nearest_value > dist_node_x):
                self.nearest_point = node.dom_element
                self.nearest_value = dist_node_x

            print(node.dom_element, depth, self.nearest_value, node.dom_element[axis], x[axis])
            # 判断是否需要节点的其他区域寻找
            if (abs(x[axis]- node.dom_element[axis]) < self.nearest_value):
                if x[axis] < node.dom_element[axis]:
                    self._travel(node.right, depth+1)
                else:
                    self._travel(node.left, depth+1)

    def _dist(self, x1, x2):
        return np.sqrt(((np.array(x1) - np.array(x2)) **2).sum())

if __name__ == '__main__':
    dataset = [[2, 3],
            [5, 4],
            [9, 6],
            [4, 7],
            [8, 1],
            [7, 2]]
    dataset = np.array(dataset)
    kdtree = KdTree(dataset)
    kdtree.print_tree()
    x = np.array([3, 4.5])
    nearest_point = kdtree.search(x)
    print(nearest_point)
