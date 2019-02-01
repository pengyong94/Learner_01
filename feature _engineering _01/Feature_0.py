# -*- coding: utf-8 -*-
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import numpy as np
import jieba


dict_data = [{"city": "北京", "temperature": 100}, {"city": "上海", "temperature": 60},
              {"city": "深圳", "temperature": 50}]

txt_data = ["life is short, i like python", "life is too long, i dislike python"]

cont_1 = "隔壁老王要去10公里外的一个地方办事，他可以选择走路，骑自行车或者开车，并花费了一定时间到达目的地。"
cont_2 = "老王早上起床的时候觉得精神不错，想锻炼下身体，决定跑步过去；也可能老王想做个文艺青年试试最近流行的共享单车，决定骑车过去；也可能老王想炫个富，决定开车过去。"
cont_3 = "老王决定步行过去，那么很大可能10公里的距离大约需要两个小时；较小可能是老王平时坚持锻炼，跑步过去用了一个小时；更小可能是老王是个猛人，40分钟就到了。老王决定骑车过去，很可能一个小时就能到；"

mm_data = [[34, 45, 23, 56], [12, 34, 56, 67], [87, 98, 45,68 ]]

def dictver():
    # 实例化DictVectorizer类
    dict = DictVectorizer(sparse=False)   # 不采用sparse矩阵输出
    #调用fit_transform进行拟合
    data= dict.fit_transform(dict_data)
    print(dict.get_feature_names())   # 获取各个特征的名字
    print(dict.inverse_transform(data))  # 反向还原数据信息
    print(data)
    return None

def countvec():
    # 对文本进行特征值化
    cv = CountVectorizer()
    data = cv.fit_transform(txt_data)
    print(cv.get_feature_names())
    print(data)
    print(data.toarray())    # 将数据列表转化为数组array

    return None

def cutword():
    con1 = list(jieba.cut(cont_1))
    con2 = list(jieba.cut(cont_2))
    con3 = list(jieba.cut(cont_3))

    # 把切分好的字符列表转换成字符串
    c1 = "".join(con1)
    c2 = "".join(con2)
    c3 = "".join(con3)

    return c1, c2, c3

def hanzi():
    """中文特征值化"""
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    # 实例化
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())


def hanzi_tf():
    """中文特征值化"""
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    # 实例化
    tf = TfidfVectorizer()
    data = tf.fit_transform([c1, c2, c3])
    print(tf.get_feature_names())
    print(data.toarray())


def mm():
    """归一化处理"""
    mm = MinMaxScaler(feature_range=(2, 3 ))
    data  = mm.fit_transform(mm_data)
    print(data)

def stand():
    """标准化缩放"""
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3],[2, 4, 2],[4., 6., -1.]])
    print(data)


def im():
    """缺失值处理"""
    # 缺失值NaN,nan
    im = Imputer(missing_values="NaN", strategy="mean",axis=0)
    data = im.fit_transform([[1, 2],[np.nan,3], [7, 6]])
    print(data)


def Var():
    """特征选择——删除低方差的特征"""
    var = VarianceThreshold(threshold=1.0)
    data = var.fit_transform([[0,2,0,3],[0,1,4,3],[0,1,1,3]])
    print(data)


def pca():
    """主成分分析进行特征降维"""
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2,8,4,5], [6,3,0,8], [5,4,9,1]])
    print(data)



if __name__ == "__main__":
    # dictver()      # 字典类型数据特征化
    # countvec()     # 文本类型数据特征化
    # hanzi()        # 中文文本类型数据特征化
    # mm()             # 归一化缩放
    # stand()          #  标准化缩放
    # im()          # 缺失值处理
    # Var()
    pca()




