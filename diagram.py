import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import listdir
from sklearn.decomposition import *
from sklearn.manifold import LocallyLinearEmbedding

def printLoss():
    loss = np.load("loss_64.npy")
    epoch = [x*10 for x in range(1,101)]
    print(epoch)
    fig,ax = plt.subplots() #定義一個圖像窗口
    ax.plot(epoch, loss)
    ax.set(xlabel="epoch", ylabel="loss")
    plt.savefig("loss_em64.jpg")
    plt.show()

def printScatter(id2label):
    article_emb_w = tf.get_variable("article_emb_w", [1816, 64],
                                         initializer=tf.random_normal_initializer(0, 0.1))
    keyword_emb_w = tf.get_variable("keyword_emb_w", [4200, 64],
                                         initializer=tf.random_normal_initializer(0, 0.1))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "model1000.ckpt")
        artical = sess.run(article_emb_w)

    # PCA
    data = []
    # for i in artical:
    #     data.append(i.reshape(1,64))
    pca = PCA(n_components=2)
    reduced_data_pca = pca.fit_transform(artical)
    print(reduced_data_pca)

    AIDS_X=[]
    AIDS_Y=[]
    dengue_X=[]
    dengue_Y=[]
    Ebola_X=[]
    Ebola_Y=[]
    for i in range(len(reduced_data_pca)-1):
        # print(artical[i])
        if id2label[str(i)]=='AIDS':
            AIDS_X.append(reduced_data_pca[i][0])
            AIDS_Y.append(reduced_data_pca[i][1])
        elif  id2label[str(i)]=='dengue':
            dengue_X.append(reduced_data_pca[i][0])
            dengue_Y.append(reduced_data_pca[i][1])
        else:
            Ebola_X.append(reduced_data_pca[i][0])
            Ebola_Y.append(reduced_data_pca[i][1])

    plt.scatter(AIDS_X, AIDS_Y, s=30, c='red', marker='o', alpha=0.5, label='AIDS')
    plt.scatter(dengue_X, dengue_Y, s=30, c='blue', marker='x', alpha=0.5, label='Dengue')
    plt.scatter(Ebola_X, Ebola_Y, s=30, c='green', marker='s', alpha=0.5, label='Ebola')

    plt.xlabel('variables x')
    plt.ylabel('variables y')
    plt.legend(loc='upper right')  # 这个必须有，没有你试试看
    plt.show()  # 这个可以没有
def productLable():
    pubmedid2id = {}
    pubmedid2label = {}
    with open("data/articleTable.txt",'r') as rf:
        for i in rf.readlines():
            pubmedid2id[i.split()[1]] = i.split()[0]


    files = listdir("data/text_pre/")[1:]
    for f in files:
        pubmedid2label[f.split(".xml")[1]] = f.split(".xml")[0]


    id2label = {}
    for i in pubmedid2id:
        id2label[pubmedid2id[i]] = pubmedid2label[i]
    print(id2label)
    return id2label


printScatter(productLable())
