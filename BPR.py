import tensorflow as tf
import numpy as np
import random

class preprocessData():
    def __init__(self,batchSize):
        self.datapath = "data/"
        self.articleKeywords = self.readfile(self.datapath+"art2keyIDTable_windows.txt")
        self.bachSize=batchSize
        self.selfKeySize = 4200
        self.selfArticleSize = 1816

    def readfile(self,path):
        articleKeywords = {}
        with open(path,'r') as rf:
            for i in rf.readlines():
                temp = i.split()
                articleKeywords[temp[0]]=temp[1:]
        #print(articleKeywords)
        return  articleKeywords

    def generate_train_batch(self):
        trainBatch = []
        for i in range(self.bachSize):
            sampleA = int(random.sample(self.articleKeywords.keys(),1)[0])
            sampleK_pos = int(random.sample(self.articleKeywords[str(sampleA)],1)[0])
            negList = list(set(list(range(0,self.selfKeySize))).difference(set(list(self.articleKeywords[str(sampleA)]))))
            sampleK_neg = int(random.sample(negList,1)[0])
            trainBatch.append([sampleA,sampleK_pos,sampleK_neg])
        #print(trainBatch)

        return np.array(trainBatch)

class MF():
    def __init__(self,data,K):
        self.data = data
        self.K = K
    def creatPlacrholder(self):
        with tf.name_scope("input_data"):
            self.u = tf.placeholder(tf.int32, [None])
            self.i = tf.placeholder(tf.int32, [None])
            self.j = tf.placeholder(tf.int32, [None])

    def createVariables(self):
        with tf.name_scope("embedding"):
            self.article_emb_w = tf.get_variable("article_emb_w", [self.data.selfArticleSize , self.K],
                                         initializer=tf.random_normal_initializer(0, 0.1))
            self.keyword_emb_w = tf.get_variable("keyword_emb_w", [self.data.selfKeySize , self.K],
                                         initializer=tf.random_normal_initializer(0, 0.1))

    def createInference(self):
        with tf.name_scope("inference"):
            self.u_emb = tf.nn.embedding_lookup(self.article_emb_w, self.u)
            self.i_emb = tf.nn.embedding_lookup(self.keyword_emb_w, self.i)
            self.j_emb = tf.nn.embedding_lookup(self.keyword_emb_w, self.j)

            self.x = tf.reduce_sum(tf.multiply(self.u_emb, (self.i_emb - self.j_emb)), 1, keep_dims=True)

    def creatLoss(self):
        with tf.name_scope("loss"):
            self.mf_auc = tf.reduce_mean(tf.to_float(self.x > 0))

            l2_norm = tf.add_n([
                tf.reduce_sum(tf.multiply(self.u_emb, self.u_emb)),
                tf.reduce_sum(tf.multiply(self.i_emb, self.i_emb)),
                tf.reduce_sum(tf.multiply(self.j_emb, self.j_emb))
            ])

            regulation_rate = 0.0001
            self.bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(self.x)))

    def creatOptimizer(self):
        with tf.name_scope("optimizer"):
            self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.bprloss)

    def buildGraph(self):
        self.creatPlacrholder()
        self.createVariables()
        self.createInference()
        self.creatLoss()
        self.creatOptimizer()


with tf.Graph().as_default(), tf.Session() as session:
    data = preprocessData(20)
    mf = MF(data, 2)
    mf.buildGraph()
    session.run(tf.initialize_all_variables())

    loss = []
    for epoch in range(1, 1001):
        _batch_bprloss = 0
        for k in range(1, 5000):  # uniform samples from training set
            uij = data.generate_train_batch()
            _bprloss, _train_op = session.run([mf.bprloss, mf.train_op],feed_dict={mf.u:uij[:,0], mf.i:uij[:,1], mf.j:uij[:,2]})
            _batch_bprloss += _bprloss

        print("epoch: ", epoch)
        print("bpr_loss: ", _batch_bprloss / k)
        print("_train_op")
        if epoch % 10 == 0:
            loss.append(_batch_bprloss / k)
            saver = tf.train.Saver({"article_emb_w": mf.article_emb_w,"keyword_emb_w": mf.keyword_emb_w})
            save_path = saver.save(session, "model/model"+str(epoch)+".ckpt")
        user_count = 0
        _auc_sum = 0.0

    np.save("loss.npy",np.array(loss))

