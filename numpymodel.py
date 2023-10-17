#import tensorflow-gpu as tf
import numpy as np
import os
import collections

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ElmanRNN:
    def __init__ (self, hidden_size, input_size, output_size, lc):
        self.wx = np.random.rand(hidden_size, input_size) * 2 - 1
        self.wy = np.random.rand(output_size, hidden_size) * 2 - 1
        self.b = np.random.rand(hidden_size, 1) * 2 - 1
        self.u = np.random.rand(1, hidden_size) * 2 - 1
        self.ht_old = np.zeros((hidden_size, 1), dtype = np.float64)
        self.ht_new = np.zeros((hidden_size, 1), dtype = np.float64)
        self.lc = lc
        self.del_wy = np.zeros((output_size, hidden_size))
        self.del_wx = np.zeros((hidden_size, input_size))
        self.del_b = np.zeros((hidden_size, 1))
        self.del_u = np.zeros((hidden_size, 1))
        self.gamma = 0.3 * lc

    def train_inst (self, x, y):
        x_comp = np.dot(self.wx, x)
        h_comp = np.transpose(self.u) * self.ht_new
        self.ht_old = self.ht_new
        comp = x_comp + h_comp + self.b
        self.ht_new = np.tanh(comp)
        y_pred = sigmoid(np.dot(self.wy, self.ht_new))

        d_out = (y - y_pred) * (1 - y_pred) * (y_pred)
        del_wy = np.dot(d_out, np.transpose(self.ht_new)) * self.lc
        self.wy += (self.gamma * self.del_wy) + del_wy
        self.del_wy = del_wy

        d_h = np.dot(np.transpose(self.wy), d_out) * (1 - self.ht_new ** 2)
        del_wx = np.dot(d_h, np.transpose(x)) * self.lc
        del_b = d_h * self.lc
        del_u = (d_h * self.ht_old)
        del_u = del_u * self.lc
        self.u += ((self.del_u.T) * self.gamma) + del_u.T
        self.wx += (self.del_wx * self.gamma) + del_wx
        self.b += (self.del_b * self.gamma) + del_b
        self.del_wx = del_wx
        self.del_b = del_b
        self.del_u = del_u

        #print(np.sum(np.absolute(del_u)))
        #print(np.sum(np.absolute(del_wy)))
        #print(np.sum(np.absolute(del_b)))

    def pred_inst (self, x, ht):
        x_comp = np.dot(self.wx, x)
        h_comp = np.transpose(self.u) * ht
        ht = np.tanh(x_comp + h_comp + self.b)
        y_pred = sigmoid(np.dot(self.wy, ht))
        return np.argmax(y_pred), ht

def convert_to_list(data, dic):
    return [dic[c] for c in data]

def get_inp():
    train_path = os.path.join("./", "ptb.train.txt")
    voc = collections.Counter(open(train_path, 'r').read().replace("<unk>", ""))
    counter = 0
    dic = {}
    for key in voc:
        dic[key] = counter
        counter = counter + 1
    #print(dic)
    train_data = convert_to_list(open(train_path, 'r').read().replace("<unk>", ""), dic)
    valid_path = os.path.join("./", "ptb.valid.txt")
    valid_data = convert_to_list(open(valid_path, 'r').read().replace("<unk>", ""), dic)
    test_path = os.path.join("./", "ptb.test.txt")
    test_data = convert_to_list(open(test_path, 'r').read().replace("<unk>", ""), dic)
    return train_data, valid_data, test_data, dic
    #print(ord("\n"))
    #print(train_data)

def runElmanRNN():
    train, valid, test, dic = get_inp()
    dic_len = len(dic)
    test_len = len(test)
    Elman = ElmanRNN(32, dic_len, dic_len, 0.1)
    batches = np.array_split(np.array(train), 100)
    cnt = 0
    print("ElmanRNN")
    print("No of batches = " + str(len(batches)))
    for (ind, batch) in enumerate(batches):
        print("Batch - " + str(ind))
        #print(Elman.wy)
        for i,j in zip(batch[:-1], batch[1:]):
            x = np.zeros((dic_len, 1))
            x[i][0] = 1
            y = np.zeros((dic_len, 1))
            y[j][0] = 1
            Elman.train_inst(x, y)
        #print(Elman.wy)
        Elman.lc -= Elman.lc * 0.02
        Elman.gamma -= Elman.gamma * 0.02
        print("Batch done")
        if (ind % 10) == 0:
            print("Checking")
            cnt = 0
            ht_temp = np.zeros((32, 1), dtype = np.float64)
        #    st = np.random.randint(test_len - 1024)
            for i,j in zip(test[:-1], test[1:]):
                x = np.zeros((dic_len, 1))
                x[i][0] = 1
                temp, ht_temp = Elman.pred_inst(x, ht_temp)
                if temp != j:
                    cnt += 1
            print(str(cnt) + " out of " + str(len(test)) + " are incorrect!")
    cnt = 0
    ht_temp = np.zeros((32, 1), dtype = np.float64)
#    st = np.random.randint(test_len - 1024)
    for i,j in zip(test[:-1], test[1:]):
        x = np.zeros((dic_len, 1))
        x[i][0] = 1
        temp, ht_temp = Elman.pred_inst(x, ht_temp)
        if temp != j:
            cnt += 1
    print(str(cnt) + " out of " + str(len(test)) + " are incorrect!")

#runElmanRNN()

def thres(x, e):
    if x < e and x > 1 - e:
        return 1.0
    return 0.0

class VCRNN:
    def __init__ (self, hidden_size, input_size, output_size, lc, sh, ep):
        self.wx = np.random.rand(hidden_size, input_size) * 2 - 1
        self.wy = np.random.rand(output_size, hidden_size) * 2 - 1
        self.b = np.random.rand(hidden_size, 1) * 2 - 1
        self.u = np.random.rand(1, hidden_size) * 2 - 1
        self.ht_old = np.zeros((hidden_size, 1), dtype = np.float64)
        self.ht_new = np.zeros((hidden_size, 1), dtype = np.float64)
        self.lc = lc
        self.param_u = np.random.rand(1, hidden_size) * 2 - 1
        self.param_v = np.random.rand(1, input_size) * 2 - 1
        self.param_b = np.random.rand() * 2 - 1
        self.ep = ep
        self.sh = sh
        self.del_wy = np.zeros((output_size, hidden_size))
        self.del_wx = np.zeros((hidden_size, input_size))
        self.del_b = np.zeros((hidden_size, 1))
        self.del_u = np.zeros((hidden_size, 1))
        self.gamma = 0.3 * lc

    def train_inst (self, x, y):
        mt = 1.0/(1.0+np.exp(-(np.dot(self.param_u, self.ht_new) + np.dot(self.param_v, x)+ self.param_b)))
        etx = sigmoid(np.array([self.sh * (mt*len(x) - ind) for ind in range(0, len(x))]))
        htx = sigmoid(np.array([self.sh * (mt*len(self.ht_new) - ind) for ind in range(0, len(self.ht_new))]))
        etx = np.array([[thres(a, self.ep) for a in etx]]).T
        htx = np.array([[thres(a, self.ep) for a in htx]]).T
        x = etx * x

        x_comp = np.dot(self.wx, x)
        self.ht_old = self.ht_new
        self.ht_new = htx * self.ht_new
        h_comp = np.transpose(self.u) * self.ht_new
        self.ht_new = (np.tanh(x_comp + h_comp + self.b)) * htx + (1 - htx) * self.ht_old
        y_pred = sigmoid(np.dot(self.wy, self.ht_new))

        d_out = (y - y_pred) * (1 - y_pred) * (y_pred)
        del_wy = np.dot(d_out, np.transpose(self.ht_new)) * self.lc
        self.wy += (self.gamma * self.del_wy) + del_wy
        self.del_wy = del_wy

        d_h = np.dot(np.transpose(self.wy), d_out) * (1 - self.ht_new ** 2)
        del_wx = np.dot(d_h, np.transpose(x)) * self.lc
        del_b = d_h * self.lc
        del_u = (d_h * self.ht_old)
        del_u = del_u * self.lc
        self.u += ((self.del_u.T) * self.gamma) + del_u.T
        self.wx += (self.del_wx * self.gamma) + del_wx
        self.b += (self.del_b * self.gamma) + del_b
        self.del_wx = del_wx
        self.del_b = del_b
        self.del_u = del_u

    def pred_inst (self, x, ht2):
        mt = 1.0/(1.0+np.exp(-(np.dot(self.param_u, ht2) + np.dot(self.param_v, x)+ self.param_b)))
        etx = sigmoid(np.array([self.sh * (mt*len(x) - ind) for ind in range(0, len(x))]))
        htx = sigmoid(np.array([self.sh * (mt*len(ht2) - ind) for ind in range(0, len(ht2))]))
        etx = np.array([[thres(a, self.ep) for a in etx]]).T
        htx = np.array([[thres(a, self.ep) for a in htx]]).T
        #print("start")
        x = etx * x
        ht = htx * ht2
        x_comp = np.dot(self.wx, x)
        h_comp = np.transpose(self.u) * ht
        ht = (np.tanh(x_comp + h_comp + self.b)) * htx + (1-htx) * ht2
        y_pred = sigmoid(np.dot(self.wy, ht))
        return np.argmax(y_pred), ht

def runVCRNN():
    train, valid, test, dic = get_inp()
    dic_len = len(dic)
    test_len = len(test)
    batches = np.array_split(np.array(train), 100)
    vc = VCRNN(32, dic_len, dic_len, 0.1, 0.1, 0.8)
    cnt = 0
    print("VCRNN")
    print("No of batches = " + str(len(batches)))
    for (ind, batch) in enumerate(batches):
        print("Batch - " + str(ind))
        for i,j in zip(batch[:-1], batch[1:]):
            x = np.zeros((dic_len, 1))
            x[i][0] = 1
            y = np.zeros((dic_len, 1))
            y[j][0] = 1
            vc.train_inst(x, y)
        print("Batch done")
        vc.lc -= vc.lc * 0.02
        vc.gamma -= vc.gamma * 0.02
        if vc.sh < 1:
            vc.sh += 0.1
        if (ind % 10) == 0:
            print("Checking")
            cnt = 0
            ht_temp = np.zeros((32, 1), dtype = np.float64)
        #    st = np.random.randint(test_len - 1024)
            for i,j in zip(test[:-1], test[1:]):
                x = np.zeros((dic_len, 1))
                x[i][0] = 1
                temp, ht_temp = vc.pred_inst(x, ht_temp)
                if temp != j:
                    cnt += 1
            print(str(cnt) + " out of " + str(len(test)) + " are incorrect!")
    cnt = 0
    ht_temp = np.zeros((32, 1), dtype = np.float64)
#    st = np.random.randint(test_len - 1024)
    for i,j in zip(test[:-1], test[1:]):
        x = np.zeros((dic_len, 1))
        x[i][0] = 1
        temp, ht_temp = vc.pred_inst(x, ht_temp)
        if temp != j:
            cnt += 1
    print(str(cnt) + " out of " + str(len(test)) + " are incorrect!")

runVCRNN()
