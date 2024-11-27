import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
class forget_gate:
    def __init__(self):
        self.w_f_input2sum = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.w_f_shortmen2sum = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.b_f = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.variables = [self.w_f_input2sum,self.w_f_shortmen2sum,self.b_f]
    def forget_calc(self,inputdata,short_memory):
        return tf.nn.sigmoid(short_memory*self.w_f_shortmen2sum+inputdata*self.w_f_input2sum+self.b_f)

class input_gate:
    def __init__(self):
        self.w_i_shortmen2percentage_sum = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.w_i_input2percentage_sum = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.b_i_percentage = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.w_i_shortmen2sum = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.w_i_input2sum = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.b_i = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.variables = [self.w_i_shortmen2percentage_sum,self.w_i_input2percentage_sum,self.b_i_percentage,self.w_i_shortmen2sum,self.w_i_input2sum,self.b_i]
    def input_calc(self,inputdata,short_memory):
        return tf.nn.sigmoid(short_memory*self.w_i_shortmen2percentage_sum+inputdata*self.w_i_input2percentage_sum+self.b_i_percentage)*tf.nn.tanh(short_memory * self.w_i_shortmen2sum + inputdata * self.w_i_input2sum + self.b_i)

class output_gate:
    def __init__(self):
        self.w_o_shortmen2sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.w_o_input2sum = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.b_o = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.variables=[self.w_o_shortmen2sum,self.w_o_input2sum,self.b_o]
    def output_calc(self,inputdata,short_memory,long_memory):
        return tf.nn.sigmoid(short_memory*self.w_o_shortmen2sum+inputdata*self.w_o_input2sum+self.b_o) * tf.nn.tanh(long_memory)


class LSTM():
    def __init__(self):
        self.short_mem = 0
        self.long_mem = 0
        self.Forget_gate = forget_gate()
        self.Input_gate = input_gate()
        self.Output_gate = output_gate()
        self.variables = self.Forget_gate.variables + self.Input_gate.variables + self.Output_gate.variables

    def fit(self,input_train_x,input_train_y,num_variables):
        ''''
        input_train_x must be a [variables]*time_steps numpy array
        a stock data example:
        date  |  open    close   highest
        day1  |  12      20      34
        day2  |  20      12      24
        :
        :
        day40 |  50      65      84

        input_train_y must be a list of length=time_steps
        num_variables is the number of variables
        '''
        self.num_variables = num_variables
        if type(input_train_y) != np.ndarray:
            self.input_train_y = np.array(input_train_y)
        else:
            self.input_train_y = input_train_y

        if type(input_train_x) != np.ndarray:
            self.input_train_x = np.array(input_train_x)
        else:
            self.input_train_x = input_train_x
        self.input_b = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.input_w = tf.Variable(np.random.randn(num_variables,10),dtype=tf.float32)
        self.input_w2 = tf.Variable(np.random.randn(10,1),dtype=tf.float32)
        self.input_b2 = tf.Variable(np.random.randn(),dtype=tf.float32)
        self.variables.append(self.input_w)
        self.variables.append(self.input_b)
        self.variables.append(self.input_w2)
        self.variables.append(self.input_b2)

    def standardize(self,type=None):
        '''
        default preprocessing would be doing nothing
        prividing two measures:z_score and Max_Min
        input_x must be a numpy array
        while y be a list or numpy array
        
        if type==none,self.standardize=false,meaning no modification was made
        else self.standardize records how dataset is modified i.e MaxMin or z_score
        '''
        if type==None:
            self.standardized = False
            return
        elif type=='MaxMin':
            self.standardized = 'MaxMin'
            self.train_y_max = max(self.input_train_y)
            self.train_y_min = min(self.input_train_y)
            self.input_train_y = (np.array(self.input_train_y,dtype=float)-self.train_y_min)/(self.train_y_max-self.train_y_min)
            self.train_x_max = []
            self.train_x_min = []
            for col in range(self.input_train_x.shape[1]):
                self.train_x_max.append(max(self.input_train_x[:,col]))
                self.train_x_min.append(min(self.input_train_x[:,col]))
                self.input_train_x[:,col] = (self.input_train_x[:,col]-self.train_x_min[col])/(self.train_x_max[col]-self.train_x_min[col])
        elif type=='z_score':
            self.standardized = 'z_score'
            self.train_y_mean = self.input_train_y.mean()
            self.train_y_sigma = np.std(self.input_train_y)
            self.input_train_y = (self.input_train_y - self.train_y_mean)/self.train_y_sigma
            self.train_x_mean = []
            self.train_x_sigma = []
            for col in range(self.num_variables):
                self.train_x_mean.append(self.input_train_x[:,col].mean())
                self.train_x_sigma.append(np.std(self.input_train_x[:,col]))
                self.input_train_x[:,col] = (self.input_train_x[:,col]-self.train_x_mean[col])/self.train_x_sigma[col]
        else:
            raise'measure not fund/supported!'

    def destandardize(self):
        if self.standardized == "MaxMin":
            self.input_train_y = self.input_train_y*(self.train_y_max-self.train_y_min)+self.train_y_min
            for col in range(self.num_variables):
                self.input_train_x[:, col] = self.input_train_x[:, col]*(self.train_x_max[col]-self.train_x_min[col])+self.train_x_min
        elif self.standardized == "z_score":
            self.input_train_y = self.input_train_y*self.train_y_sigma+self.train_y_mean
            for col in range(self.num_variables):
                self.input_train_x[:, col] = self.input_train_x[:,col]*self.train_x_sigma[col]+self.train_x_mean[col]

        return 1
    def show(self):
        print("-----------------------------input X-------------------------------")
        print(self.input_train_x)
        print("-----------------------------input Y-------------------------------")
        print(self.input_train_y)
    def train(self,epoch,lr):
        total_loss_keep = []
        with tf.GradientTape(persistent=True) as tape:
            for _ in range(epoch):
                self.short_mem = 0.1
                self.long_mem = 0.1
                loss = 0

                input_train_x = tf.nn.sigmoid((self.input_train_x @ self.input_w + self.input_b)@ self.input_w2+ self.input_b2)

                for X,Y in zip(input_train_x,self.input_train_y):
                    self.long_mem = self.long_mem * self.Forget_gate.forget_calc(X,self.short_mem) + self.Input_gate.input_calc(inputdata=X,short_memory=self.short_mem)
                    self.short_mem = self.Output_gate.output_calc(X,self.short_mem,self.long_mem)
                    loss+=(self.short_mem-Y)**2/2

                total_loss_keep.append(loss.numpy())
                gradients = tape.gradient(loss,self.variables)

                for i in range(len(self.variables)):
                    self.variables[i].assign_sub(gradients[i]*lr)

                if _%10==0:
                    print("epoch{}        loss:{}".format(_,loss))


        #save output
        self.train_answer = []
        self.short_mem = 0
        self.long_mem = 0
        input_train_x = tf.nn.sigmoid(
            (self.input_train_x @ self.input_w + self.input_b) @ self.input_w2 + self.input_b2)
        for X in input_train_x:
            self.long_mem = self.long_mem * self.Forget_gate.forget_calc(X,self.short_mem) + self.Input_gate.input_calc(inputdata=X, short_memory=self.short_mem)
            self.short_mem =self.Output_gate.output_calc(X, self.short_mem, self.long_mem)
            self.train_answer.append(self.short_mem)


        plt.title("Loss")
        plt.plot(np.arange(epoch),total_loss_keep)
        plt.show()

        return self.train_answer,self.variables