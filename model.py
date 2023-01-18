import tensorflow as tf
import os
import numpy as np
       
class workerModel:
    def __init__(self):
        self.learning_rate = 0.01
        self.num_steps_self = 250
        self.num_steps = 250
        self.batch_size = 64
        self.display_step = 250
        self.num_input = 784
        self.num_classes = 10
        #self.checkpoint_dir = args.checkpoint_dir
        #self.checkpoint_file = "workermodel"
        #self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + ".ckpt")
        #self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32]), name="%s" % ("wc1")),
            'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64]), name="%s" % ("wc2")),
            'wd1': tf.Variable(tf.random_normal([12*12*64, 128]), name="%s" % ("wd1")),
            'out': tf.Variable(tf.random_normal([128, self.num_classes]), name="%s" % ("out"))
        }
        self.build_model()
        self.saver = tf.train.Saver()
   
    def conv2d(self, x, W, strides=1):
        with tf.name_scope("conv2d"), tf.variable_scope("conv2d"):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
            return tf.nn.relu(x)

    def maxpool2d(self, x, k):
        with tf.name_scope("maxpool2d"), tf.variable_scope("maxpool2d"):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1],
                                  padding='SAME')
    
    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s" % ("xinput"))
        self.Y = tf.placeholder(tf.int32, [None, self.num_classes], name="%s" % ("xinput"))
        self.co_portion = tf.placeholder(tf.float32)
        self.co_logits = tf.placeholder(tf.float32, [self.batch_size, self.num_classes], name='codistillation')
        #self.keep_prob = tf.placeholder(tf.float32)
        #self.keep_prob_2 = tf.placeholder(tf.float32)

        with tf.name_scope("convmaxpool"), tf.variable_scope("convmaxpool"):
            x1 = tf.reshape(self.X, [-1,28,28,1])
            conv1 = self.conv2d(x1, self.weights['wc1'])
            conv1 = tf.nn.relu(conv1)
            conv2 = self.conv2d(conv1, self.weights['wc2'])
            conv2 = tf.nn.relu(conv2)
            pool1 = self.maxpool2d(conv2, k=2)
            #pool1_drop = tf.nn.dropout(pool1, self.keep_prob_1)   

        with tf.name_scope("fclayer"), tf.variable_scope("fclayer"):
            fc1 = tf.reshape(pool1, [-1, self.weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.matmul(fc1, self.weights['wd1'])
            fc1 = tf.nn.relu(fc1)
            #fc1_drop = tf.nn.dropout(fc1, self.keep_prob)
            self.logits = tf.matmul(fc1, self.weights['out'])

        with tf.name_scope("prediction"), tf.variable_scope("prediction"):
            self.prediction = tf.nn.softmax(self.logits)
            self.co_prediction = self.co_logits
            #self.avg_logits = tf.add(tf.multiply(self.logits, self.portion), tf.multiply(self.co_logits, self.co_portion))
            self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        with tf.name_scope("optimization"), tf.variable_scope("optimization"):
            self.loss_op_standard = tf.clip_by_value(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y)),1e-10, 1e+15)
            
            self.loss_co_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.co_prediction))
            self.total_loss = tf.add(self.loss_op_standard, tf.multiply(self.loss_co_op, self.co_portion))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss)

        with tf.name_scope("summarization"), tf.variable_scope("summarization"):
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)
            
    def start_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def close_session(self):
        self.sess.close()

    def train(self, dataset, target, result):
        print("\n")
        print("Starting Training")
        max_accuracy = 0
        train_data_x, train_data_y = dataset.get_train_data()

        # IID Setting
        if target == 0:
            #train_data_x=np.load('data_x_1.npy')
            #train_data_y=np.load('data_y_1.npy')
            train_data_x=train_data_x[0:499]
            train_data_y=train_data_y[0:499]
        elif target == 1:
            #train_data_x=np.load('data_x_2.npy')
            #train_data_y=np.load('data_y_2.npy')
            train_data_x=train_data_x[500:999]
            train_data_y=train_data_y[500:999]
        elif target == 2:
            #train_data_x=np.load('data_x_3.npy')
            #train_data_y=np.load('data_y_3.npy')
            train_data_x=train_data_x[1000:1499]
            train_data_y=train_data_y[1000:1499]            
        elif target == 3:
            #train_data_x=np.load('data_x_4.npy')
            #train_data_y=np.load('data_y_4.npy')
            train_data_x=train_data_x[1500:1999]
            train_data_y=train_data_y[1500:1999]
        elif target == 4:
            #train_data_x=np.load('data_x_5.npy')
            #train_data_y=np.load('data_y_5.npy')
            train_data_x=train_data_x[2000:2499]
            train_data_y=train_data_y[2000:2499]
        elif target == 5:
            #train_data_x=np.load('data_x_6.npy')
            #train_data_y=np.load('data_y_6.npy')
            train_data_x=train_data_x[2500:2999]
            train_data_y=train_data_y[2500:2999]
        elif target == 6:
            #train_data_x=np.load('data_x_7.npy')
            #train_data_y=np.load('data_y_7.npy')
            train_data_x=train_data_x[3000:3499]
            train_data_y=train_data_y[3000:3499]
        elif target == 7:
            #train_data_x=np.load('data_x_8.npy')
            #train_data_y=np.load('data_y_8.npy')
            train_data_x=train_data_x[3500:3999]
            train_data_y=train_data_y[3500:3999]
        elif target == 8:
            #train_data_x=np.load('data_x_9.npy')
            #train_data_y=np.load('data_y_9.npy')
            train_data_x=train_data_x[4000:4499]
            train_data_y=train_data_y[4000:4499]
        elif target == 9:
            #train_data_x=np.load('data_x_10.npy')
            #train_data_y=np.load('data_y_10.npy')
            train_data_x=train_data_x[4500:4999]
            train_data_y=train_data_y[4500:4999]            
        #train_data_x=train_data_x[0:500]
        #train_data_y=train_data_y[0:500]
        index = np.array(range(len(train_data_x)))
                
        def dev_step():
            temp_loss = 0;
            temp_acc = 0;
            validation_x, validation_y = dataset.get_validation_data()
            for iter in range(39):
                temp_validation_x=validation_x[iter*self.batch_size :iter*self.batch_size + self.batch_size]
                temp_validation_y=validation_y[iter*self.batch_size :iter*self.batch_size + self.batch_size]
                temp_loss += self.sess.run(self.total_loss, feed_dict={self.X: temp_validation_x, self.Y: temp_validation_y,  self.co_portion: 0, self.co_logits:np.zeros([self.batch_size,self.num_classes])})  
                temp_acc += self.sess.run(self.accuracy, feed_dict={self.X: temp_validation_x, self.Y: temp_validation_y, self.co_portion: 0, self.co_logits:np.zeros([self.batch_size,self.num_classes])})
            else:
                loss = temp_loss / 39;
                acc = temp_acc / 39;
            print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                loss)  + ", Validation Accuracy= " + "{:.3f}".format(acc))

        if result == "0":
            print("Self Training Start")
            print("\n")
            for step in range(1, self.num_steps_self + 1):
                np.random.shuffle(index)
                batch_x=train_data_x[index[0:self.batch_size]]
                batch_y=train_data_y[index[0:self.batch_size]]                 
                probability, _ = self.sess.run([self.logits, self.train_op], feed_dict={self.X: batch_x, self.Y: batch_y, self.co_portion: 0, self.co_logits:np.zeros([self.batch_size,self.num_classes])})            
                if (step % self.display_step) == 0 or step == 1:
                    dev_step()
                if step == self.num_steps_self:
                    prediction=np.ones((10,10))
                    prediction_prime=np.zeros((10,10))
                    div=np.zeros((self.num_classes))
                    for batch in range(int(len(train_data_x)/100)):
                        return_value=self.sess.run(self.prediction,feed_dict={self.X: train_data_x[100*batch:100*batch+100]})
                        for set in range(100):
                            for n in range(10):
                                if train_data_y[100*batch+set][n]==1:
                                    div[n]=div[n]+1
                                    prediction_prime[n][:]=prediction_prime[n][:]+return_value[set][:]
                    for k in range(10):
                        if div[k]!=0:
                            prediction[k][:]=prediction_prime[k][:]/div[k]
                    return(prediction)

        elif result == "1":
            print("Not considered")
            print("\n")
            for step in range(1, self.num_steps + 1):
                np.random.shuffle(index)
                batch_x=train_data_x[index[0:self.batch_size]]
                batch_y=train_data_y[index[0:self.batch_size]]                 
                probability, _ = self.sess.run([self.logits, self.train_op], feed_dict={self.X: batch_x, self.Y: batch_y, self.co_portion: 0, self.co_logits:np.zeros([self.batch_size,self.num_classes])})            
                if (step % self.display_step) == 0 or step == 1:
                    dev_step()
                if step == self.num_steps:
                    prediction=np.ones((10,10))
                    prediction_prime=np.zeros((10,10))
                    div=np.zeros((self.num_classes))
                    for batch in range(int(len(train_data_x)/100)):
                        return_value=self.sess.run(self.prediction,feed_dict={self.X: train_data_x[100*batch:100*batch+100]})
                        for set in range(100):
                            for n in range(10):
                                if train_data_y[100*batch+set][n]==1:
                                    div[n]=div[n]+1
                                    prediction_prime[n][:]=prediction_prime[n][:]+return_value[set][:]
                    for k in range(10):
                        if div[k]!=0:
                            prediction[k][:]=prediction_prime[k][:]/div[k]
                    return(prediction)
            
        else:
            print("Checkpoint")
            for step in range(1, self.num_steps + 1):
                np.random.shuffle(index)
                batch_x=train_data_x[index[0:self.batch_size]]
                batch_y=train_data_y[index[0:self.batch_size]]
                soft_targets=np.zeros((self.batch_size,self.num_classes))
                for l in range(1, self.batch_size+1):
                    for m in range(1, self.num_classes+1):
                        if batch_y[l-1][m-1]==1:
                            soft_targets[l-1][:] = result[m-1][:];
                probability, _ = self.sess.run([self.logits, self.train_op], feed_dict={self.X: batch_x, self.Y: batch_y, self.co_portion: 0.01, self.co_logits:soft_targets})                      
                if (step % self.display_step) == 0 or step == 1:
                    dev_step()
                if step == self.num_steps:
                    prediction=np.ones((10,10))
                    prediction_prime=np.zeros((10,10))
                    div=np.zeros((self.num_classes))
                    for batch in range(int(len(train_data_x)/100)):
                        return_value=self.sess.run(self.prediction,feed_dict={self.X: train_data_x[100*batch:100*batch+100]})
                        for set in range(100):
                            for n in range(10):
                                if train_data_y[100*batch+set][n]==1:
                                    div[n]=div[n]+1
                                    prediction_prime[n][:]=prediction_prime[n][:]+return_value[set][:]
                    for k in range(10):
                        if div[k]!=0:
                            prediction[k][:]=prediction_prime[k][:]/div[k]
                    return(prediction)                   
        print("Optimization Finished!")

    def run_inference(self,dataset):
        sum = 0;
        test_images, test_labels = dataset.get_test_data()
        for step in range(78):
            temp_test_images=test_images[step*self.batch_size :step*self.batch_size + self.batch_size]
            temp_test_labels=test_labels[step*self.batch_size :step*self.batch_size + self.batch_size]         
            sum += self.sess.run(self.accuracy, feed_dict={self.X: temp_test_images, self.Y: temp_test_labels, self.co_portion: 0, self.co_logits:np.zeros([self.batch_size,self.num_classes])})
        else:
            avg = sum / 78;
            return avg
            #print("Testing Accuracy:", avg)
                                                                
    def run_inference_per_label(self,dataset):
        test_images, test_labels = dataset.get_test_data()
        for k in range(10):
            avg = 0
            summation = 0
            div = 0
            data_x = np.zeros((10000, self.num_input))
            data_y = np.zeros((10000, self.num_classes))
            for i in range(10000):
                if test_labels[i][k]==1:
                    data_x[div]=test_images[i]
                    data_y[div][k]=test_labels[i][k]
                    div=div+1
            for step in range(int(div/self.batch_size)):
                temp_test_images=data_x[step*self.batch_size:step*self.batch_size + self.batch_size]
                temp_test_labels=data_y[step*self.batch_size :step*self.batch_size + self.batch_size]
                summation += self.sess.run(self.accuracy, feed_dict={self.X: temp_test_images, self.Y: temp_test_labels})
            else:
                avg = summation/int(div/self.batch_size)
                print("Testing Accuracy per label :", avg)            