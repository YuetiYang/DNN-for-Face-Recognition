import glob
import numpy as np
from torch.autograd import Variable
import torch
from myalexnet import MyAlexNet
from faces import get_dataset
from scipy.io import loadmat
import matplotlib.pyplot as plt

LABELS =['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
NUM_LABELS = 6
CONV4_DIM = 256 * 6 * 6

RESULTS_FILEPATH = "results/"

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

def get_train_alexnet(M, img_side_len=227):
    batch_xs = np.zeros((0, 3, img_side_len, img_side_len)).astype(np.float32)
    batch_y_s = np.zeros( (0, NUM_LABELS))
    
    train_k = ["train_"+i for i in LABELS]
    for k in range(NUM_LABELS):
        cur_data = M[train_k[k]]
        new_data = np.zeros((len(cur_data), 3, img_side_len, img_side_len)).astype(np.float32)
        for i in range(len(cur_data)):
            im = np.reshape(cur_data[i], (img_side_len, img_side_len, 3))
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)
            new_data[i,:,:,:] = im
        
        batch_xs = np.vstack((batch_xs, new_data))
        one_hot = np.zeros(NUM_LABELS)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[train_k[k]]), 1))))
    return batch_xs, batch_y_s 

def get_validation_alexnet(M, img_side_len=227):
    batch_xs = np.zeros((0, 3, img_side_len, img_side_len)).astype(np.float32)
    batch_y_s = np.zeros( (0, NUM_LABELS))
    
    validation_k = ["validation_"+i for i in LABELS]
    for k in range(NUM_LABELS):
        cur_data = M[validation_k[k]]
        new_data = np.zeros((len(cur_data), 3, img_side_len, img_side_len)).astype(np.float32)
        for i in range(len(cur_data)):
            im = np.reshape(cur_data[i], (img_side_len, img_side_len, 3))
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)
            new_data[i,:,:,:] = im
            
        batch_xs = np.vstack((batch_xs, new_data))
        one_hot = np.zeros(NUM_LABELS)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[validation_k[k]]), 1))))
    return batch_xs, batch_y_s 

def get_test_alexnet(M, img_side_len=227):
    batch_xs = np.zeros((0, 3, img_side_len, img_side_len)).astype(np.float32)
    batch_y_s = np.zeros( (0, NUM_LABELS))
    
    test_k = ["test_"+i for i in LABELS]
    for k in range(NUM_LABELS):
        cur_data = M[test_k[k]]
        new_data = np.zeros((len(cur_data), 3, img_side_len, img_side_len)).astype(np.float32)
        for i in range(len(cur_data)):
            im = np.reshape(cur_data[i], (img_side_len, img_side_len, 3))
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)
            new_data[i,:,:,:] = im
            
        batch_xs = np.vstack((batch_xs, new_data))
        one_hot = np.zeros(NUM_LABELS)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[test_k[k]]), 1))))
    return batch_xs, batch_y_s  
    

def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.normal_(0.0, 0.03)
        
def part10_helper(conv4_train_x, conv4_val_x, conv4_test_x, 
                  train_y_227, validation_y_227, test_y_227,
                  n_epochs=200, batch_size=32, 
                  dim_h=300, learning_rate=1e-3):
    torch.manual_seed(0)
    
    # initialize nn classifier
    classifier = torch.nn.Sequential(
        torch.nn.Linear(CONV4_DIM, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, NUM_LABELS),
    )
    classifier.apply(init_weights)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    
    def getAccuracy(X, Y, classifier, accLst):
        accLst.append(0)
        torch_X = Variable(torch.from_numpy(X), requires_grad=False).type(dtype_float)
        Y_pred = classifier(torch_X).data.numpy()
        accLst[-1] = np.mean(np.argmax(Y_pred, 1) == np.argmax(Y, 1))
        
    # Learning curves on training, validation and test data
    trainAcc, validateAcc, testAcc = [], [], []
    
    # train using an optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)    
    
    # train using mini-batches
    np.random.seed(0)

    for epoch in range(n_epochs):
        classifier.train()
        permutation = np.random.permutation(range(conv4_train_x.shape[0]))
    
        for i in range(0, conv4_train_x.shape[0], batch_size):
            train_batch_idx = permutation[i:i+batch_size] 
            batch_x = Variable(torch.from_numpy(conv4_train_x[train_batch_idx]), 
                               requires_grad=False).type(dtype_float)
            batch_y = Variable(torch.from_numpy(np.argmax(train_y_227[train_batch_idx], 1)), 
                               requires_grad=False).type(dtype_long)

            train_y_pred = classifier(batch_x)
            loss = loss_fn(train_y_pred, batch_y)
           
            classifier.zero_grad()  # Zero out the previous gradient computation
            loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to make a step


        classifier.eval()
        # ===============training accuracy===============
        getAccuracy(conv4_train_x, train_y_227, classifier, trainAcc)
        # ===============validation accuracy===============
        getAccuracy(conv4_val_x, validation_y_227, classifier, validateAcc)
        # ===============test accuracy===============
        getAccuracy(conv4_test_x, test_y_227, classifier, testAcc)

    return trainAcc, validateAcc, testAcc

    
    
def part10():
    if not glob.glob('faces_all_227.mat'):
        get_dataset(227)
    data_227 = loadmat('faces_all_227.mat')
    
    #Alexnet input: (n, 3, 227, 227)
    train_x_227, train_y_227 = get_train_alexnet(data_227, 227)
    validation_x_227, validation_y_227 = get_validation_alexnet(data_227, 227)
    test_x_227, test_y_227 = get_test_alexnet(data_227, 227)    

    # Extract the values of the activations of AlexNet in Conv4 layer
    model = MyAlexNet()
    conv4_train_x = np.zeros((0, CONV4_DIM))
    conv4_val_x = np.zeros((0, CONV4_DIM))
    conv4_test_x = np.zeros((0, CONV4_DIM))

    for i_train in train_x_227:
        im_tr = Variable(torch.from_numpy(i_train).unsqueeze_(0), requires_grad=False)
        im_tr = model.features(im_tr)
        im_tr = im_tr.view(im_tr.size(0), CONV4_DIM)
        conv4_train_x = np.vstack((conv4_train_x, im_tr.data.numpy()))
      

    for i_val in validation_x_227:
        im_v = Variable(torch.from_numpy(i_val).unsqueeze_(0), requires_grad=False)
        im_v = model.features(im_v)
        im_v = im_v.view(im_v.size(0), CONV4_DIM)
        conv4_val_x = np.vstack((conv4_val_x, im_v.data.numpy()))
        

    for i_test in test_x_227:
        im_t = Variable(torch.from_numpy(i_test).unsqueeze_(0), requires_grad=False)
        im_t = model.features(im_t)
        im_t = im_t.view(im_t.size(0), CONV4_DIM)
        conv4_test_x = np.vstack((conv4_test_x, im_t.data.numpy()))

    # learn a fully-connected neural network that takes in the activations 
    # of the units in the AlexNet layer as inputs
    trainAcc, validateAcc, testAcc = \
        part10_helper(conv4_train_x, conv4_val_x, conv4_test_x,
                      train_y_227, validation_y_227, test_y_227)
            
    print('Training accuracy: {}'.format(str(trainAcc[-1])))
    print('Validation accuracy: {}'.format(str(validateAcc[-1])))
    print('Test accuracy: {}'.format(str(testAcc[-1])))
    
    xAxis = range(0, 200)

    plt.plot(xAxis, trainAcc, label='Training')
    plt.plot(xAxis, validateAcc, label='Validation')
    plt.plot(xAxis, testAcc, label='Test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(RESULTS_FILEPATH + 'part10_learning.png')
    plt.clf()
    
    
if __name__ == "__main__":
    part10()