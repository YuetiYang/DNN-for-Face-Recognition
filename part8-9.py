from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import hashlib
from collections import OrderedDict
import glob
import math
import os
from scipy.misc import imread
from scipy.misc import imresize
from textwrap import wrap
from scipy.io import loadmat
from scipy.io import savemat
import requests

ACT =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
LABELS =['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
NUM_LABELS = 6

TRAINING_SET_SIZE = 70
VALIDATION_SET_SIZE = 20
TEST_SET_SIZE = 20

RESULTS_FILEPATH = "results/"

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

def load_data():
    faceset32 = OrderedDict()
    
    if not os.path.exists("uncropped/"):
        os.makedirs("uncropped/")
        
    for a in ACT:
        name = a.split()[1].lower()
        
        i = 0
        for line in open("faces_subset.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]

                try:
                    response = requests.get(line.split()[4], timeout=60)
                    if response.status_code == 200:
                        with open("uncropped/"+filename, 'wb') as f:
                            f.write(response.content)                    
                    
                except:
                    continue

                if os.path.isfile("uncropped/"+filename):
                    img_file = open("uncropped/"+filename, 'rb').read()
                    input_hash = line.split()[6]
                    img_hash = hashlib.sha256(img_file).hexdigest()

                    if input_hash != img_hash:
                        continue
                    
                    try:
                        img = imread("uncropped/"+filename)
    
                    except IOError:
                        print('Cannot identify image file ' + filename)
                        continue
    
                    x1 = int(line.split()[5].split(',')[0])
                    y1 = int(line.split()[5].split(',')[1])
                    x2 = int(line.split()[5].split(',')[2])
                    y2 = int(line.split()[5].split(',')[3])
                    
                    face = img[y1:y2, x1:x2]

                    face32 = imresize(np.copy(face), (32, 32, 3))

                    # save the face in faceset for later use
                    if filename not in faceset32:
                        faceset32[filename] = face32

                    i += 1
                    
                else:
                    continue
        
    np.savez_compressed("cropped_faces_32", faceset32)


def get_dataset(img_side_len=32):
    if not glob.glob('cropped_faces_32.npz'):
        load_data()
        
    faceset = np.load('cropped_faces_32.npz', encoding = 'latin1')['arr_0'].item()

    faces_all = {}
    shuffled_keys = list(faceset.keys())
    np.random.seed(1)
    np.random.shuffle(shuffled_keys)

    for a in ACT:
        name = a.split()[1].lower()
        
        count = 0
        for filename in shuffled_keys:
            if name in filename:
                count += 1
                
        # resize dataset sizes if total number of images is less than required
        if count < TRAINING_SET_SIZE + VALIDATION_SET_SIZE + TEST_SET_SIZE:
            training_set_prop = (TRAINING_SET_SIZE*1.0) / (TRAINING_SET_SIZE+VALIDATION_SET_SIZE+TEST_SET_SIZE)
            training_set_size = int(math.floor(training_set_prop*count))
            validation_set_size = int(math.floor((count-training_set_size)*0.5))
            test_set_size = count - training_set_size - validation_set_size
        else:
            training_set_size = TRAINING_SET_SIZE
            validation_set_size = VALIDATION_SET_SIZE
            test_set_size = TEST_SET_SIZE            
            
        # initialize datasets
        training_set = np.zeros((0, img_side_len*img_side_len*3))
        validation_set = np.zeros((0, img_side_len*img_side_len*3))
        test_set = np.zeros((0, img_side_len*img_side_len*3))        
                
        i = 0
        for filename in shuffled_keys:
            if name in filename \
               and i < training_set_size + validation_set_size + test_set_size:
                img = faceset.get(filename)
                # resize img to requested img_side_len
                if img_side_len != 32:
                    img = imresize(img, (img_side_len, img_side_len, 3))
                
                try:
                    face = img[:,:,:3]
                except IndexError:
                    # replicate the gray channel 3 times to get an RGB image
                    face = np.stack((img,)*3, -1)

                img_vector = np.reshape(np.ndarray.flatten(face), (1, img_side_len*img_side_len*3))
                # normalize between -1 and 1
                img_vector = 2.0 * (img_vector-img_vector.min())/(img_vector.max()-img_vector.min()) - 1.0
                
                if i < training_set_size:
                    training_set = np.vstack((training_set, img_vector))
                elif training_set_size <= i < training_set_size + validation_set_size:
                    validation_set = np.vstack((validation_set, img_vector))
                else:
                    test_set = np.vstack((test_set, img_vector))

                i += 1
    
        faces_all["train_"+name] = training_set
        faces_all["validation_"+name] = validation_set
        faces_all["test_"+name] = test_set
    
    output_filename = 'faces_all_{}.mat'.format(str(img_side_len))
    savemat(output_filename, faces_all, do_compression=True)


def get_test_faces(M, img_side_len=32):
    batch_xs = np.zeros((0, img_side_len*img_side_len*3))
    batch_y_s = np.zeros( (0, NUM_LABELS))
    
    test_k = ["test_"+i for i in LABELS]
    for k in range(NUM_LABELS):
        batch_xs = np.vstack((batch_xs, (np.array(M[test_k[k]])[:])))
        one_hot = np.zeros(NUM_LABELS)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[test_k[k]]), 1))))
    return batch_xs, batch_y_s

def get_train_faces(M, img_side_len=32):
    batch_xs = np.zeros((0, img_side_len*img_side_len*3))
    batch_y_s = np.zeros( (0, NUM_LABELS))
    
    train_k = ["train_"+i for i in LABELS]
    for k in range(NUM_LABELS):
        batch_xs = np.vstack((batch_xs, (np.array(M[train_k[k]])[:])))
        one_hot = np.zeros(NUM_LABELS)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[train_k[k]]), 1))))
    return batch_xs, batch_y_s

def get_validation_faces(M, img_side_len=32):
    batch_xs = np.zeros((0, img_side_len*img_side_len*3))
    batch_y_s = np.zeros( (0, NUM_LABELS))
    
    validation_k = ["validation_"+i for i in LABELS]
    for k in range(NUM_LABELS):
        batch_xs = np.vstack((batch_xs, (np.array(M[validation_k[k]])[:])))
        one_hot = np.zeros(NUM_LABELS)
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(M[validation_k[k]]), 1))))
    return batch_xs, batch_y_s


def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.data.normal_(0.0, 0.03)
 
def part8_helper(train_x, train_y, 
                 validation_x, validation_y, 
                 test_x, test_y, 
                 n_epochs=200, batch_size=32,
                 img_side_len=32, dim_h=300,
                 learning_rate=1e-3):
    torch.manual_seed(0)
    
    # initialize nn model
    model = torch.nn.Sequential(
        torch.nn.Linear(img_side_len*img_side_len*3, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, NUM_LABELS),
    )
    model.apply(init_weights)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    
    def getAccuracy(X, Y, model, accLst):
        accLst.append(0)
        torch_X = Variable(torch.from_numpy(X), requires_grad=False).type(dtype_float)
        Y_pred = model(torch_X).data.numpy()
        accLst[-1] = np.mean(np.argmax(Y_pred, 1) == np.argmax(Y, 1))
        
    # Learning curves on training, validation and test data
    trainAcc, validateAcc, testAcc = [], [], []
    
    # train using an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
    
    # train using mini-batches
    np.random.seed(0)
    for epoch in range(n_epochs):
        model.train()
        permutation = np.random.permutation(range(train_x.shape[0]))
        for i in range(0, train_x.shape[0], batch_size):
            train_batch_idx = permutation[i:i+batch_size] 
            batch_x = Variable(torch.from_numpy(train_x[train_batch_idx]), 
                               requires_grad=False).type(dtype_float)
            batch_y = Variable(torch.from_numpy(np.argmax(train_y[train_batch_idx], 1)), 
                               requires_grad=False).type(dtype_long)
            
            train_y_pred = model(batch_x)
            loss = loss_fn(train_y_pred, batch_y)
            #import pdb; pdb.set_trace()
           
            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to make a step


        model.eval()
        # ===============training accuracy===============
        getAccuracy(train_x, train_y, model, trainAcc)
        # ===============validation accuracy===============
        getAccuracy(validation_x, validation_y, model, validateAcc)
        # ===============test accuracy===============
        getAccuracy(test_x, test_y, model, testAcc)

    print('Parameters: resolution={}x{}, n_epochs={}, batch_size={}, dim_h={}, learning_rate={}'
            .format(img_side_len, img_side_len, n_epochs, 
                    batch_size, dim_h, learning_rate))
    print('Training accuracy: {}'.format(str(trainAcc[-1])))
    print('Validation accuracy: {}'.format(str(validateAcc[-1])))
    print('Test accuracy: {}'.format(str(testAcc[-1])))
    
    return trainAcc, validateAcc, testAcc, model[0].weight, model[0].bias
    

def part8_grid_search(use_best_param):
    # load data for resolution 32x32
    if not glob.glob('faces_all_32.mat'):
        get_dataset(32) 

    data_32 = loadmat('faces_all_32.mat')
    train_x_32, train_y_32 = get_train_faces(data_32, 32)
    validation_x_32, validation_y_32 = get_validation_faces(data_32, 32)
    test_x_32, test_y_32 = get_test_faces(data_32, 32)
    
    # Use best ones we already found!
    if use_best_param:
        max_trainAcc, max_validateAcc, max_testAcc, max_W0, max_b0 \
             = part8_helper(train_x_32, train_y_32, 
                            validation_x_32, validation_y_32, 
                            test_x_32, test_y_32)
        max_params = tuple([32,200,32,300,1e-3])
        
    # Let's run grid search to find the best parameters
    else:
        # Parameters 0: image resolution
        img_side_lens = [32]
        # Parameters 1: number of epochs
        n_epochs = [20, 100, 200]
        # Parameters 2: size of mini-batch
        batch_size = [32, 64, 128]
        # Parameters 3: number of hidden units
        dim_h = [100, 200, 300]
        # Parameters 4: learning rate
        learning_rate = [1e-4, 1e-3, 1e-2]
  
        # vectorized evaluations of 5-D fields for cross validation
        p0, p1, p2, p3, p4 = np.meshgrid(img_side_lens, n_epochs, 
                                         batch_size, dim_h, learning_rate, indexing='ij')
        
        results = {}
        for px0 in range(len(img_side_lens)):
            for px1 in range(len(n_epochs)):
                for px2 in range(len(batch_size)):
                    for px3 in range(len(dim_h)):
                        for px4 in range(len(learning_rate)):
                            cur_p0 = p0[px0, px1, px2, px3, px4]
                            cur_p1 = p1[px0, px1, px2, px3, px4]
                            cur_p2 = p2[px0, px1, px2, px3, px4]
                            cur_p3 = p3[px0, px1, px2, px3, px4]
                            cur_p4 = p4[px0, px1, px2, px3, px4]
                            
                            trainAcc, validateAcc, testAcc, W0, b0 = \
                                part8_helper(train_x_32, train_y_32, 
                                             validation_x_32, validation_y_32, 
                                             test_x_32, test_y_32, 
                                             n_epochs=int(cur_p1), batch_size=int(cur_p2),
                                             img_side_len=int(cur_p0), dim_h=int(cur_p3),
                                             learning_rate=float(cur_p4))
                                
                            results[tuple([cur_p0,cur_p1,cur_p2,cur_p3,cur_p4])] = \
                                [trainAcc, validateAcc, testAcc, W0, b0]
                            
        
        max_testAcc = 0.0
        max_avgAcc = 0.0
        for params, result in results.items():
            testAcc = result[2]
            avg_acc = np.mean([result[0][-1], result[1][-1], result[2][-1]])
            
            if testAcc > max_testAcc and avg_acc > max_avgAcc:
                max_avgAcc = avg_acc
                max_params = params
                max_trainAcc = result[0]
                max_validateAcc = result[1]
                max_testAcc = testAcc
                max_W0 = result[3]
                max_b0 = result[4]
            
    return max_params, max_trainAcc, max_validateAcc, max_testAcc, max_W0, max_b0
    

def part8(use_best_param):
    max_params, max_trainAcc, max_validateAcc, max_testAcc, max_W0, max_b0 = part8_grid_search(use_best_param)
    
    print('========== PART 8 BEST PERFORMANCE ==========')
    print('Parameters: resolution={}x{}, n_epochs={}, batch_size={}, dim_h={}, learning_rate={}'
            .format(max_params[0], max_params[0], max_params[1], 
                    max_params[2], max_params[3], max_params[4]))    
    print('Best Training accuracy: {}'.format(str(max_trainAcc[-1])))
    print('Best Validation accuracy: {}'.format(str(max_validateAcc[-1])))
    print('Best Test accuracy: {}'.format(str(max_testAcc[-1])))
    
    xAxis = range(0, max_params[1])
    plt_title = \
        'resolution={}x{}, n_epochs={}, batch_size={}, dim_h={}, learning_rate={}'\
        .format(max_params[0], max_params[0], max_params[1], 
                max_params[2], max_params[3], max_params[4])
    plt.title("\n".join(wrap(plt_title)))
    plt.plot(xAxis, max_trainAcc, label='Training')
    plt.plot(xAxis, max_validateAcc, label='Validation')
    plt.plot(xAxis, max_testAcc, label='Test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(RESULTS_FILEPATH + 'part8_learning.png')
    plt.clf()
    
    # save trained nn parameters for later use
    part8_nn = {}
    part8_nn['W0'] = max_W0
    part8_nn['b0'] = max_b0
    
    np.save("part8_nn.npy", part8_nn)
    
 
def part9_helper(train_x, train_y, act, part8_W0, part8_b0, img_side_len=32):
    dim_h = int(part8_W0.shape[0])
    act_indices = [i for i, y in enumerate(train_y) if np.argmax(y) == act]
    name = ACT[act].split()[1].lower()

    torch_x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    W0 = Variable(part8_W0.data[:], requires_grad=False).type(dtype_float)
    b0 = Variable(part8_b0.data[:].view(1,dim_h), requires_grad=False).type(dtype_float)
    activations = torch.nn.ReLU()(torch.matmul(torch_x, W0.t()) + b0.repeat(torch_x.shape[0], 1))
    
    hunit_freq = np.zeros(dim_h)
    for act_ind in act_indices:
        cur_hidden_unit = activations[act_ind,:]
        most_useful_unit_val, most_useful_unit_ind = cur_hidden_unit.max(0)
        hunit_freq[int(most_useful_unit_ind)] += 1
    top_three_units = hunit_freq.argsort()[::-1][:3]

    fig = plt.figure()
    for d in range(3):
        top_i = top_three_units[d]
        top_weight = part8_W0.data.numpy()[top_i,:].reshape((img_side_len,img_side_len,3))
        
        top_weight_r = top_weight[:,:,0]
        top_weight_g = top_weight[:,:,1]
        top_weight_b = top_weight[:,:,2]
        top_weight_sum = top_weight_r + top_weight_g + top_weight_b
        
        subplt = fig.add_subplot(1, 3, d + 1)
        plt.imshow(top_weight_sum.reshape((img_side_len,img_side_len)),
                   cmap=cm.coolwarm)
        subplt.set_title('hidden unit #{}'.format(str(top_i)), fontsize=10)
                         
    plt.savefig(RESULTS_FILEPATH + 'part9_{}.png'.format(name), bbox_inches='tight')
    plt.clf()


def part9():
    # Peri Gilpin and Bill Hader
    part9_actors = [1, 4]
    
    try:
        part8_nn = np.load('part8_nn.npy', encoding = 'latin1').item()
    except IOError:
        print('Please run part 8 first to get weights and bias') 

    # load data for resolution 32x32
    if not glob.glob('faces_all_32.mat'):
        get_dataset(32)
    data_32 = loadmat('faces_all_32.mat')
    train_x_32, train_y_32 = get_train_faces(data_32, 32)
    
    for act in part9_actors:
        part9_helper(train_x_32, train_y_32, act, part8_nn['W0'], part8_nn['b0'])
    
    
if __name__ == "__main__":
    # Use grid search
    #part8(False)
    
    # Don't use grid search, use best parameters that we already found
    #part8(True)
    
    part9()