from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import pickle
import os



def plot_data(data):

    return True

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
FEATURES = ['class', 'dwell_time', 'ave_speed', 'e2e_speed', 'obj_ave_size', 'obj_max_size', 'obj_num', 'obj_max_w_h']  
if __name__ == "__main__":

    TRAIN_DATA = './train_data'
    MODEL_FOLDER = './models'
    lines = [x.strip() for x in open(TRAIN_DATA).readlines()]
    
    X = []
    y = [] 
    for line in lines:
        data = [float(x) for x in line.split(',')][: len(FEATURES)]
        score = float(line.split(',')[-1])

        X += [data]
        y += [score]

    regr = DecisionTreeRegressor(max_depth = 5)
    refr_clf = regr.fit(X, y)
    with open(os.path.join(MODEL_FOLDER, 'regr_0.5_3.pickle'), 'wb') as fh:
        pickle.dump(refr_clf, fh)

    linear_regr = linear_model.LinearRegression()
    linear_regr.fit(X, y) 
    with open(os.path.join(MODEL_FOLDER, 'linear_regr_3.pickle'), 'wb') as fh:
        pickle.dump(linear_regr, fh)

