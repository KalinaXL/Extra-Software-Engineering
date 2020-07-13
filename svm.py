import os
import pickle
class SVM:
    model_path = os.path.join('model', 'clf_svm.pickle')
    clf = None
    @staticmethod
    def load_model(force_load = False):
        if SVM.clf is None or force_load:
            SVM.clf = pickle.loads(open(SVM.model_path, 'rb').read())
    @staticmethod
    def predict(input):
        if len(input.shape) == 1:
            input = [input]
        return SVM.clf.predict(input)[0]
    
        
