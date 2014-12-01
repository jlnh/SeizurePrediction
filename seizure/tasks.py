from collections import namedtuple
import os.path
import numpy as np
import pylab as pl
import scipy.io
import common.time as time
from sklearn import cross_validation, preprocessing
from sklearn.metrics import roc_curve, auc
from scipy.signal import resample
from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression as IR
from copy import deepcopy
from matplotlib.backends.backend_pdf import PdfPages

TaskCore = namedtuple('TaskCore', ['cached_data_loader', 'data_dir', 'target', 'pipeline', 'classifier_name',
                                   'classifier', 'normalize', 'gen_preictal', 'cv_ratio', 'plot2file'])


class Task(object):
    """
    A Task computes some work and outputs a dictionary which will be cached on disk.
    If the work has been computed before and is present in the cache, the data will
    simply be loaded from disk and will not be pre-computed.
    """
    def __init__(self, task_core):
        self.task_core = task_core

    def filename(self):
        raise NotImplementedError("Implement this")

    def run(self):
        return self.task_core.cached_data_loader.load(self.filename(), self.load_data)


class LoadpreictalDataTask(Task):
    """
    Load the preictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y, 'latencies': latencies}
    """
    def filename(self):
        return 'data_preictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, 'preictal', self.task_core.pipeline,
                           self.task_core.gen_preictal)


class LoadInterictalDataTask(Task):
    """
    Load the interictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """
    def filename(self):
        return 'data_interictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, 'interictal', self.task_core.pipeline)


class LoadTestDataTask(Task):
    """
    Load the test mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X}
    """
    def filename(self):
        return 'data_test_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, 'test', self.task_core.pipeline)


class TrainingDataTask(Task):
    """
    Creating a training set and cross-validation set from the transformed preictal and interictal data.
    """
    def filename(self):
        return None  # not cached, should be fast enough to not need caching

    def load_data(self):
        preictal_data = LoadpreictalDataTask(self.task_core).run()
        interictal_data = LoadInterictalDataTask(self.task_core).run()
        return prepare_training_data(preictal_data, interictal_data, self.task_core.cv_ratio)


class CrossValidationScoreTask(Task):
    """
    Run a classifier over a training set, and give a cross-validation score.
    """
    def filename(self):
        return 'score_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        classifier_data = train_classifier(self.task_core.classifier, data, normalize=self.task_core.normalize)
        del classifier_data['classifier'] # save disk space
        return classifier_data


class TrainClassifierTask(Task):
    """
    Run a classifier over the complete data set (training data + cross-validation data combined)
    and save the trained models.
    """
    def filename(self):
        return 'classifier_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        return train_classifier(self.task_core.classifier, self.task_core.plot2file, data, use_all_data=True, normalize=self.task_core.normalize)


class MakePredictionsTask(Task):
    """
    Make predictions on the test data.
    """
    def filename(self):
        return 'predictions_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        y_classes = data.y_classes
        del data

        classifier_data = TrainClassifierTask(self.task_core).run()
        test_data = LoadTestDataTask(self.task_core).run()
        X_test = flatten(test_data.X)

        return make_predictions(self.task_core.target, self.task_core.plot2file, X_test, y_classes, classifier_data)

class GetCrossSubjectDataTask(Task):
    """
    assemble all the data cross the subject
    and save the trained models.
    """
    def filename(self):
        return 'assemble_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        def concat(a, b):
            return np.concatenate((a, b), axis=0)
        
        preictal_data = LoadpreictalDataTask(self.task_core).run()
        interictal_data = LoadInterictalDataTask(self.task_core).run()
        test_data = LoadTestDataTask(self.task_core).run()
        preictal_X, preictal_y = flatten(preictal_data.X), preictal_data.y
        interictal_X, interictal_y = flatten(interictal_data.X), interictal_data.y
        X_train = concat(preictal_X, interictal_X)
        y_train = concat(preictal_y, interictal_y)
        X_test = flatten(test_data.X)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test
        }

class TrainCrossSubjectClassifierTask(Task):
    """
    Run a classifier over the complete data set (training data + cross-validation data combined)
    and save the trained models.
    """
    def filename(self):
        return 'classifier_cross_%s_%s_%s' % (self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        return train_classifier(self.task_core.classifier, data, use_all_data=True, normalize=self.task_core.normalize)


# a list of pairs indicating the slices of the data containing full seizures
# e.g. [(0, 5), (6, 10)] indicates two ranges of seizures
def seizure_ranges_for_latencies(latencies):
    indices = np.where(latencies == 0)[0]

    ranges = []
    for i in range(1, len(indices)):
        ranges.append((indices[i-1], indices[i]))
    ranges.append((indices[-1], len(latencies)))

    return ranges


#generator to iterate over competition mat data
def load_mat_data(data_dir, target, component):
    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        if i < 10:
            nstr = '000%d' %i
        elif i < 100:    
            nstr = '00%d' %i
        elif i < 1000:
            nstr = '0%d' %i
        else:
            nstr = '%d' %i
            
        filename = '%s/%s_%s_segment_%s.mat' % (dir, target, component, nstr)
        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)
            yield(data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True


# process all of one type of the competition mat data
# data_type is one of ('preictal', 'interictal', 'test')
def parse_input_data(data_dir, target, data_type, pipeline, subjectID=0, gen_preictal=False):
    preictal = data_type == 'preictal'
    interictal = data_type == 'interictal'
    targetFrequency = 100   #re-sample to target frequency
    sampleSizeinSecond = 600
    totalSample = 12

    mat_data = load_mat_data(data_dir, target, data_type)

    # for each data point in preictal, interictal and test,
    # generate (X, <y>) per channel
    def process_raw_data(mat_data):
        start = time.get_seconds()
        print 'Loading data',
        #print mat_data
        X = []
        y = []
        previous_transformed_data = []  #used in two window model
        previous_sequence = 0
        for segment in mat_data:
            for skey in segment.keys():
                if "_segment_" in skey.lower():
                    mykey = skey
                    

            if preictal:
                preictual_sequence = segment[mykey][0][0][4][0][0]
                y_value = preictual_sequence    #temporarily set to sequence number 
                if preictual_sequence != previous_sequence+1:
                    previous_transformed_data = []  #if data is not in sequence
                previous_sequence = preictual_sequence  
            elif interictal:
                y_value = 0
                previous_transformed_data = []  #interictal data is not in sequence between files
            else:
                previous_transformed_data = []  #test data is not in sequence between files
                
            
            data = segment[mykey][0][0][0]
            sampleFrequency = segment[mykey][0][0][2][0][0]
            axis = data.ndim - 1
            if sampleFrequency > targetFrequency:   #resample to target frequency
                data = resample(data, targetFrequency*sampleSizeinSecond, axis=axis)

            '''DataSampleSize: split the 10 minutes data into several clips: 
            For one second data clip, patient1 and patient2 were finished in 3 hours. Dog1 clashed after 7+ hours for out of memory
            try ten second data clip
            '''
            DataSampleSize = data.shape[1]/(totalSample *1.0)  #try to split data into equal size
            splitIdx = np.arange(DataSampleSize, data.shape[1], DataSampleSize)
            splitIdx = np.int32(np.ceil(splitIdx))
            splitData = np.hsplit(data,splitIdx)
#             for i  in range(totalSample):
#                 s = splitData[i]
#                 s2 = splitData[i+totalSample]
                
            for s in splitData:
                if s.size > 0:    #is not empty
#                     s = 1.0 * s     #convert int to float
#                     s_scale = preprocessing.scale(s, axis=0, with_std = True)
#                     transformed_data = pipeline.apply([subjectID, s])
                    transformed_data = pipeline.apply(s)
#                     previous_transformed_data.append(transformed_data)
#                         transformed_data2 = pipeline.apply([subjectID, s1])
#                     if len(previous_transformed_data) > totalSample/2:
#                         combined_transformed_data = np.concatenate((transformed_data, previous_transformed_data.pop(0)), axis=transformed_data.ndim-1)
#                         X.append(combined_transformed_data)
                    X.append(transformed_data)
                    if preictal or interictal:
                        y.append(y_value)
                                

        print '(%ds)' % (time.get_seconds() - start)

        X = np.array(X)
        if preictal or interictal:
            y = np.array(y)
            print 'X', X.shape, 'y', y.shape
            return X, y
        else:
            print 'X', X.shape
            return X

    data = process_raw_data(mat_data)

    if len(data) == 2:
        X, y = data
        return {
            'X': X,
            'y': y
        }
    else:
        X = data
        return {
            'X': X
        }


# flatten data down to 2 dimensions for putting through a classifier
def flatten(data):
    if data.ndim > 2:
        return data.reshape((data.shape[0], np.product(data.shape[1:])))
    else:
        return data


# split up preictal and interictal data into training set and cross-validation set
def prepare_training_data(preictal_data, interictal_data, cv_ratio):
    print 'Preparing training data ...',
    preictal_X, preictal_y = flatten(preictal_data.X), preictal_data.y
    interictal_X, interictal_y = flatten(interictal_data.X), interictal_data.y

    # split up data into training set and cross-validation set for both seizure and early sets
    preictal_X_train, preictal_y_train, preictal_X_cv, preictal_y_cv = split_train_preictal(preictal_X, preictal_y, cv_ratio)
    interictal_X_train, interictal_y_train, interictal_X_cv, interictal_y_cv = split_train_random(interictal_X, interictal_y, cv_ratio)

    def concat(a, b):
        return np.concatenate((a, b), axis=0)

    X_train = concat(preictal_X_train, interictal_X_train)
    y_train = concat(preictal_y_train, interictal_y_train)
    X_cv = concat(preictal_X_cv, interictal_X_cv)
    y_cv = concat(preictal_y_cv, interictal_y_cv)

    y_classes = np.unique(concat(y_train, y_cv))

    start = time.get_seconds()
    elapsedSecs = time.get_seconds() - start
    print "%ds" % int(elapsedSecs)

    print 'X_train:', np.shape(X_train)
    print 'y_train:', np.shape(y_train)
    print 'X_cv:', np.shape(X_cv)
    print 'y_cv:', np.shape(y_cv)
    print 'y_classes:', y_classes

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_cv': X_cv,
        'y_cv': y_cv,
        'y_classes': y_classes
    }


# split interictal segments at random for training and cross-validation
def split_train_random(X, y, cv_ratio):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=cv_ratio, random_state=0)
    return X_train, y_train, X_cv, y_cv

def split_train_preictal(X, y, cv_ratio):
    X_train = []
    X_cv = []
    y_train = []
    y_cv = []
    zxy = zip(X,y)
    y = 1
    for i in range(1,7):
#     for i in range(1,2):
        X1 = []
        y1 = []
        for r in zxy:
            X2, y2 = r
            if y2 <= i and y2 > i-1:
                X1.append(X2)
#                 y1.append(y2)
                y1.append(y)    #set back to two level classification
                
        X1_train, X1_cv, y1_train, y1_cv = cross_validation.train_test_split(X1, y1, test_size=cv_ratio, random_state=0)
        X_train.append(X1_train)
        y_train.append(y1_train)
        X_cv.append(X1_cv)
        y_cv.append(y1_cv)
    
    X_train = np.concatenate(X_train)    
    y_train = np.concatenate(y_train)    
    X_cv = np.concatenate(X_cv)    
    y_cv = np.concatenate(y_cv)    
    return X_train, y_train, X_cv, y_cv
    
def train(classifier, X_train, y_train, X_cv, y_cv, y_classes):
    print "Training ..."

    print 'Dim', 'X', np.shape(X_train), 'y', np.shape(y_train), 'X_cv', np.shape(X_cv), 'y_cv', np.shape(y_cv)
    start = time.get_seconds()
    classifier.fit(X_train, y_train)
    print "Scoring..."
    score = score_classifier_auc(classifier, X_cv, y_cv, y_classes)

    elapsedSecs = time.get_seconds() - start
    print "t=%ds score=%f" % (int(elapsedSecs), score)
    return score


# train classifier for predictions
def train_all_data(classifier, plot2file, X_train, y_train, X_cv, y_cv):
    print "Training ..."
    X = np.concatenate((X_train, X_cv), axis=0)
    y = np.concatenate((y_train, y_cv), axis=0)
    print 'Dim', np.shape(X), np.shape(y)
    start = time.get_seconds()
    classifier_cv = deepcopy(classifier)
    classifier.fit(X, y)
    classifier_cv.fit(X_train, y_train)
    score_classifier_auc(classifier_cv, plot2file, X_cv, y_cv, y_cv)
    y_estimate = classifier_cv.predict_proba(X_cv)
    elapsedSecs = time.get_seconds() - start
    print "t=%ds" % int(elapsedSecs)
    return y_estimate


# sub mean divide by standard deviation
def normalize_data(X_train, X_cv):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)

    return X_train, X_cv

# depending on input train either for predictions or for cross-validation
def train_classifier(classifier, plot2file, data, use_all_data=False, normalize=False):
    X_train = data.X_train
    y_train = data.y_train
    X_cv = data.X_cv
    y_cv = data.y_cv

    if normalize:
        X_train, X_cv = normalize_data(X_train, X_cv)

    if not use_all_data:
        score = train(classifier, X_train, y_train, X_cv, y_cv, data.y_classes)
        return {
            'classifier': classifier,
            'score': score,
        }
    else:
        y_estimate = train_all_data(classifier, plot2file, X_train, y_train, X_cv, y_cv)
        calibrate_matrix = train_calibrator(y_cv, y_estimate, plot2file) 
        lr = LR()      
        lr.fit(y_estimate, y_cv) 
        return {
            'classifier': classifier,
            'calibrate_matrix': calibrate_matrix,
            'LR' : lr
        }

def train_calibrator(y, y_estimate, plot2file):
    print "Training calibrator..."
    start = time.get_seconds()
    preictal_predictions = []
    p_y_cv = [0.0 if x == 0.0 else 1.0 for x in y]
    for i in range(len(y_estimate)):
        p = y_estimate[i]
        preictal = translate_prediction(p)
        preictal_predictions.append(preictal)

    fpr, tpr, thresholds = roc_curve(p_y_cv, preictal_predictions)
    p_roc_auc = auc(fpr, tpr)
    
    y_av = np.average(p_y_cv)
    y_std = np.std(p_y_cv)
    ye_av = np.average(preictal_predictions)
    ye_std = np.std(preictal_predictions)
    
    pl.clf()
    pl.hist(preictal_predictions, bins=50)
    pl.xlabel('preictal estimate')
    pl.ylabel('counts')
    pl.title('CV histogram (mean_cv= %0.3f, mean_es=%0.3f, std_es=%0.3f)' %(y_av, ye_av, ye_std))
#     pl.show()
    plot2file.savefig()
    calibrate_matrix = np.array([ye_av, ye_std])
    
    elapsedSecs = time.get_seconds() - start
    print "t=%ds score=%f" % (int(elapsedSecs), p_roc_auc)
    return calibrate_matrix
    
# convert the output of classifier predictions into (Seizure, Early) pair
def translate_prediction(prediction):
    if prediction.shape[0] == 7:
        interictal, p1, p2, p3, p4, p5, p6 = prediction
        preictal = p1 + p2 + p3 + p4 + p5 + p6

        return preictal
    elif prediction.shape[0] == 2:
        interictal, p1 = prediction
        preictal = p1 
        return preictal
    elif prediction.shape[0] == 1:
        return prediction[0]
    else:
        raise NotImplementedError()


# use the classifier and make predictions on the test data
def make_predictions(target, plot2file, X_test, y_classes, classifier_data):
    print classifier_data
    classifier = classifier_data.classifier
    lr = classifier_data.LR
    predictions_proba = classifier.predict_proba(X_test)
#     predictions_calibrated = lr.predict_proba(predictions_proba)
    predictions_calibrated = predictions_proba
    data = calibrate_prediction(plot2file, predictions_calibrated, classifier_data.calibrate_matrix)
    predictions_calibrated = data['preictal_calibrated']
    is_aggressive = data['is_aggressive']

    lines = []
    totalSample = 12
    for i in range(len(predictions_calibrated)/totalSample):
        j = i+1
        if j < 10:
            nstr = '000%d' %j
        elif j < 100:    
            nstr = '00%d' %j
        elif j < 1000:
            nstr = '0%d' %j
        else:
            nstr = '%d' %j
        
        preictal_segments = []
        for k in range(totalSample):
            p = predictions_calibrated[i*totalSample+k]
            preictal = translate_prediction(p)
            preictal_segments.append(preictal)            
        
        preictalOverAllSample = get_combine_prediction(preictal_segments, is_aggressive)    
        lines.append('%s_test_segment_%s.mat,%.15f' % (target, nstr, preictalOverAllSample))

    return {
        'data': '\n'.join(lines)
    }


def get_combine_prediction(preictal_segments, is_aggressive):
    from scipy.stats.mstats import *
    #average method: arithmetic, geometry and harmonic
    
    interictal_amean = 1.0 - np.mean(preictal_segments)
    interictal = 1.0 - np.array(preictal_segments)
    interictal_gmean = gmean(interictal)
    interictal_hmean = hmean(interictal)
    interictal_agmean = 0.5 * (interictal_amean + interictal_gmean)
    interictal_hgmean = 0.5 * (interictal_hmean + interictal_gmean)
    
#     combine_prediction = 1.0 - interictal_hmean
    if is_aggressive:
        return 1.0 - interictal_hmean
    else:
        return 1.0 - interictal_amean


# the scoring mechanism used by the competition leaderboard
def score_classifier_auc(classifier, plot2file, X_cv, y_cv, y_classes):
    predictions = classifier.predict_proba(X_cv)
    preictal_predictions = []
    p_y_cv = [0.0 if x == 0.0 else 1.0 for x in y_cv]

    for i in range(len(predictions)):
        p = predictions[i]
        preictal = translate_prediction(p)
        preictal_predictions.append(preictal)

    fpr, tpr, thresholds = roc_curve(p_y_cv, preictal_predictions)
    p_roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    pl.clf()
    pl.subplot(211)
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % p_roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    
    pl.subplot(212)
    pl.plot(thresholds, tpr, label='tpr')
    pl.plot(thresholds, fpr, label='fpr')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('thresholds')
    pl.ylabel('True/false Positive Rate')
#     pl.title('thresholds vs. True/false Positive Rate')
    pl.legend(loc="upper right")
    plot2file.savefig()
    
#     pl.show()
    

    return p_roc_auc

def calibrate_prediction(plot2file, predictions, calibrate_matrix):
    
    cmean = calibrate_matrix[0]
    cstd = calibrate_matrix[1]
    preictal_predictions = []
    for i in range(len(predictions)):
        p = predictions[i]
        preictal = translate_prediction(p)
        preictal_predictions.append(preictal)

    ye_av = np.average(preictal_predictions)
    ye_std = np.std(preictal_predictions)
    
    pl.clf()
    pl.hist(preictal_predictions, bins=50)
    pl.xlabel('preictal estimate')
    pl.ylabel('counts')
    pl.title('Test data set histogram ( mean_es=%0.3f, std_es=%0.3f)' %(ye_av, ye_std))
    plot2file.savefig()
#     pl.show()
    
    target = cmean
    for i in range(1):
        if i==0:
            pp = np.percentile(preictal_predictions, (1.0-target)*100.0)
            tobecalibrate = preictal_predictions
        else:
            pp = np.percentile(preictal_predictions, target*100.0)
            
        preictal_calibrated = []
        upper_limit = max(tobecalibrate)
        lower_limit = min(tobecalibrate)
        ratio1 = target/pp
        ratio2 = (upper_limit-target)/(upper_limit-pp)
        for p in tobecalibrate:
            if p <= pp:
                pc = ratio1 * (p - lower_limit)
            else:
                pc = target + ratio2 * (p-pp)
               
            preictal_calibrated.append(pc)
            
        tobecalibrate = preictal_calibrated
        
    preictal_calibrated = np.reshape(preictal_calibrated, (len(preictal_calibrated),1))
    yc_av = np.average(preictal_calibrated)
    yc_std = np.std(preictal_calibrated)
    
    pl.clf()
    pl.hist(preictal_calibrated, bins=50)
    pl.xlabel('preictal calibrated')
    pl.ylabel('counts')
    pl.title('histogram of preictal calibrated ( mean_es=%0.3f, std_es=%0.3f)' %(yc_av, yc_std))
    
    plot2file.savefig()
    if ye_av > 0.4:
        is_aggressive = False
    else:
        is_aggressive = True
        
    return {
        'preictal_calibrated': preictal_calibrated,
        'is_aggressive': is_aggressive
    }


