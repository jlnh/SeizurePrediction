import json
import os.path
import numpy as np
from common import time
from common.data import CachedDataLoader, makedirs
from common.pipeline import Pipeline
from seizure.transforms import FFT, Slice, Magnitude, Log10, FFTWithTimeFreqCorrelation, MFCC, Resample, Stats, \
    DaubWaveletStats, TimeCorrelation, FreqCorrelation, TimeFreqCorrelation
from seizure.tasks import TaskCore, CrossValidationScoreTask, MakePredictionsTask, TrainClassifierTask, \
    GetCrossSubjectDataTask, translate_prediction, normalize_data
from seizure.scores import get_score_summary, print_results

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib.backends.backend_pdf import PdfPages

def run_seizure_detection(build_target):
    """
    The main entry point for running seizure-detection cross-validation and predictions.
    Directories from settings file are configured, classifiers are chosen, pipelines are
    chosen, and the chosen build_target ('cv', 'predict', 'train_model') is run across
    all combinations of (targets, pipelines, classifiers)
    """

    with open('SETTINGS.json') as f:
        settings = json.load(f)

    data_dir = str(settings['competition-data-dir'])
    cache_dir = str(settings['data-cache-dir'])
    submission_dir = str(settings['submission-dir'])
    figure_dir = str(settings['figure-dir'])

    makedirs(submission_dir)

    cached_data_loader = CachedDataLoader(cache_dir)

    ts = time.get_millis()
    
    targets = [
        'Dog_1',
        'Dog_2',
        'Dog_3',
        'Dog_4',
        'Dog_5',
        'Patient_1',
        'Patient_2',
    ]
    pipelines = [
        # NOTE: you can enable multiple pipelines to run them all and compare results
        Pipeline(gen_preictal=True,  pipeline=[FFTWithTimeFreqCorrelation(50, 2500, 400, 18, 'usf')]), # winning submission
    ]
    classifiers = [
        # NOTE: you can enable multiple classifiers to run them all and compare results
#         (RandomForestClassifier(n_estimators=300, min_samples_split=1, max_features=0.5, bootstrap=False, n_jobs=-1, random_state=0), 'rf300mss1mf05Bfrs0'),

#         (ExtraTreesClassifier(n_estimators=3000, min_samples_split=1, max_features=0.15, bootstrap=False, n_jobs=-1, random_state=0), 'ET3000mss1mf015Bfrs0'),
#         
#         (GradientBoostingClassifier(n_estimators=3000, min_samples_split=1, max_features=0.15, learning_rate=0.02, subsample = 0.5, random_state=0), 'GBRT3000mms1mf015Lr002Ss05rs0'),

        (SVC(C=1e6, kernel='rbf', gamma=0.01, coef0=0.0, shrinking=True, probability=True, tol=1e-5, cache_size=2000, class_weight='auto', max_iter=-1, random_state=0), 'svcce6rbfg001co0stte-5cwautors0'),
    ]
    cv_ratio = 0.5

    def should_normalize(classifier):
        clazzes = [LogisticRegression]
        return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)

    def train_full_model(make_predictions):
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                guesses = ['clip,preictal']
                classifier_filenames = []
                plot2file = PdfPages(os.path.join(figure_dir, ('figure%d-_%s_%s_.pdf' % (ts, classifier_name, pipeline.get_name()))))
                for target in targets:
                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_preictal=pipeline.gen_preictal,
                                         cv_ratio=cv_ratio, plot2file = plot2file)

                    if make_predictions:
                        predictions = MakePredictionsTask(task_core).run()
                        guesses.append(predictions.data)
                    else:
                        task = TrainClassifierTask(task_core)
                        task.run()
                        classifier_filenames.append(task.filename())

                if make_predictions:
                    filename = 'submission%d-%s_%s.csv' % (ts, classifier_name, pipeline.get_name())
                    filename = os.path.join(submission_dir, filename)
                    with open(filename, 'w') as f:
                        print >> f, '\n'.join(guesses)
                    print 'wrote', filename
                else:
                    print 'Trained classifiers ready in %s' % cache_dir
                    for filename in classifier_filenames:
                        print os.path.join(cache_dir, filename + '.pickle')
                        
                plot2file.close()

    def predict_all(make_predictions):
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                lines = ['clip,preictal']
                subjectID = 0
                X_train = y_train = X_test = test_size = []
                for target in targets:
                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_preictal=pipeline.gen_preictal,
                                         cv_ratio=cv_ratio)
                    
                    data = GetCrossSubjectDataTask(task_core).run()
#                     a = np.shape(data.X_test)[0]
                    test_size.append(np.shape(data.X_test)[0])
                    if subjectID > 0:
                        X_train = np.concatenate((X_train, data.X_train), axis=0)
                        y_train = np.concatenate((y_train, data.y_train), axis=0)
                        X_test = np.concatenate((X_test, data.X_test), axis=0)
                    else:
                        X_train = data.X_train
                        y_train = data.y_train
                        X_test = data.X_test
                    subjectID += 1
                    
                #Training
                task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                     target=[], pipeline=pipeline,
                                     classifier_name=classifier_name, classifier=classifier,
                                     normalize=should_normalize(classifier), gen_preictal=pipeline.gen_preictal,
                                     cv_ratio=cv_ratio)
                y_train = np.ceil(0.1*y_train)
                y_train.astype('int_')
                if should_normalize(classifier):
                    X_train, temp = normalize_data(X_train, X_train)
                    
                print "Training ..."
                print 'Dim', np.shape(X_train), np.shape(y_train)
                start = time.get_seconds()
                classifier.fit(X_train, y_train)
                elapsedSecs = time.get_seconds() - start
                print "t=%ds" % int(elapsedSecs)
                
                y_estimate = classifier.predict_proba(X_train)
                lr = LogisticRegression(random_state = 0)      
                lr.fit(y_estimate, y_train)
                predictions_proba = classifier.predict_proba(X_test)
                predictions_calibrated = lr.predict_proba(predictions_proba)
                
                #output
                m = 0
                totalSample = 12
                startIdx = 0
                for target in targets:
                    for i in range(test_size[m]/totalSample):
                        j = i+1
                        if j < 10:
                            nstr = '000%d' %j
                        elif j < 100:    
                            nstr = '00%d' %j
                        elif j < 1000:
                            nstr = '0%d' %j
                        else:
                            nstr = '%d' %j
                        
                        preictalOverAllSample = 0
                        for k in range(totalSample):
                            p = predictions_calibrated[i*totalSample+k+startIdx]
                            preictal = translate_prediction(p)
                            preictalOverAllSample += preictal/totalSample
                         
                        newline =  '%s_test_segment_%s.mat,%.15f' % (target, nstr, preictalOverAllSample)   
                        lines.append(newline)
                        
                    print newline
                    startIdx = startIdx + test_size[m]
                    m += 1
                
                filename = 'submission%d-%s_%s.csv' % (ts, classifier_name, pipeline.get_name())
                filename = os.path.join(submission_dir, filename)
                with open(filename, 'w') as f:
                    print >> f, '\n'.join(lines)
                print 'wrote', filename

    def do_cross_validation():
        summaries = []
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                scores = []
                for target in targets:
                    print 'Processing %s (classifier %s)' % (target, classifier_name)

                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
#                                          target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_preictal=pipeline.gen_preictal,
                                         cv_ratio=cv_ratio)

                    data = CrossValidationScoreTask(task_core).run()
                    score = data.score

                    scores.append(score)

                    print '%.3f' % score

                if len(scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    summary = get_score_summary(name, scores)
                    summaries.append((summary, np.mean(scores)))
                    print summary

            print_results(summaries)

    if build_target == 'cv':
        do_cross_validation()
    elif build_target == 'train_model':
        train_full_model(make_predictions=False)
    elif build_target == 'make_predictions':
        train_full_model(make_predictions=True)
    elif build_target == 'predict_all':
        predict_all(make_predictions=True)
    else:
        raise Exception("unknown build target %s" % build_target)
