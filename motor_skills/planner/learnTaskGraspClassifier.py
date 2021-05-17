import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn import svm
import sklearn.metrics
from sklearn.model_selection import GridSearchCV

class TaskGraspClassifier(object):
    def __init__(self, grasp_family_generator, labels, indices, input_feature_size='full', percent_train=0.8, percent_train_set_to_use=1.0):
        super(TaskGraspClassifier, self).__init__()

        grasp_data = None
        if input_feature_size == 'full':
            grasp_family_generator.generateGraspSpace()
            grasp_data = grasp_family_generator.grasp_family_spaces[0][indices, :]
        elif input_feature_size == 'compressed':
            grasp_family_generator.generateGrabstraction(compression_alg="autoencoder", embedding_dim=3, autoencoder_file=["/home/mcorsaro/grabstraction_results/saved_models/autoencoder_door_0"])
            grasp_data = grasp_family_generator.grabstracted_inputs[0][indices, :]

        grasp_data, labels = skshuffle(grasp_data, labels)

        cutoff = int(len(grasp_data)*percent_train)

        train_cutoff = int(cutoff*percent_train_set_to_use)

        self.td, self.tl = grasp_data[:train_cutoff], labels[:train_cutoff]
        self.vd, self.vl = grasp_data[cutoff:], labels[cutoff:]

    def gridSearch(self, C_vals_SVM=[0.1, 1, 5, 25, 50, 75, 100, 150, 500, 1000], gamma_vals_SVM=[1e-2, 1e-1, 1e0, 1e1, 1e2, 50, 1e3, 150, 200, 500, 1e4, 'scale', 'auto'], poly_degree_SVM=[2, 3, 4, 5]):
        #https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
        print("Training with", self.td.shape, "\nTesting with", self.vd.shape)

        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html - auto = 1/n_features, scale = 1/(n_features * X.var())
        gamma_auto = 1./self.td.shape[1]
        # Default
        gamma_scale = 1./(self.td.shape[1]*self.td.var())
        print("Performing grid search with auto gamma", gamma_auto, "and scale gamma", gamma_scale)
        param_grid = [
            {'C': C_vals_SVM, 'kernel': ['linear']},
            {'C': C_vals_SVM, 'gamma': gamma_vals_SVM, 'kernel': ['rbf']},
            #{'C': C_vals_SVM, 'gamma': gamma_vals_SVM, 'kernel': ['poly'], 'degree': poly_degree_SVM},
        ]
        scores = ['precision', 'recall']
        for score in scores:
            clf = GridSearchCV(svm.SVC(), param_grid, scoring='%s_macro' % score, n_jobs=-1)
            clf.fit(self.td, self.tl)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            pred_vl = clf.predict(self.vd)
            print(sklearn.metrics.classification_report(self.vl, pred_vl))
            print()

    def trainSVM(self):
        print("Training with", self.td.shape, "\nTesting with", self.vd.shape)
        clf = svm.SVC()
        clf.fit(self.td, tl)

        pred_vl = clf.predict(self.vd)
        print("Test accuracy", sklearn.metrics.accuracy_score(self.vl, pred_vl))
