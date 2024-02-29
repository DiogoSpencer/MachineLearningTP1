import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KernelDensity
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

errors = {"training":{"SVM":[],"NB":[]},
        "validation":{"SVM":[],"NB":[]}}

NUM_OF_FOLDS = 5

def loadData(FileTrain, FileTest):
    # CARREGAMENTO DA TRAINING DATA
    matTrain = np.loadtxt(FileTrain, delimiter = '\t')
    dataTrain = shuffle(matTrain)
    Xtrain = dataTrain[:,:-1]
    Ytrain = dataTrain[:,-1]
    means = np.mean(Xtrain,axis = 0)
    stdevs = np.std(Xtrain,axis=0)
    Xtrain = (Xtrain-means)/stdevs
    
    # CARREGAMENTO DA TEST DATA
    matTest = np.loadtxt(FileTest, delimiter='\t')
    dataTest = shuffle(matTest)
    Xtest = dataTest[:,:-1]
    Ytest = dataTest[:,-1]
    means = np.mean(Xtest,axis = 0)
    stdevs = np.std(Xtest,axis=0)
    Xtest = (Xtest-means)/stdevs
    
    return Xtrain, Ytrain, Xtest, Ytest


# CROSS VALIDATION
# PRIMEIRO CLASSIFIER
def KDEfit(Xs, Ys, bw):
    classifiers = []
    #DIVIDIR O TRAINING SET EM CLASSE 0 E CLASSE 1
    training = [Xs[Ys == 0],
                Xs[Ys == 1]]

    #PROBABILIDADE APRIO DE PERTENCER A CADA CLASSE
    logAPrioProb = [np.log(training[0].shape[0] / Ys.shape[0]),
                    np.log(training[1].shape[0] / Ys.shape[0])]
    
    for xs in training:
        for i in range(Xs.shape[1]):
            classifiers.append(KernelDensity(bandwidth=bw,kernel='gaussian').fit(xs[:,i].reshape(-1,1)))
    return classifiers, logAPrioProb

def NBwithKDE_score(classifiers, logAPrioProb, Xs, Ys,):
    number_of_features = Xs.shape[1]
    number_of_points = Xs.shape[0]
    resC0 = np.ones((number_of_points, number_of_features))
    resC1 = np.ones((number_of_points, number_of_features))
    feature_idx = 0
    for clf in classifiers:
        if(feature_idx < number_of_features):
            resC0[:,feature_idx] = clf.score_samples(Xs[:,feature_idx].reshape(-1,1)) 
        else:
            resC1[:,feature_idx-number_of_features] = clf.score_samples(Xs[:,feature_idx-number_of_features].reshape(-1,1)) 
        feature_idx = feature_idx + 1
    # SOMAR A PROBABILIDADE DE CADA FEATURE PARA CADA CLASSE COM A PROBABILIDADE APRIO E ESCOLHER O VALOR MAXIMO
    resultC0 = np.sum(resC0, axis=1) + logAPrioProb[0]
    resultC1 = np.sum(resC1, axis=1) + logAPrioProb[1]
    result = np.maximum(resultC0, resultC1)
    result = result - resultC0
    result[result != 0] = 1
    return accuracy_score(Ys, result), result

def NBwithKDE(Xs_tr, Ys_tr):
    optimal_parameters={"bandwidth":0,"accuracy":0,"classifiers":[]}
    # PROCURAR A MELHOR BANDWIDTH
    for b in np.arange(0.02, 0.62, 0.02):
        curr_accuracy = 0
        kf = StratifiedKFold(n_splits = NUM_OF_FOLDS)
        for train_ix,valid_ix in kf.split(Ys_tr, Ys_tr):
            classifiers, logAPrioProb = KDEfit(Xs_tr[train_ix], Ys_tr[train_ix], b)
            curr_accuracy += NBwithKDE_score(classifiers, logAPrioProb, Xs_tr[valid_ix], Ys_tr[valid_ix])[0]
        curr_accuracy = curr_accuracy / NUM_OF_FOLDS
        errors["training"]["NB"].append(1 - NBwithKDE_score(classifiers, logAPrioProb, Xs_tr, Ys_tr)[0])
        errors["validation"]["NB"].append(1-curr_accuracy)
        # ARRANAJR O MELHOR VALOR DA BANDWITH 
        if(optimal_parameters["accuracy"] < curr_accuracy):
           optimal_parameters["accuracy"] = curr_accuracy
           optimal_parameters["bandwidth"] = b
    print("MY NB WITH KDE: BEST ERROR = ", 1 - optimal_parameters["accuracy"], "BEST BANDWIDTH = ", optimal_parameters["bandwidth"])
    # RETORNAR UM NOVO CLASSIFIER TREINADO EM TODOS OS TRAINING SET
    return KDEfit(Xs_tr, Ys_tr, optimal_parameters["bandwidth"])  
    
## SEGUNDO CLASSIFICADOR - GAUSSIAN NB 
def GaussianNB_SKL(Xs, Ys):
    GNBClf = GaussianNB()
    return GNBClf.fit(Xs, Ys)
    
## TERCEIRO CLASSIFICADOR - SVM COM KERNEL RBF 
def SVMwith_radialBF(Xs,Ys):
    optimal_parameters = {"gamma":0,"c":1,"score":0}
    kf = StratifiedKFold(n_splits = NUM_OF_FOLDS)
    # PARAMETRO GAMMA
    for g in np.arange(0.2, 6.2, 0.2):
        curr_score = 0
        for train_ix,valid_ix in kf.split(Ys, Ys):
            sv =  SVC(C=1, kernel= 'rbf', gamma= g)
            sv.fit(Xs[train_ix], Ys[train_ix])
            curr_score += sv.score(Xs[valid_ix], Ys[valid_ix])
        curr_score = curr_score / NUM_OF_FOLDS
        errors["training"]["SVM"].append(1 - sv.score(Xs,Ys))
        errors["validation"]["SVM"].append(1 - curr_score)
        if(optimal_parameters["score"] < curr_score):
            optimal_parameters["gamma"] = g
            optimal_parameters["score"] = curr_score
    print("SVM COM RBF KERNEL : MELHOR ERRO = ", 1 - optimal_parameters["score"], "MELHOR GAMMA = ", optimal_parameters["gamma"], "MELHOR C = ", optimal_parameters["c"])
    sv = SVC(C=optimal_parameters["c"], kernel= 'rbf', gamma= optimal_parameters["gamma"])
    return sv.fit(Xs,Ys)

## PLOT DAS IMAGENS DO NB E SVM
def plots():
    plt.title("Training VS Validation error for NB")
    plt.plot(np.arange(0.02, 0.62, 0.02),errors["training"]["NB"], label="NB Training Error")
    plt.plot(np.arange(0.02, 0.62, 0.02),errors["validation"]["NB"], label="NB Validation Error")
    plt.ylabel("Error")
    plt.xlabel("Bandwidth")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("NB.png")
    plt.show()
    plt.title("Training VS Validation error for SVM")
    plt.plot(np.arange(0.2, 6.2, 0.2),errors["training"]["SVM"], label="SVM Training Error")
    plt.plot(np.arange(0.2, 6.2, 0.2),errors["validation"]["SVM"], label="SVM Validation Error")
    plt.ylabel("Error")
    plt.xlabel("Gamma")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("SVM.png")
    plt.show()
    
## CALCULO DO TEST ERROR DE TODOS OS CLASSIFICADORES
def errorTest():
    ownNB_error, NB_res = NBwithKDE_score(NBWKDE, logAPrioProb, Xs_t, Ys_t)
    ownNB_error = 1 - ownNB_error
    NBSKL_error = 1 - GNBSKL.score(Xs_t, Ys_t)
    svm_error = 1 - SVM.score(Xs_t, Ys_t)
    print("ERRO NO TEST SET GNB SK LEARN = ", NBSKL_error, "(1 - score)","\n")
    print("ERRO NO TEST SET OWN NBWKDE = ", ownNB_error, "(1 - accuracy_score)","\n")
    print("ERRO NO TEST SET SVM = ", svm_error, "(1 - score)","\n")
    return svm_error, NB_res, ownNB_error, NBSKL_error

##  APROXIMACAO NORMAL
def apprNTest():
    # N - TAMANHO DO SET
    # X - NUMERO DE EXEMPLOS MAL CLASSIFICADOS
    # P0 = X/N == ERRO
    N = Xs_t.shape[0]
    badly_Nclassified_SVM = svm_error * N
    P0_SVM = badly_Nclassified_SVM / N
    sig_SVM = np.sqrt(N * P0_SVM * (1-P0_SVM))
    print("Svm teste de aproximacao normal")
    print("X +- 1,96sig = ", badly_Nclassified_SVM , "+-", 1.96 * sig_SVM, "\n")
    X_NB = ownNB_error * N
    badly_Nclassified_NB = ownNB_error
    sig_NB = np.sqrt(N * badly_Nclassified_NB * (1 - badly_Nclassified_NB))
    print("Own Naive Bayes teste de aproximacao normal")
    print("X +- 1,96sig = ", X_NB , "+-", 1.96 * sig_NB, "\n")
    X_sciNB = NBSKL_error * N
    badly_Nclassified_sciNB = NBSKL_error
    sig_sciNB = np.sqrt(N * badly_Nclassified_sciNB * (1 - badly_Nclassified_sciNB))
    print("Scikit learn Naive Bayes teste de aproximacao normal")
    print("X +- 1,96sig = ", X_sciNB , "+-", 1.96 * sig_sciNB, "\n")
    
## TEST MCNEMAR
def mcNemarTest():
    # SVM VS OWN NB
    svm_pred = SVM.predict(Xs_t)
    good_classified_NB = np.where(NB_res - Ys_t == 0)[0]
    badly_classified_NB = np.where(NB_res - Ys_t != 0)[0]
    good_classified_SVM = np.where(svm_pred - Ys_t == 0)[0]
    badly_classified_SVM = np.where(svm_pred - Ys_t != 0)[0] 
    e01 = len(np.intersect1d(badly_classified_NB, good_classified_SVM)) 
    e10 = len(np.intersect1d(good_classified_NB, badly_classified_SVM)) 
    print("McNemar’s test SVM Vs Own Naive Bayes")
    print("Svm VS own Naive Bayes = ",((np.abs(e01-e10) - 1)**2) / (e01+e10),"\n")
    svm_pred = SVM.predict(Xs_t)
    scik_nb_pred = GNBSKL.predict(Xs_t)
    good_classified_sciNB = np.where(scik_nb_pred - Ys_t == 0)[0]
    badly_classified_sciNB = np.where(scik_nb_pred - Ys_t != 0)[0]
    good_classified_SVM = np.where(svm_pred - Ys_t == 0)[0]
    badly_classified_SVM = np.where(svm_pred - Ys_t != 0)[0] 
    e01 = len(np.intersect1d(badly_classified_sciNB, good_classified_SVM)) 
    e10 = len(np.intersect1d(good_classified_sciNB, badly_classified_SVM)) 
    print("McNemar’s test SVM Vs Scikit-learn Naive Bayes")
    print("Svm vs scikit learn Naive Bayes = ",((np.abs(e01-e10) - 1)**2) / (e01+e10),"\n")
    return badly_classified_sciNB, good_classified_sciNB, scik_nb_pred

## COMPARACAO DOS CLASSIFICADORES NB
def sckNBvsownNB():
    good_classified_sciNB = np.where(scik_nb_pred - Ys_t == 0)[0]
    badly_classified_sciNB = np.where(scik_nb_pred - Ys_t != 0)[0]
    good_classified_NB = np.where(NB_res - Ys_t == 0)[0]
    badly_classified_NB = np.where(NB_res - Ys_t != 0)[0] 
    e01 = len(np.intersect1d(badly_classified_NB, good_classified_sciNB)) 
    e10 = len(np.intersect1d(good_classified_NB, badly_classified_sciNB)) 
    print("McNemar’s test Own Naive Bayes Vs Scikit learn Naive Bayes")
    print("Own Naive Bayes Vs Scikit learn Naive Bayes = ",((np.abs(e01-e10) - 1)**2) / (e01+e10),"\n")
    

Xs_tr,Ys_tr,Xs_t, Ys_t = loadData(FileTrain="TP1_train.tsv", FileTest="TP1_test.tsv")
GNBSKL = GaussianNB_SKL(Xs_tr,Ys_tr)
NBWKDE, logAPrioProb = NBwithKDE(Xs_tr, Ys_tr)
SVM = SVMwith_radialBF(Xs_tr, Ys_tr)
plots()
svm_error, NB_res, ownNB_error, NBSKL_error = errorTest()
apprNTest()
badly_classified_sciNB, good_classified_sciNB, scik_nb_pred = mcNemarTest()
sckNBvsownNB()
