from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer

def resample_data(oXtrain, oYtrain, method = "normal", seed = 42):

    # NORMAL
    if (method in ["normal", "weighted"]):
        sXtrain = oXtrain
        Ytrain = oYtrain

    # RANDOM UNDER SAMPLING
    elif(method == "under"):
        rus = RandomUnderSampler(random_state=seed)
        sXtrain, Ytrain = rus.fit_resample(oXtrain, oYtrain)
    

    return sXtrain, Ytrain

geometric_mean_scorer = make_scorer(geometric_mean_score, greater_is_better=True)
    
