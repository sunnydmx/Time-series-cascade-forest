import csv
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
import winsound


def writeCsv(dataset, trainingTime, testingTime, accuracy, n_bins, bin_subsample, n_jobs, type, n_estimators, n_tree):
    row = [dataset, trainingTime, testingTime, accuracy, n_bins, bin_subsample, n_jobs, type, n_estimators, n_tree]
    out = open("result.csv", "a", newline="")
    csv_writer = csv.writer(out, dialect="excel")
    csv_writer.writerow(row)
    out.close()


dataList = [
    # "ACSF1",
    # "Adiac",  
    # "ArrowHead",
    # "Beef",  
    # "BeetleFly",
    # "BirdChicken",
    # "BME",
    # "Car",
    # "CBF",
    # "Chinatown",
    # "ChlorineConcentration",
    # "Coffee",
    # "Computers",
    # "CricketX",  
    # "CricketY",
    # "CricketZ",
    # "Crop",
    # "CinCECGTorso",
    # "DiatomSizeReduction",  
    # "DistalPhalanxOutlineAgeGroup",
    # "DistalPhalanxOutlineCorrect",
    # "DistalPhalanxTW",
    # "DodgerLoopGame",
    # "DodgerLoopWeekend",
    # "ECG200",
    # "ECG5000",  
    # "ECGFiveDays",
    # "EOGHorizontalSignal",
    # "EOGVerticalSignal",
    # "Earthquakes",
    # "ElectricDevices",
    # "EthanolLevel",
    # "FaceAll",  
    # "FaceFour",  
    # "FacesUCR",  
    # "Fish",  
    # "FiftyWords",  
    # "FordA",
    # "FordB",
    # "FreezerRegularTrain",  
    # "FreezerSmallTrain",  
    # "Fungi",
    # "GunPoint",
    # "GunPointAgeSpan",  
    # "GunPointMaleVersusFemale",  
    # "GunPointOldVersusYoung",  
    # "Ham",
    # "HandOutlines",
    # "Haptics",
    # "Herring",
    # "HouseTwenty",
    # "InlineSkate",
    # "InsectEPGRegularTrain",
    # "InsectEPGSmallTrain",  
    # "ItalyPowerDemand",
    # "LargeKitchenAppliances",
    # "Lightning2",
    # "Lightning7",
    # "Mallat",  
    # "Meat",  
    # "MedicalImages", 
    # "MiddlePhalanxOutlineAgeGroup", 
    # "MiddlePhalanxOutlineCorrect", 
    # "MiddlePhalanxTW",
    # "MixedShapesSmallTrain",  
    # "MoteStrain",
    # "NonInvasiveFetalECGThorax1",
    # "NonInvasiveFetalECGThorax2",
    # "OliveOil",  
    # "OSULeaf",  
    # "PhalangesOutlinesCorrect",  
    # "Phoneme",
    # "PigAirwayPressure",
    # "PigArtPressure",
    # "PigCVP",
    # "Plane",
    # "PowerCons",
    # "ProximalPhalanxOutlineAgeGroup", 
    # "ProximalPhalanxOutlineCorrect",
    # "ProximalPhalanxTW",
    # "RefrigerationDevices",
    # "Rock",
    # "ScreenType",
    # "SemgHandGenderCh2",
    # "SemgHandMovementCh2",
    # "SemgHandSubjectCh2",
    # "ShapeletSim",
    # "ShapesAll",
    # "SmallKitchenAppliances",
    # "SmoothSubspace", 
    # "SonyAIBORobotSurface1",
    # "SonyAIBORobotSurface2",
    # "StarLightCurves",
    # "Strawberry",
    # "SwedishLeaf",
    # "Symbols",  
    # "SyntheticControl",
    # "ToeSegmentation1",
    # "ToeSegmentation2",
    # "Trace",
    # "TwoLeadECG",
    # "TwoPatterns",
    # "UMD",
    # "UWaveGestureLibraryAll",  
    # "UWaveGestureLibraryX",
    # "UWaveGestureLibraryY",
    # "UWaveGestureLibraryZ",
    # "Wafer",
    # "Wine",
    # "WordSynonyms",
    # "Worms",
    # "WormsTwoClass",
    # "Yoga",
]
ac = []
for a in range(1):
    for x in dataList:
        dataset_train_name = '../dataset/' + x + '_TRAIN.txt'
        dataset_test_name = '../dataset/' + x + '_TEST.txt'

        dataset_train = pd.read_csv(dataset_train_name, dtype={'code': str}, sep='\t', index_col='Unnamed: 0')
        col = dataset_train.columns.values.tolist()
        col1 = col[1:]
        data_train_x = np.array(dataset_train[col1])
        data_train_y = dataset_train['label']

        dataset_test = pd.read_csv(dataset_test_name, dtype={'code': str}, sep='\t', index_col='Unnamed: 0')
        col = dataset_test.columns.values.tolist()
        col1 = col[1:]
        data_test_x = np.array(dataset_test[col1])
        data_test_y = dataset_test['label']

        X_train, X_test, y_train, y_test = np.array(dataset_train[col1]), np.array(dataset_test[col1]), dataset_train[
            'label'], dataset_test['label']
        
        print("【" + x + "】" + " start.")
        # windowsize1 = int(X_train.shape[1] * 0.6)
        # windowsize2 = int(X_train.shape[1] * 0.8)
        # windowsize3 = int(X_train.shape[1] * 1.0)
        accuracy = []
        start_ = []
        middle_ = []
        end_ = []
        for j in range(1):
            start = time.time()
            model = CascadeForestClassifier(shape_1X=X_train.shape[1],
                                            window=[windowsize1, windowsize2, windowsize3],
                                            random_state=1,
                                            n_jobs=-1, n_estimators=2, n_bins=255,
                                            bin_subsample=2e5,
                                            n_trees=50)
            model.fit(X_train, y_train)
            middle = time.time()
            y_pred = model.predict(X_test)
            end = time.time()
            acc = accuracy_score(y_test, y_pred) * 100
            accuracy.append(accuracy_score(y_test, y_pred))
            start_.append(start)
            middle_.append(middle)
            end_.append(end)
            print("\n" + "第" + str(j+1) + "次 " + "【" + x + "】" + "Testing Accuracy: {:.3f} %".format(acc) + "\n")
            type = "tscf"
            writeCsv(x, '{:.3f}'.format(middle - start),
                     '{:.3f}'.format(end - start),
                     '{:.3f}'.format(acc), model.n_bins,
                     model.bin_subsample, model.n_jobs, type, model.n_estimators, model.n_trees)
        type = "tscf"
        print("[" + str(a) + "]" + '{:.3f}'.format(sum(accuracy)/len(accuracy)))
        ac.append(sum(accuracy)/len(accuracy))

duration = 2000  # millisecond
freq = 1000  # Hz
winsound.Beep(freq, duration)
