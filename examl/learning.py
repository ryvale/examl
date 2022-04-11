from typing import Callable, Mapping, Iterable, OrderedDict
import pandas as pd
from pandas import DataFrame
from .processors import DataProcessor

from sklearn.model_selection import train_test_split, GridSearchCV

class InputMan:
    def normalize(self) -> pd.DataFrame:
        pass
class StandardInputMan(InputMan):

    def __init__(self, processor : DataProcessor):
        self.__processor = processor
        
    def execute(self, df: DataFrame) -> DataFrame:
        return self.__processor.execute(df)

class MultiDFInputMan(InputMan):

    def __init__(self, processor : DataProcessor, dfs : Iterable[DataFrame]):
         self.__processor = processor
         self.__dfs = dfs

    def normalize(self) -> DataFrame:
        dfArray = []

        for df in self.__dfs:
            processedDF = self.__processor.execute(df)
            dfArray.append(processedDF)

        return pd.concat(dfArray, ignore_index=True)


class XLSFileInputMan(InputMan):

    def __init__(self, file : str, processor : Mapping[str, DataProcessor] ) -> None:
        super().__init__()
        self.__file = file
        self.__processor = processor

    def loadSheet(self, sheet_name : str) -> DataFrame:
        return pd.read_excel(self.__file, sheet_name = sheet_name)
        

    def normalize(self) -> DataFrame:
        sheetDataArray = []

        for sheet_name in self.__processor.keys():

            dfRawData = self.loadSheet(sheet_name)

            processor = self.__processor[sheet_name]

            dfRawData = processor.execute(dfRawData)

            sheetDataArray.append(dfRawData)

        return pd.concat(sheetDataArray, ignore_index=True)

class SupervisedLearner:

    def __init__(self, dataProcessors : Mapping[str, DataProcessor], regressors : Mapping[str, Callable[[], object]], evalMetrics : Mapping[str, Callable[[object, object], object]]):
        self.__dataProcessors = dataProcessors
        self.__regressors = regressors
        self.__evalMetrics = evalMetrics

    def __prepareForLearning(self, df : pd.DataFrame, targetCol : str):
        x = df.drop(targetCol, axis = 1)
        y = df[targetCol]

        return x, y

    def __return(self, cb : Callable[[str, object], object], step: str, data : object=None):
        if cb is None: return

        cb(step, data)

    def acquireKnowledge(self, df : pd.DataFrame, targetCol : str, testSize = 0.2, firstDataProcessor : DataProcessor = None, 
        ramdomState = None, getTempData : Callable[[str, object], object] = None, trainMetrics : bool = False, excludeRegressors : Iterable[object] = []):
        res = OrderedDict()

        if not firstDataProcessor is None:
            self.__return(getTempData, "LOG:preprocessing starts ...")
            df = firstDataProcessor.execute(df)
            self.__return(getTempData, "LOG:preprocessing ended")

        self.__return(getTempData, "LOG:split data starts")
        trainDF, testDF = train_test_split(df, test_size = testSize, random_state=ramdomState)
        self.__return(getTempData, "LOG:split data ended")

        self.__return(getTempData, "trainset", trainDF)
        self.__return(getTempData, "testset", testDF)
        
        imDFs = InputManDataFrames(trainDF, self.__dataProcessors)

        processors = OrderedDict()

        res['processors'] = processors

        self.__return(getTempData, "LOG:Training process starts ...")
        for imDF in imDFs:
            xTrain, yTrain = self.__prepareForLearning(imDF.df, targetCol)

            #self.__return(getTempData, f"LOG:Additional data processing ({imDF.name}) starts ...")
            tDF = testDF.copy()
            tDF = imDF.processor.execute(tDF)
            xTest, yTest = self.__prepareForLearning(tDF, targetCol)

            
            procProps = OrderedDict()
            #self.__return(getTempData, "LOG:Additional data processing ended")

            self.__return(getTempData, imDF.name + "_xTrain", xTrain)
            self.__return(getTempData, imDF.name + "_yTrain", yTrain)

            self.__return(getTempData, imDF.name + "_xTest", xTest)
            self.__return(getTempData, imDF.name + "_yTest", yTest)

            processors[imDF.name] = procProps

            regressionDict = OrderedDict()
            procProps['regressors'] = regressionDict
            for rk in self.__regressors.keys():

                if rk in excludeRegressors: 
                    self.__return(getTempData, f"LOG:'{rk}' Regressor skip")
                    continue

                self.__return(getTempData, f"LOG:'{rk}' Regression process starts ...")
                regressor = self.__regressors[rk]()
                regressor.fit(xTrain, yTrain)
                self.__return(getTempData, "LOG:Regression completed (fit)")

                #self.__return(getTempData, f"LOG:'{rk}' Regressor test predictions starts ...")
                yTestPred = regressor.predict(xTest)
                #self.__return(getTempData, f"LOG:'{rk}' Regressor test predictions ended")

                regProps = OrderedDict()
                regProps['model'] = regressor

                self.__return(getTempData, imDF.name + "_" + rk + "_pred", yTestPred)

                regressionDict[rk] = regProps

                #self.__return(getTempData, f"LOG:Getting '{rk}' Regressor test score ...")
                natifScore = regressor.score(xTest, yTest)
                regProps['natif-test-score'] = natifScore
                self.__return(getTempData, f"LOG:test score is : {natifScore}", natifScore)
                
                #self.__return(getTempData, f"LOG:'{rk}' Regressor test scoring starts ...")
                testMetricsResDict = OrderedDict()
                regProps['scoring-test'] = testMetricsResDict
                for emk in self.__evalMetrics.keys():
                    em = self.__evalMetrics[emk]

                    testScore = em(yTest, yTestPred)
                    
                    testMetricsResDict[emk] = testScore
                    self.__return(getTempData, f"SCORE:'{emk}' score getted", testScore)

                self.__return(getTempData, "LOG:Regressor test scoring completed")

                if trainMetrics:
                    #self.__return(getTempData, f"LOG:'{rk}' Regressor train scoring starts ...")
                    yTrainPred = regressor.predict(xTrain)
                    trainMetricsResDict = OrderedDict()
                    regProps['scoring-train'] = trainMetricsResDict
                    for emk in self.__evalMetrics.keys():
                        em = self.__evalMetrics[emk]

                        testScore = em(yTrain, yTrainPred)

                        trainMetricsResDict[emk] = testScore
                    self.__return(getTempData, "LOG:Regressor train scoring ended")
            self.__return(getTempData, "LOG:Training process ended")
        return res

    def __learningReportParams(self, knowledge : Mapping[str, object]):
        processors = knowledge['processors']
        
        key0 = next( iter(processors.keys()))

        regKey0 = next(iter(processors[key0]['regressors'].keys()))

        nbReg = len(processors[key0]['regressors'])
        oneReg = processors[key0]['regressors'][regKey0]

        evalMetricsKeys = [k for k in oneReg['scoring-train'].keys()]

        aProc = []
        aReg = []
        aScore = dict()

        for pk in processors.keys():
            proc = processors[pk]
            aProc += [pk for i in range(nbReg)]

            for rk in proc['regressors'].keys():
                aReg.append(rk)
                reg = proc['regressors'][rk]

                for sk in evalMetricsKeys:
                    trainSK = "Train-" + sk

                    if trainSK in aScore.keys():
                        trainScores = aScore[trainSK]
                    else:
                        trainScores = []
                        aScore[trainSK] = trainScores

                    trainScores.append(reg['scoring-train'][sk])

                    testSK = "Test-" + sk
                    if testSK in aScore.keys():
                        testScores = aScore[testSK]
                    else:
                        testScores = []
                        aScore[testSK] = testScores
                        
                    testScores.append( reg['scoring-test'][sk])                      

        return aProc, aReg, aScore, evalMetricsKeys


    def generateReport(self, knowledgeCollection : Mapping[str, object], order) -> DataFrame:
        initProcs = []
        procs = []
        regs = []

        evals = OrderedDict()

        for ip in  knowledgeCollection.keys():
            knowledge = knowledgeCollection[ip]
            knowledgeParams = self.__learningReportParams(knowledge)

            initProcs += [ip for i in range(len(knowledgeParams[0]))]
            procs += knowledgeParams[0]
            regs += knowledgeParams[1]

            evalMetricsKeys = knowledgeParams[3]

            for ek in evalMetricsKeys:
                trainEK = "Train-" + ek
                if trainEK in evals.keys():
                    trScores = evals[trainEK]
                else:
                    trScores = []
                trScores += knowledgeParams[2][trainEK]
                evals[trainEK] = trScores

            for ek in evalMetricsKeys:
                testEK = "Test-" + ek
                if testEK in evals.keys():
                    teScores = evals[testEK]
                else:
                    teScores = []
                teScores += knowledgeParams[2][testEK]
                evals[testEK] = teScores

        data = {'Collection name' : initProcs, 'Processors' : procs, 'Regressor' : regs}

        for ek in evals.keys():
            data[ek] = evals[ek]

        regReportDF = DataFrame(data).sort_values(order, ascending=False)

        return regReportDF
        
        

    def optimize(estimator : Callable[[], object], tunedParameters, x, y, scoring : Iterable[str]):

        res = OrderedDict()
        for s in scoring:
            gscv = GridSearchCV(estimator(), tunedParameters, scoring=s)
            gscv.fit(x, y)

            params =  dict()

            params['best-params'] = gscv.best_params_
            params['cv-results'] = gscv.cv_results_

            res[s] = params
        return res


        
class InputManDataFrame:

    def __init__(self, name : str, df : pd.DataFrame, processor : DataProcessor):
        self.__df = df
        self.__name = name
        self.__processor = processor

    def __get_df(self):
        return self.__df

    def __get_name(self):
        return self.__name

    def __get_processor(self):
        return self.__processor

    df = property(__get_df)
    name = property(__get_name)
    processor = property(__get_processor)


class InputManDataFrames:

    def __init__(self, df : pd.DataFrame, processors : Mapping[str, DataProcessor]):
        self._df = df
        self._processors = processors

    def __iter__(self):
        return InputManIterator(self)

class InputManIterator:

    def __init__(self, inputManDFs : InputManDataFrames):
        self.__inputManDFs = inputManDFs
        keys = inputManDFs._processors.keys()
        self.__index = iter(keys)


    def __next__(self):

        k = next(self.__index)

        processor =  self.__inputManDFs._processors[k]

        df = self.__inputManDFs._df.copy()

        df = processor.execute(df)

        return InputManDataFrame(k, df, processor)