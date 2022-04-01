from typing import Callable, Mapping, Iterable, OrderedDict
import pandas as pd
from .processors import DataProcessor

from sklearn.model_selection import train_test_split

class InputMan:
    def normalize(self) -> pd.DataFrame:
        pass
class StandardInputMan(InputMan):

    def __init__(self, processor : DataProcessor):
        self.__processor = processor
        
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.__processor.execute(df)

class MultiDFInputMan(InputMan):

    def __init__(self, processor : DataProcessor, dfs : Iterable[pd.DataFrame]):
         self.__processor = processor
         self.__dfs = dfs

    def normalize(self) -> pd.DataFrame:
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

    def loadSheet(self, sheet_name : str) -> pd.DataFrame:
        return pd.read_excel(self.__file, sheet_name = sheet_name)
        

    def normalize(self) -> pd.DataFrame:
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

    def __return(self, cb : Callable[[str, object]], step: str, data : object):
        if cb is None: return

        cb(step, data)

    def acquireKnowledge(self, df : pd.DataFrame, targetCol : str, testSize = 0.2, firstDataProcessor : DataProcessor = None, ramdomState = None, getTempData : Callable[[str, object], object] = None):
        res = OrderedDict()

        if not firstDataProcessor is None:
            df = firstDataProcessor.execute(df)

        trainDF, testDF = train_test_split(df, test_size = testSize, random_state=ramdomState)

        self.__return(getTempData, "trainset", trainDF)
        self.__return(getTempData, "testset", testDF)
        
        imDFs = InputManDataFrames(trainDF, self.__dataProcessors)

        processors = OrderedDict()

        res['processors'] = processors

        for imDF in imDFs:
            #print(f'Data processor : {imDF.name}')
            #print(imDF.df.head())

            xTrain, yTrain = self.__prepareForLearning(imDF.df, targetCol)

            tDF = testDF.copy()
            tDF = imDF.processor.execute(tDF)
            xTest, yTest = self.__prepareForLearning(tDF, targetCol)

            procProps = OrderedDict()
            self.__return(getTempData, imDF.name + "_xTrain", xTrain)
            self.__return(getTempData, imDF.name + "_yTrain", yTrain)

            self.__return(getTempData, imDF.name + "_xTest", xTest)
            self.__return(getTempData, imDF.name + "_yTest", yTest)

            processors[imDF.name] = procProps

            regressionDict = OrderedDict()
            procProps['regressors'] = regressionDict
            for rk in self.__regressors.keys():

                regressor = self.__regressors[rk]()
                regressor.fit(xTrain, yTrain)

                yTestPred = regressor.predict(xTest)

                regProps = OrderedDict()
                regProps['model'] = regressor

                self.__return(getTempData, imDF.name + "_" + rk + "_pred", yTestPred)
                #regProps['pred'] = yTestPred

                regressionDict[rk] = regProps
                
                metricsResDict = OrderedDict()
                regProps['scoring'] = metricsResDict
                for emk in self.__evalMetrics.keys():
                    em = self.__evalMetrics[emk]

                    testScore = em(yTest, yTestPred)

                    metricsResDict[emk] = testScore

                    
                    
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