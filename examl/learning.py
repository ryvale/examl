from typing import Mapping, Iterable, Sequence
import pandas as pd
from processors import DataProcessor

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


class Learner:

    def process(self, inputMan : InputMan):
        normalizedInput = inputMan.normalize()
        
class InputManDataFrames:

    def __init__(self, df : pd.DataFrame, processors : Sequence[DataProcessor]):
        self._df = df
        self._processors = processors

    def __iter__(self):
        return InputManIterator(self)

class InputManIterator:

    def __init__(self, inputManDFs : InputManDataFrames):
        self.__inputManDFs = inputManDFs
        self.__index = 0


    def __next__(self):
        if self.__index < len(self.__inputManDFs._processors):
            processor =  self.__inputManDFs._processors[self.__index]

            res = processor.execute(self.__inputManDFs._df)

            self.__index +=1

            return res
        
        raise StopIteration