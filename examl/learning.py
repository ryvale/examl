from typing import Mapping, Sequence, Dict, Iterable
import pandas as pd
from collections import OrderedDict;

class InputMan:
    def normalize(self) -> pd.DataFrame:
        pass


class DataProcessor:
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

class AggConfig:
    def __init__(self, gbColumns : Sequence[str], aggFuncConfig : Sequence[Dict] ):
        self.__gbColumns = gbColumns
        self.__aggFuncConfig = aggFuncConfig
        

    def __get_gbColumns(self):
        return self.__gbColumns

    def __get_aggFuncConfig(self):
        return self.__aggFuncConfig

    gbColumns = property(__get_gbColumns)
    aggFuncConfig = property(__get_aggFuncConfig)



class StandardDataProcessor(DataProcessor):

    def __init__(self, uselessColumns : Iterable[str] = None, groubByConfig : AggConfig  = None, orderByColumns : Sequence[str] = None, newColumns : Mapping[str, Mapping[str, object]] = None, digitColumns : Mapping[str, Mapping[str, object]] = None):
        self.__uselessColumns = uselessColumns
        self.__groubByConfig = groubByConfig
        self.__orderByColumns = orderByColumns
        self.__newColumns = newColumns
        self.__digitColumns = digitColumns


    def execute(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.__uselessColumns is None:
            df.drop(self.__uselessColumns, axis=1, inplace=True)

        if not self.__newColumns is None:
            for nc in self.__newColumns.keys():
                ncConfig = self.__newColumns[nc]
                ncProc = ncConfig['func']
                df[nc] = ncProc(df)

                if 'drop' in ncConfig.keys():
                    df.drop(ncConfig['drop'], axis=1, inplace=True)

        if not self.__digitColumns is None:
            for colToDigitalize in self.__digitColumns.keys():
                digitConfig = self.__digitColumns[colToDigitalize]
                nbValue = digitConfig['nbValue']
                if 'colNames' in digitConfig:
                    colNames = digitConfig['colNames']
                elif 'prefix'  in digitConfig:
                    colNames = [digitConfig['prefix'] + "_" + str(i) for i in range(1, nbValue+1)]
                else:
                    colNames = [colToDigitalize + "_" + str(i) for i in range(1, nbValue+1)]

                mapFunc = digitConfig['mapFunc']
                for i in range(nbValue):
                    df[colNames[i]] = df[colToDigitalize].map(lambda x : mapFunc(i, x))
                
                if 'drop' in digitConfig:
                    if digitConfig['drop']:
                        df.drop(ncConfig['drop'], axis=1, inplace=True)


        if not self.__groubByConfig is None:

            aggParams = {}
            aggFieldNames = OrderedDict()

            for gbConf in  self.__groubByConfig.aggFuncConfig:

                for fieldName in gbConf.keys():
                    if not fieldName in aggParams:
                        aggParams[fieldName] = []

                    aggFn = gbConf[fieldName]

                    if fieldName in aggFieldNames.keys():
                        fieldsPos = aggFieldNames[fieldName]
                    else:
                        fieldsPos=[]
                        aggFieldNames[fieldName] = fieldsPos
                       
                    if isinstance(aggFn, str):
                        aggParams[fieldName].append(aggFn)
                        fieldsPos.append(aggFn + "_" + fieldName)
                    else:
                        aggParams[fieldName].append(aggFn['aggFn'])
                        colName = aggFn['colName'] if 'colName' in aggFn.keys() else aggFn['aggFn'] + '_' + fieldName
                        fieldsPos.append(colName)

            fieldNames = []
            for afn in aggFieldNames.keys():
                fieldNames += aggFieldNames[afn]

            
            dfgb = df.groupby(self.__groubByConfig.gbColumns).agg(aggParams)
            dfgb.columns = fieldNames

            df = dfgb.reset_index()

        if not self.__orderByColumns is None:
            df.sort_values(self.__orderByColumns, inplace=True)

        return df


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
        