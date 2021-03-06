from typing import Callable, Mapping, OrderedDict, Sequence, Dict, Iterable
import pandas as pd
import numpy as np
from pandas import DataFrame

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


class DataProcessor:

    def fit(self, df: DataFrame):
        pass
    
    def execute(self, df: DataFrame) -> DataFrame:
        return df

class StandardizableDataProcessor(DataProcessor):

    def __init__(self, standardizers : Iterable[object]):
        self.__standardizers = standardizers

    def fit(self, df: DataFrame):
        for s in self.__standardizers:
            s.fit(df)

class SequentialDataProcessor(DataProcessor):

    def __init__(self, processors : Sequence[DataProcessor]):
        self.__processors = processors

    def fit(self, df: DataFrame):
        for p in self.__processors:
            p.fit(df)

    def execute(self, df: DataFrame) -> DataFrame:
        proccesedDF = df
        for p in self.__processors:
            proccesedDF = p.execute(proccesedDF)

        return proccesedDF

class StandardDataProcessor(DataProcessor):

    def __init__(self, uselessColumns : Iterable[str] = None, groubByConfig : AggConfig  = None, orderByColumns : Sequence[str] = None, newColumns : Mapping[str, Mapping[str, object]] = None, digitColumns : Mapping[str, Mapping[str, object]] = None, excludeRows : Iterable[Callable[[pd.DataFrame], object]] = None):
        self.__uselessColumns = uselessColumns
        self.__groubByConfig = groubByConfig
        self.__orderByColumns = orderByColumns
        self.__newColumns = newColumns
        self.__digitColumns = digitColumns
        self.__excludeRows = excludeRows


    def execute(self, df: DataFrame) -> DataFrame:

        if not self.__uselessColumns is None:
            df.drop(self.__uselessColumns, axis=1, inplace=True)

        if not self.__newColumns is None:
            for nc in self.__newColumns.keys():
                ncConfig = self.__newColumns[nc]

                if 'func' in ncConfig.keys():
                    ncProc = ncConfig['func']
                    df[nc] = ncProc(df)

                if 'apply' in ncConfig.keys():
                    ncApplyProc = ncConfig['apply']
                    df[nc] = df.apply(lambda l : ncApplyProc(df, l), axis=1)

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
                newColDFs = [df]
                for i in range(nbValue):
                    ndf = df[colToDigitalize].map(lambda x : mapFunc(i, x))
                    ndf.name = colNames[i]
                    newColDFs.append(ndf)

                df = pd.concat(newColDFs, axis=1)
                    #df[colNames[i]] = df[colToDigitalize].map(lambda x : mapFunc(i, x))
                
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

        if not self.__excludeRows is None:
            
            condition = None
            for cnd in self.__excludeRows:
                if condition is None:
                    condition = cnd(df)
                else:
                    condition |= cnd(df)

            if not condition is None: df.drop(df[condition].index, axis=0, inplace=True)

        if not self.__orderByColumns is None:
            df.sort_values(self.__orderByColumns, inplace=True)

        

        return df

class ForwardDataProcessor(DataProcessor):
    
    def __init__(self, targetField : str, epochField : str, mainFields, nextEpoch : Callable[[object], object], newTargetColName : str = None, preprocessor : DataProcessor = None, dropTargetNa : bool = True):
        self.__targetField = targetField
        self.__mainFields = mainFields
        self.__epochField = epochField
        self.__preprocessor = preprocessor
        self.__newTargetColName = (targetField + '_future') if newTargetColName is None else newTargetColName
        self.__nextEpoch = nextEpoch
        self.__dropTargetNa = dropTargetNa
    
    def __futureTargetValue(self, df : DataFrame, r):
        condition = (df[self.__epochField] == self.__nextEpoch(r[self.__epochField]))
        
        for f in self.__mainFields:
            condition &= (df[f] == r[f])
            
        v0 = df[condition][self.__targetField]
        
        return np.nan if len(v0.values) == 0 else v0.values[0]
        
    
    def fit(self, df: DataFrame):
        if self.__preprocessor is None : return
        
        self.__preprocessor.fit(df)
        
        
    def execute(self, df: DataFrame) -> DataFrame:
        if not self.__preprocessor is None : df = self.__preprocessor.execute(df)
        
        df[self.__newTargetColName] = df.apply(lambda r : self.__futureTargetValue(df, r), axis = 1)
        
        if self.__dropTargetNa: df = df[df[self.__newTargetColName].notna()]
        
        return df
