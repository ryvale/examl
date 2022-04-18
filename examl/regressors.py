from typing import Callable
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures

class PolynomialRegressor:
    
    def __init__(self, regressor : Callable[[], object], degree = 2 ):
        self.__regressor = regressor()
        self.__poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        
    def __normalizeDF(self, df : DataFrame):
        data = dict()
        for col in  df.columns:
            data[col] = df[col].squeeze()
            
        return DataFrame(data)
    
    def fit(self, xDF, YDF):
        tmpDF = self.__normalizeDF(xDF)

        polyFeatures = self.__poly.fit_transform(tmpDF)
        
        recodeY = YDF.squeeze()
        self.__regressor.fit(polyFeatures, recodeY)
        
    def predict(self, xDF):
        tmpDF = self.__normalizeDF(xDF)
        polyFeatures = self.__poly.fit_transform(tmpDF)
        
        return self.__regressor.predict(polyFeatures)
    
    def score(self, x, y, sample_weight=None):
        tmpDF = self.__normalizeDF(x)
        polyFeatures = self.__poly.fit_transform(tmpDF)
        return self.__regressor.score(polyFeatures, y, sample_weight = sample_weight)


class StandardizableRegressor:
    def __init__(self, regressor : Callable[[], object], standardizer):
        self.__regressor = regressor()
        self.__standardizer = standardizer


    def __normalizeDF(self, df : DataFrame):
        return self.__standardizer.transform(df)


    def fit(self, xDF, YDF):
        stdDF = self.__normalizeDF(xDF)
        
        self.__regressor.fit(stdDF, YDF)

    def predict(self, xDF):
        stdDF = self.__normalizeDF(xDF)
        
        return self.__regressor.predict(stdDF)

    def score(self, x, y, sample_weight=None):
        stdDF = self.__normalizeDF(x)

        return self.__regressor.score(stdDF, y, sample_weight = sample_weight)

    def __get_estimator(self):
        return self.__regressor

    estimator = property(__get_estimator)