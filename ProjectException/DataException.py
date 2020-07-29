# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   DataException
   Description :
   Author:      qisen
   date:        2020/7/15 / 下午5:20
-------------------------------------------------
   Change Activity:
                   2020/7/15:
-------------------------------------------------
"""
from ProjectException.BasicProjectException import BasicProjectException


# ================================== pydoc ==================================
# Data framework exception
# ================================ end pydoc ================================
class DataTypeRequireError(BasicProjectException):
    """
    -------------------------------------------------------------------------------------------
    This DataTypeRequireError class is used to : define the program process data with different data type or status
    -------------------------------------------------------------------------------------------
    """
    exceptionCode = 700
    exceptionMsg = ">>>ERR: The Program meet a data require type error."
    errorCode = 2000

    @classmethod
    def exceptionProcess(cls, showInformation):
        """
        -------------------------------------------------------------------------------------------
        This function is used to :process the data require error
        -------------------------------------------------------------------------------------------
        Params
            :param showInformation: the information of data require error
        Return
        -------------------------------------------------------------------------------------------
        """
        print("*" * 20)
        print("ERROR CODE: {}.".format(cls.errorCode))
        print(cls.exceptionMsg)
        print("Your Data status is: {}".format(showInformation))
        print("*" * 20)


class DataShapeSizeError(BasicProjectException):
    """
    -------------------------------------------------------------------------------------------
    This DataShapeSizeError class is used to : define the project process data with different data shape
    -------------------------------------------------------------------------------------------
    """
    exceptionCode = 701
    exceptionMsg = ">>>ERR: The Program meet a data transform shape error."
    errorCode = 2001

    @classmethod
    def exceptionProcess(cls, requireLength, inputLength):
        """
        -------------------------------------------------------------------------------------------
        This function is used to :process the data input error
        -------------------------------------------------------------------------------------------
        Params
            :param requireLength: the program need data length
            :param inputLength: the program get data length
        Return
        -------------------------------------------------------------------------------------------
        """
        print("*" * 20)
        print("ERROR CODE: {}.".format(cls.errorCode))
        print(cls.exceptionMsg)
        print("Your input data dim is: {},the program need data dim is {}".format(inputLength, requireLength))
        print("*" * 20)
