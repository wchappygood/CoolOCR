# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   DataException
   Description :
   Author:      qisen
   date:        2020/7/15 / 下午3:44
-------------------------------------------------
   Change Activity:
                   2020/7/15:
-------------------------------------------------
"""
from ProjectException.ProjectBaseException import ProjectBaseException


# ================================== pydoc ==================================
# define the different exception to process in coolOCR project

# Data exception error
# ================================ end pydoc ================================
class DataRequireError(ProjectBaseException):
    """
    -------------------------------------------------------------------------------------------
    This DataRequireError class is used to : define the different datatype or status error
    -------------------------------------------------------------------------------------------
    """
    exceptionCode = 700
    exceptionMsg = ">>>ERR: The Program meet a data require type error."
    errorCode = 2000

    @classmethod
    def exceptionProcess(cls, showInformation):
        """
        -------------------------------------------------------------------------------------------
        This function is used to : process the data require error.
        -------------------------------------------------------------------------------------------
        Params
            :param showInformation: show error information
        Return
        -------------------------------------------------------------------------------------------
        """
        print("*" * 20)
        print("ERROR CODE: {}.".format(cls.errorCode))
        print(cls.exceptionMsg)
        print("Your Data status is: {}".format(showInformation))
        print("*" * 20)


class DataInputSizeError(ProjectBaseException):
    """
    -------------------------------------------------------------------------------------------
    This DataInputSizeError class is used to :define the data input length error
    -------------------------------------------------------------------------------------------
    """
    exceptionCode = 701
    exceptionMsg = ">>>ERR: The Program meet a data transform length error."
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
