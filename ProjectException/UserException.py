# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   UserException
   Description :
   Author:      qisen
   date:        2020/7/15 / 下午5:21
-------------------------------------------------
   Change Activity:
                   2020/7/15:
-------------------------------------------------
"""

from ProjectException.BasicProjectException import BasicProjectException


# ================================== pydoc ==================================
# user illegal input parameter
# ================================ end pydoc ================================
class UserIllegalInputParamError(BasicProjectException):
    """
    -------------------------------------------------------------------------------------------
    This UserIllegalInputParamError class is used to : define the user illegal input parameter
    -------------------------------------------------------------------------------------------
    """
    exceptionCode = 600
    exceptionMsg = ">>>ERR: You have illegal input param."
    errorCode = 1000

    @classmethod
    def exceptionProcess(cls, userIllegalInput):
        """
        -------------------------------------------------------------------------------------------
        This function is used to :process the user input error
        -------------------------------------------------------------------------------------------
        Params
            :param userIllegalInput: the user wrong input
        Return
        -------------------------------------------------------------------------------------------
        """
        print("*" * 20)
        print("ERROR CODE: {}.".format(cls.errorCode))
        print(cls.exceptionMsg)
        print("Your Input is: {}".format(userIllegalInput))
        print("*" * 20)


class UserIllegalFilePathError(BasicProjectException):
    """
    -------------------------------------------------------------------------------------------
    This UserIllegalFilePathError class is used to : define the user illegal input file path
    -------------------------------------------------------------------------------------------
    """
    exceptionCode = 601
    exceptionMsg = ">>>ERR: Your file path is wrong."
    errorCode = 1001

    @classmethod
    def exceptionProcess(cls, userIllegalFilePath):
        """
        -------------------------------------------------------------------------------------------
        This function is used to :process the illegal file path error
        -------------------------------------------------------------------------------------------
        Params
            :param userIllegalFilePath: user illegal file path
        Return
        -------------------------------------------------------------------------------------------
        """
        print("*" * 20)
        print("ERROR CODE: {}.".format(cls.errorCode))
        print(cls.exceptionMsg)
        print("Your Input is: {}".format(userIllegalFilePath))
        print("*" * 20)
