# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   UserInputException
   Description :
   Author:      qisen
   date:        2020/7/15 / 下午3:45
-------------------------------------------------
   Change Activity:
                   2020/7/15:
-------------------------------------------------
"""
from ProjectException.ProjectBaseException import ProjectBaseException


# ================================== pydoc ==================================
# User exception Error
# ================================ end pydoc ================================
class UserIllegalInputParam(ProjectBaseException):
    """
    -------------------------------------------------------------------------------------------
    This UserIllegalInputParam class is used to :define the user illegal input param
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


class UserIllegalFilepath(ProjectBaseException):
    """
    -------------------------------------------------------------------------------------------
    This UserIllegalFilepath class is used to :define the user illegal file path error
    -------------------------------------------------------------------------------------------
    """
    exceptionCode = 601
    exceptionMsg = ">>>ERR: Your file path is Wrong."
    errorCode = 1001

    @classmethod
    def exceptionProcess(cls, userInputFilePath):
        """
        -------------------------------------------------------------------------------------------
        This function is used to :process the user input file path error
        -------------------------------------------------------------------------------------------
        Params
            :param userInputFilePath: the user input file path
        Return
        -------------------------------------------------------------------------------------------
        """
        print("*" * 20)
        print("ERROR CODE: {}.".format(cls.errorCode))
        print(cls.exceptionMsg)
        print("Your Input is: {}".format(userInputFilePath))
        print("*" * 20)
