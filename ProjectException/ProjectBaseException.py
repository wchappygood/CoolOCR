# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   ProjectBaseException
   Description :
   Author:      coolCIman
   date:        2020/7/15 / 下午1:56
-------------------------------------------------
   Change Activity:
                   2020/7/15:
-------------------------------------------------
"""


# ================================== pydoc ==================================
# define the project base exception.
# ================================ end pydoc ================================
class ProjectBaseException(Exception):
    """
    -------------------------------------------------------------------------------------------
    This ProjectBaseException class is used to :initialize the project base exception
    -------------------------------------------------------------------------------------------
    """
    exceptionCode = -1
    exceptionMsg = "Exception Message"
    errorCode = -1

    def __init__(self, exceptionCode=None, exceptionMsg=None, errorCode=None):
        """
        -------------------------------------------------------------------------------------------
        This function is used to :initialize the base exception
        -------------------------------------------------------------------------------------------
        Params
            :param exceptionMsg: describe the exception message
            :param exceptionCode: setting the exception code
            :param errorCode: define the error code
        Return
        -------------------------------------------------------------------------------------------
        """
        if exceptionMsg is not None:
            self.exceptionMsg = exceptionMsg

        if exceptionCode is not None:
            self.exceptionCode = exceptionCode

        if errorCode is not None:
            self.errorCode = errorCode
        pass

    @classmethod
    def exceptionProcess(cls, *args, **kwargs):
        """
        -------------------------------------------------------------------------------------------
        This function is used to :process exception function
        -------------------------------------------------------------------------------------------
        Params
            :param :NONE
        Return
        -------------------------------------------------------------------------------------------
        """
        pass
