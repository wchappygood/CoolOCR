# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   BasicProjectException
   Description :
   Author:      qisen
   date:        2020/7/6 / 上午9:17
-------------------------------------------------
   Change Activity:
                   2020/7/6:
-------------------------------------------------
"""


# ================================== pydoc ==================================
# define the basic project exception
# ================================ end pydoc ================================

class BasicProjectException(Exception):
    """
    -------------------------------------------------------------------------------------------
    This BasicProjectException class is used to : define the basic project exception and
    initialize the project exception
    -------------------------------------------------------------------------------------------
    """
    exceptionCode = -1
    exceptionMsg = "Message"
    errorCode = -2

    def __init__(self, exceptionCode=None, exceptionMsg=None, errorCode=None):
        """
        -------------------------------------------------------------------------------------------
        This function is used to : initialize the project exception
        -------------------------------------------------------------------------------------------
        Params
            :param exceptionCode: describe the exception code
            :param exceptionMsg: the exception error message
            :param errorCode: the error code
        Return
        -------------------------------------------------------------------------------------------
        """
        if exceptionMsg is not None:
            self.exceptionMsg = exceptionMsg

        if exceptionCode is not None:
            self.exceptionCode = exceptionCode

        if errorCode is not None:
            self.errorCode = errorCode

    @classmethod
    def exceptionProcess(cls, *args, **kwargs):
        """
        -------------------------------------------------------------------------------------------
        This function is used to : exception process function
        -------------------------------------------------------------------------------------------
        Params
            :param :
        Return
        -------------------------------------------------------------------------------------------
        """
        pass

# end code
