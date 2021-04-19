#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#


def load_struct_from_dict(struct_instance, dict_instance):
    """This function populates a struct recursively from a dictionary.

    Assumptions:
    Currently the struct cannot have a dictionary for one of the objects as that will start recursing.
    """
    #print(dict_instance)
    for key in dict_instance.keys():
        if(hasattr(struct_instance, key)):
            if(isinstance(dict_instance[key],dict)):
                sub_struct = load_struct_from_dict(getattr(struct_instance,key), dict_instance[key])
                setattr(struct_instance,key,sub_struct)
            else:
                setattr(struct_instance,key,dict_instance[key])
    return struct_instance
