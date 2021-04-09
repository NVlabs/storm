#
# Copyright (c) 2020-2021 NVIDIA CORPORATION. All rights reserved.
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

import os
import yaml

# get paths
def get_module_path():
    path = os.path.dirname(__file__)
    return path

def get_root_path():
    path = os.path.dirname(get_module_path())
    return path

def get_content_path():
    root_path = get_root_path()
    path = os.path.join(root_path,'content')
    return path

def get_configs_path():
    content_path = get_content_path()
    path = os.path.join(content_path,'configs')
    return path
def get_assets_path():
    content_path = get_content_path()
    path = os.path.join(content_path,'assets')
    return path

def get_weights_path():
    content_path = get_root_path()
    path = os.path.join(content_path,'weights')
    return path

def join_path(path1,path2):
    return os.path.join(path1,path2)

def load_yaml(file_path):
    with open(file_path) as file:
        yaml_params = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_params

# get paths for urdf
def get_urdf_path():
    content_path = get_content_path()
    path = os.path.join(content_path,'assets','urdf')
    return path

def get_gym_configs_path():
    config_path = get_configs_path()
    path = os.path.join(config_path, 'gym')
    return path

def get_mpc_configs_path():
    config_path = get_configs_path()
    path = os.path.join(config_path, 'mpc')
    return path
