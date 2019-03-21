# Copyright 2018 Side Li and Arun Kumar
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup, find_packages, Extension
import numpy

module = Extension('comp', sources=['morpheus/comp.cpp'], include_dirs=[numpy.get_include()])

setup(
    name='MorpheusPy',
    version='1.0',
    packages=find_packages(),
    url='',
    license='',
    author='Side Li, Arun Kumar',
    author_email='s7li@eng.ucsd.edu',
    description='',
    ext_modules=[module]
)
