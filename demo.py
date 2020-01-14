#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 Rémi Flamary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import argparse
import sys
import subprocess
import os
import configparser

path = os.path.dirname(os.path.realpath(__file__))

def load_config(path):
    cfg=configparser.ConfigParser()
    
    cfg.read(path) 
    dic={}

    for key in cfg:
        if not key=='DEFAULT':
            dic[key]=cfg[key]
    
    return dic

config=load_config(path+'/demos.ini')

parser = argparse.ArgumentParser(description='Demos executions')  
parser.add_argument('-l','--list',action='store_true', help='List all demos')
parser.add_argument('run',default='',nargs='?', help='List all demos')

args = parser.parse_args()

if args.list:
    for key in config:
        print(key)
        print('\t{}'.format(config[key]['text']))
elif args.run:
    if not args.run in config:
        print('Error: "{}" is not a valid demo name. \nExecute demo.py with -l parameter to get a list of demos '.format(args.run))
    else:
        # execute the script
        subprocess.call(['python3',config[args.run]['file']],cwd=path)
else:
    print('No demo name given, choose between the following:')
    for key in config:
        print(key)
        print('\t{}'.format(config[key]['text']))