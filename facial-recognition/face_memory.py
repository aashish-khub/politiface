#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 18:30:01 2020

@author: aashishk29

This file contains information pertaining to the encoding and storage
of faces of the politicians in question
"""


import numpy as np
import matplotlib.pyplot as plt
import face_recognition as frec
import os

SERIALIZEDDIRECTORY = "facial_encodings/"


class Person():
    name = None
    party = None
    encoding = None
    trainingSet = None
    
    def __init__(self,personName, party=None):
        self.name = personName
        self.party = party
        self.encoding = None
        self.trainingSet = []
        
    def addImage(self,fileName):
        img = frec.load_image_file(fileName)
        self.trainingSet.append(img)
    
    def encode(self):
        self.encoding = frec.face_encodings(self.trainingSet[0])[0]
        #Note: the self.trainingSet[0] is because of the current...
        #       ...approach using just a single training image
        #the second [0] is list->np.arr conv.
    
    def serialize(self):
        np.savetxt(SERIALIZEDDIRECTORY+self.name+".csv", self.encoding,delimiter=",")
    
    def deserialize(self):
        loaded = np.loadtxt(SERIALIZEDDIRECTORY+self.name+".csv",dtype=float,delimiter=",")
        self.encoding = loaded












