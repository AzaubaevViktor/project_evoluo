#!/usr/bin/env python3
# -*- coding:utf -*-
from random import random
import math

def get_random_sequence2d(X,Y): # checked
    return [[random() for x in range(X)] for y in range(Y)]

def cosine_interpolation(a,b,alpha): #
    f = (1-cos(aplha*math.pi))
    return a*(1-f)+b*f

def smooth_noise_2d(ns,x,y): #checked
    xm = len(ns[0])
    ym = len(ns)
    k = 8 * (1 + 2 ** 0.5)
    corners = (ns[y-1][x-1]+ns[y-1][(x+1) % xm]
        +
        ns[(y+1) % ym][x-1]+
        ns[(y+1) % ym][(x+1) % xm]) / (k * 2 ** .5)
    sides = (ns[y][x-1]+ns[y][(x+1) % xm]+ns[y-1][x]+ns[(y+1) % ym][x]) / k
    center = ns[y][x] / 2
    return corners + sides + center

def perlin(X,Y): #checked
    ns = get_random_sequence2d(X,Y)
    return [[smooth_noise_2d(ns,x,y) for x in range(X)] for y in range(Y)]

def ones(X,Y):
    return [[1 for x in range(X)] for y in range(Y)]