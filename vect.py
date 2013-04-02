#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import math,pdb
from math import pi,sqrt,acos,cos,sin

_pi_2 = pi * 2

class Vector:
    """ Велосипед для работы с плоскими векторами """
    def __init__(self,a,b,isPolar = False):
        """ Создаёт вектор; 4 - cart, 5 - polar """ 
        global _pi_2
        if isPolar:
            self.a = [0, 0, a, b % _pi_2, 0, 1]
        else:
            self.a = [a, b, 0, 0, 1, 0]
           
    def _cartesian_to_polar(self):
        """ Переводит координаты из декартовых в полярные """
        self.a[5] = 1
        x = self.a[0]
        y = self.a[1]
        self.a[2] = r = sqrt(x*x+y*y)
        if abs(x) > abs(r):
            self.a[3] = 0.0
        elif r != 0.:
            phi = acos(x/r)
            if y < 0.:
                self.a[3] = - phi
            else:
                self.a[3] = phi
        else:
            self.a[3] = 0.0

    def _polar_to_cartesian(self):
        """ Переводит координаты из полярных в декартовы """
        self.a[4] = 1
        r = self.a[2]
        phi = self.a[3]
        self.a[0] = r*cos(phi)
        self.a[1] = r*sin(phi)

    def __getattr__(self,name):
        if (name == 'x') or (name == 'y'):
            if not self.a[4]:
                self._polar_to_cartesian()
            if name == 'x':
                return self.a[0]
            else:
                return self.a[1]

        elif (name == 'r') or (name == 'phi'):
            if not self.a[5]:
                self._cartesian_to_polar()
            if name == 'r':
                return self.a[2]
            else:
                return self.a[3]

        elif name == 'cart':
            if not self.a[4]:
                self._polar_to_cartesian()                
            return self.a[0:2]

        elif name == 'pol':
            if not self.a[5]:
                self._cartesian_to_polar()
            return self.a[2:4]

        else:
            raise AttributeError(name)

    def __setattr__(self,name,value):
        global _pi_2
        if (name == 'x') or (name == 'y'):
            if name == 'x':
                self.a[0] = value
            else:
                self.a[1] = value
            self.a[4] = 1
            self.a[5] = 0

        elif (name == 'r') or (name == 'phi'):
            if name == 'r':
                self.a[2] = value
            else:
                self.a[3] = value % _pi_2
            self.a[4] = 0
            self.a[5] = 1

        if name == 'cart':
            self.a[0], self.a[1] = value
            self.a[4] = 1
            self.a[5] = 0

        elif name == 'pol':
            self.a[2], self.a[3] = value
            self.a[4] = 0
            self.a[5] = 1

        else:
            self.__dict__[name] = value

    def __add__(self,v):
        if not self.a[4]:
            self._polar_to_cartesian()
        if not v.a[4]:
            v._polar_to_cartesian()
        return Vector(self.a[0] + v.a[0], self.a[1] + v.a[1], isPolar = False)        

    def __iadd__(self,v):
        pdb.set_trace()
        if not self.a[4]:
            self._polar_to_cartesian()
        if not v.a[4]:
            v._polar_to_cartesian()
        self.a[0] += v.a[0]
        self.a[1] += v.a[1]
        self.a[5] = 0
        return self

    def __sub__(self,v):
        if not self.a[4]:
            self._polar_to_cartesian()
        if not v.a[4]:
            v._polar_to_cartesian()
        return Vector(self.a[0] - v.a[0], self.a[1] - v.a[1], isPolar = False)   

    def __isub__(self,v):
        if not self.a[4]:
            self._polar_to_cartesian()
        if not v.a[4]:
            v._polar_to_cartesian()
        self.a[0] -= v.a[0]
        self.a[1] -= v.a[1]
        self.a[5] = 0
        return self

    def __mul__(self,k):
        if self.a[5]:
            return Vector(self.a[2] * k, self.a[3], isPolar = True)
        else:
            return Vector(self.a[0] * k, self.a[1] * k, isPolar = False)

    def __imul__(self,k):
        if self.a[5]:
            self.a[4] = 0
            self.a[2] *= k
        else:
            self.a[5] = 0
            self.a[0] *= k
            self.a[1] *= k
        return self

    def __truediv__(self,k):
        if self.a[5]:
            return Vector(self.a[2] / k, self.a[3], isPolar = True)
        else:
            return Vector(self.a[0] / k, self.a[1] / k, isPolar = False)

    def __itruediv__(self,k):
        if self.a[5]:
            self.a[4] = 0
            self.a[2] /= k
        else:
            self.a[5] = 0
            self.a[0] /= k
            self.a[1] /= k
        return self

    def __neg__(self):
        if self.a[5]:
            return Vector(- self.a[2], self.a[3], isPolar = True)
        else:
            return Vector(- self.a[0], - self.a[1], isPolar = False)

    def __str__(self):
        if not self.a[5]:
            self._cartesian_to_polar()
        if not self.a[4]:
            self._polar_to_cartesian()
        return ({"x":self.a[0],"y":self.a[1],"r":self.a[2],"phi":self.a[3]}.__str__())

    def one(self):
        """ Возвращает единичный вектор """
        return Vector(1,self.phi,isPolar = True)

    def copy(self):
        if self.a[4]:
            return Vector(self.a[0],self.a[1],isPolar = False)
        else:
            return Vector(self.a[2],self.a[3],isPolar = True)

null = lambda: Vector(0,0,isPolar = False)