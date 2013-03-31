#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import threading,copy
from time import sleep
from time import time as _time
from random import random

N = 50

class Out(threading.Thread):
    def __init__(self,info,info_lock):
        threading.Thread.__init__(self)
        self.daemon = True
        self.info_d = copy.copy(info)
        self.info_lock = info_lock
        self.last = -1
        self.is_stop = False
        print("Init thread 1!")
    def run(self):
        while True:

            self.info_lock.acquire(1)
            self.info_d = copy.copy(info)
            self.info_lock.release()

            if self.last != self.info_d[1]:
                print("%f, %d" %(self.info_d[0],self.info_d[1]))
                sleep(1)
                self.last = self.info_d[1]
            if self.info_d[1] > N:
                print("Out stop")
                self.is_stop = True

            if self.is_stop:
                break


class In(threading.Thread):
    def __init__(self,info,info_lock):
        threading.Thread.__init__(self)
        self.daemon = True
        self.info = info
        self.info_d = info_d
        self.info_lock = info_lock
        self.is_stop = False
        print("Init thread 2!")
    def run(self):
        while True:

            if not self.info_lock.locked():
                self.info[0] = random()
                self.info[1] += 1 
                sleep(0.5)
                if self.info[1] > N:
                    self.is_stop = True
                    print("In stop!")
            # else:
            #     print("In: locked")

            if self.is_stop:
                break



info = [0,0]
info_d = copy.copy(info)
info_lock = threading.Lock()

#call

thread_in = In(info,info_lock)
thread_out = Out(info,info_lock)
print("Started!")
thread_in.start()
thread_out.start()
print("Waiting!")

thread_in.join()
thread_out.join()

print("End")

# tm_old = _time()

# while True:
#     info[0] = random()
#     info[1] += 1 
#     sleep(1/N)

#     if _time() - tm_old > 0.1:
#         print("%f, %d" %(info[0],info[1]))
#         sleep(0.1)
#         tm_old = _time()

#     if info[1] > N:
#         break