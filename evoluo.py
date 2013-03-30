#!/usr/bin/env python3
#-*- coding:utf-8 -*-
""" Программа-симулятор эволюции. Тестовая версия.
Основные положения:
В самой программе и в визуализации вывод будет, скорее всего, начиная с нижней левой точки как (0,0) 
Итак, что вообще и как происходит.
В главном цикле вызывается функция step(), которая перебирает все слои и отдаёт им массив со всеми слоями.
"""
import random,math,pdb,noise,argparse,copy,sys
from math import sqrt,trunc,log,pi,sin,cos,tan,acos,asin,atan
import time as time_
import vect
from vect import Vector
import OpenGL.GL as GL
import OpenGL.GLUT as GLUT
import OpenGL.GLU as GLU

# для simm_r в зрении
field_angle = [ [-3*pi/4, -pi/2, -pi/4],
                [pi    , 0    , 0    ],
                [3*pi/4 , pi/2 , pi/4 ]]

def getpair(a,b): #костыль для быстродействия
    if a>b:
        return [b,a]
    else:
        return [a,b]

def get_mutation(v):
    return v * (1 + k_mutation * (random.random() - 0.5) )

def abs_angl(phi):
    phi %= 2*pi
    if phi > pi:
        return phi - 2*pi
    else:
        return phi

def gamecoord_to_screen(x):
    return x * k_screen

def write_inf(str,*attr):
    Informations.append((str,attr))

def get_min_distance(param,p1,p2): #проверена
    """ Возвращает минимальное расстояние между двумя точками на поле """
    _r = lambda x,y: sqrt(x*x + y*y)
    return _r(p1.x - p2.x + trunc((p2.x - p1.x) / (0.5*param[0])) * param[0]
        ,p1.y - p2.y + trunc((p2.y - p1.y) / (0.5*param[1])) * param[1])

def get_distribution(param,f0):
    """ Возвращает функцию с распределением таким, что при deep = 0 и dx = w/2 dy = h/2 даёт 0.1, а при r = 0 даёт 1, формула 1/(x^alpha+1).
    Является генератором функций.
    Пока особо не нужна, но понадобится при создании солнечного света"""
    r = (param[0] ** 2 + param[1] ** 2) ** 0.5
    alpha = log(1/f0-1)/log(r)
    def _dist(x):
        return 1/(x ** alpha + 1)
    return _dist

def write_arr(arr):
    for a in arr:
        print(end='[')
        for el in a:
            print("%.2f" %el,end=', ')
        print(']')

def _init_get_under_new():
    maxR = round(min(real_width/2,real_height/2))
    _get_under = [[] for r in range(maxR)]
    v_0 = Vector(0,0)
    for R in range(1,maxR+1):
        _quart = []
        for x in range(1,R+1):
            for y in range(1,R+1):
                if (x + y >= R):
                    if get_min_distance((real_width,real_height),Vector(x,y),v_0) <= R:
                        _quart.append([x,y])
                else:
                    _quart.append([x,y])

        _get_under[R-1] = (_quart
                        + [[-x,y] for x,y in _quart]
                        + [[x,-y] for x,y in _quart]
                        + [[-x,-y] for x,y in _quart]
                        + [[x,0] for x in range(-R,R+1)]
                        + [[0,y] for y in range(-R,R+1) if y != 0])
    return _get_under

class Screen:
    """ Суперкласс экрана. Служит основой для других классов """
    def __init__(self,type_screen,loop_func,layers):
        """ Инициализирует экран """
        self.type = type_screen
        self.width = 0
        self.height = 0
        self.loop_func = loop_func
        self.layers = layers
        pass
    def draw(self,layer):
        """ Расует слой """
        pass
    def update(self):
        """ Обновляет экран, если используется двойна буферизация """
        pass
    def change_header(self,str):
        """ Меняет заголовок экрана """
        pass
    def clear(self):
        """ Очищает экран """
        pass
    def write(self,pos,str):
        pass
    def write_inf(self,str):
        pass
    def stop(self):
        """ Останавливает экран """
        pass
    def getch(self):
        return 0
    def _loop(self):
        global tick
        if (self.ch == b'q') or (self.ch == b'\xb1') or (self.ch == 27) or (self.ch == 113):
            self.__del__()
            del(self)
            return 0
        else:
            tick += 1
            return self._main()
    def _main(self):
        self.clear()
        self.loop_func(self.layers)
        for layer in layers:
            self.draw(layer)

        self.update()

        self.ch = self.getch()
        
        return 1
    
    def _end(self):
        global tick
        print("Ticks: %d" %tick)

    def __del__(self):
        """ Деструктор класса """
        pass

import curses

class ScreenCursesInfoBackUp(Screen):
    def __init__(self):
        Screen.__init__(self,"Curses Screen")
        self._init_curses()

    def _init_curses(self):
        print("Init sprites...")
        self._get_under = _init_get_under_new()

        self._scr = curses.initscr()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_WHITE)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_WHITE)
        # curses.noecho()
        # curses.cbreak()
        # self._scr.keypad(1)
        self._scr.clear()
        # self._scr.nodelay(1)
        self.height,self.width = self._scr.getmaxyx()
        self.height -= 1
        self.wrt_pos = 0
        self.ch = None

    def line(self,pos,vect,color):
        def _from(a,b):
            if a > b:
                return reversed(range(b,a+1))
            else:
                return range(a,b+1)
        x = [0,0]
        y = [0,0]

        x[0], y[0] = pos[0], pos[1]
        x[1], y[1] = vect.x + x[0], vect.y + y[0]

        if abs(y[1] - y[0]) < abs(x[1] - x[0]):
            for _x in _from(0,int(x[1]-x[0])):
                if (x[1] - x[0]) != 0:
                    self.write(
                        ( int(x[0] + _x), int( y[0] + (y[1]-y[0]) / (x[1]-x[0]) * _x )), '*', curses.color_pair(color)
                        )
        else:
            for _y in _from(0,int(y[1]-y[0])):
                if (y[1] - y[0]) != 0:
                    self.write(
                        ( int( x[0] + (x[1]-x[0]) / (y[1]-y[0]) * _y ), int(y[0] + _y) ), '*', curses.color_pair(color)
                        )
        pass

    def get_under(self,pos,R,func,*args): # Удалить
        """ Даёт список ссылок на клетки слоя, которые находятся под окружностью радиуса R."""
        _R = int(round(R))
        dx = int(pos[0])
        dy = int(pos[1])
        for x,y in self._get_under[_R-1]:
            x += dx
            x %= self.width
            y += dy
            y %= self.height
            func(x,y,*args)

    def draw(self,layer):
        if layer.__class__ == LayerObjects:
            for obj in layer.get_objs():
                pos = obj.get_pos_screen()
                def _wrt(x,y):
                    self.write((x,y)," ",curses.A_REVERSE)
                self.get_under(pos,obj.radius * k_screen,_wrt) # Исправить на Layer. ...
                self.line(
                    pos,
                    gamecoord_to_screen(
                        Vector(obj.radius+obj._attack_range*obj._attack*obj.radius,
                        obj.pos[1],
                        isPolar = True
                        )),
                    2)
                self.line(pos,obj.speed[0] * k_screen * 3,1)
                self.write((int(pos[0]),int(pos[1])),"%d" %(obj._get_lifestate()*10),curses.A_REVERSE)

    def update(self):
        self._scr.refresh()

    def clear(self):
        for y in range(self.height):
                self.write((0,y),' ' * self.width)
        self.wrt_pos = 0

    def write(self,pos,str,*attr):
        self._scr.addstr(int(pos[1] % self.height),int(pos[0] % self.width),str,*attr)

    def write_inf(self):
        pass

    def write_ch(self,pos,ch,*attr):
        self._scr.addch(pos[1],pos[0],ch,*attr)

    def getch(self):
        return self._scr.getch()

    def __del__(self):
        # curses.nocbreak()
        # curses.echo()
        curses.endwin()
        # self._scr.keypad(0)
        # self._scr.nodelayd(1)

class ScreenCursesInfo(Screen):
    def __init__(self):
        Screen.__init__(self,"Curses Info Screen",None,None)
        self._init_curses()

    def _init_curses(self):
        # print("Init sprites...")
        # self._get_under = _init_get_under_new()
        self._scr = curses.initscr()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_WHITE)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_WHITE)
        curses.noecho()
        curses.cbreak()
        self._scr.keypad(1)
        self._scr.clear()
        self._scr.nodelay(1)
        self.height,self.width = self._scr.getmaxyx()
        self.height -= 1
        self.wrt_pos = 0
        self.ch = None

    def draw(self):
        self.clear()
        self.write_inf()
        self.update()

    def update(self):
        self._scr.refresh()

    def clear(self):
        for y in range(self.height):
                self.write((0,y),' ' * self.width)
        self.wrt_pos = 0

    def write(self,pos,str,*attr):
        self._scr.addstr(int(pos[1] % self.height),int(pos[0] % self.width),str,*attr)

    def write_inf(self):
        global Informations
        for (string,attr) in Informations:
            self._scr.addstr(self.wrt_pos % self.height,1,str(string))
            self.wrt_pos += 1
        Informations = []

    def write_ch(self,pos,ch,*attr):
        self._scr.addch(pos[1],pos[0],ch,*attr)

    def getch(self):
        return self._scr.getch()

    def __del__(self):
        curses.nocbreak()
        curses.echo()
        curses.endwin()
        self._scr.keypad(0)
        self._scr.nodelay(1)

class ScreenPrintInfo(Screen):
    def __init__(self):
        Screen.__init__(self,"Print Info Screen",None,None)
    def write_inf(self):
        global Informations
        for (string,attr) in Informations:
            print(string,*attr)
        Informations = []
    def draw(self):
        self.write_inf()

class ScreenStandartInfo(Screen):
    def __init__(self):
        Screen.__init__(self,"Standart Info Screen",None,None)
    def write_inf(self):
        global Informations
        Informations = []
    def draw(self):
        self.write_inf()

class ScreenStandart(Screen):
    def __init__(self,loop_func,layers):
        Screen.__init__(self,"Standart Screen",loop_func,layers)
        global tick
        for tick in range(1,1000):
            self._main()

class ScreenOpenGL(Screen):
    def __init__(self,loop_func,layers):
        """ Инициализирует экран и запускает его, потом всю программу """
        Screen.__init__(self,"OpenGL",loop_func,layers)

        self.window = 0
        self.quad = None
        self.ch = []

        print("Fuck")
        # self.infoScreen = ScreenCursesInfo()
        self.infoScreen = ScreenStandartInfo()
        GLUT.glutInit(sys.argv)
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_ALPHA | GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(640, 480)
        GLUT.glutInitWindowPosition(400, 400)
        self.window = GLUT.glutCreateWindow(b"Project Evoluo alpha")
        GLUT.glutDisplayFunc(self._loop) # Функция, отвечающая за рисование
        GLUT.glutIdleFunc(self._loop) # При простое перерисовывать
        GLUT.glutReshapeFunc(self._resize) # изменяет размеры окна
        GLUT.glutKeyboardFunc(self._keyPressed) # Обрабатывает нажатия
        self._initGL(640, 480)
        field_params(640, 480)
        GLUT.glutMainLoop()
        print("Fuck")

    def _initGL(self,Width,Height):
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)    # This Will Clear The Background Color To Black
        GL.glClearDepth(1.0)                   # Enables Clearing Of The Depth Buffer
        GL.glDepthFunc(GL.GL_LESS)                # The Type Of Depth Test To Do
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glEnable(GL.GL_BLEND);                         # Enable Blending
        GL.glLineWidth(1.);
        GL.glDisable(GL.GL_LINE_SMOOTH)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);

        self.width = Width
        self.height = Height
        self.quad = GLU.gluNewQuadric()

    def _resize(self,Width,Height):
        if Height == 0:
            Height = 1
        self.width = Width
        self.height = Height
        field_params(Width,Height)

        GL.glViewport(0, 0, Width, Height)       # Reset The Current Viewport And Perspective Transformation
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()

        GL.glOrtho(0.0,Width,0.0,Height,-1.0,1.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)                   # Select The Modelview Matrix
        GL.glLoadIdentity()

        write_inf(width,height)
        write_inf(Width,Height)
        write_inf(k_screen)

    def update(self):
        GLUT.glutSwapBuffers()

    def clear(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()                    # Reset The View 

    def write(self,pos,str):
        pass

    def __del__(self):
        del(self.infoScreen)
        GLUT.glutDestroyWindow(self.window)
        self._end()
        # sys.exit()

    def getch(self):
        if ch != []:
            return ch.pop(-1)[0]    

    def _keyPressed(self,*args):
        if args != None:
            ch.append(args)
            print(args)

    def draw(self,layer):
        global width,height
        if layer.__class__ == LayerObjects:
            attacked = layer._attacked
            for obj in layer.get_objs():
                pos = obj.get_pos()
                write_inf("%7d:[%4.1f;%4.1f].L:%.2f/%.2f" %(obj._id,pos[0],pos[1],obj._energy,obj._max_energy))
                pos = obj.get_pos_screen()
                # Кружок
                GL.glLoadIdentity()
                GL.glTranslatef(pos[0]-1,pos[1]-1,0)
                if attacked.get(obj._id,0) > 0:
                    red = 1
                else:
                    red = 0
                GL.glColor3f(red,obj._get_lifestate()*0.9+0.1,1-obj.age/100)
                GLU.gluDisk(self.quad,0,obj.radius*k_screen,30,1)
                #Стрелочки-направления
                att = Vector(obj.radius+obj._attack_range*obj._attack*obj.radius,
                        obj.pos[1],
                        isPolar = True
                        ) * k_screen
                speed = obj.speed[0] * k_screen
                GL.glBegin(GL.GL_LINES)
                GL.glColor3f(1,0,0)
                GL.glVertex3f(0,0,-1)
                GL.glVertex3f(att.x,att.y,-1)
                GL.glColor3f(0,0,1)
                GL.glVertex3f(0,0,-0.5)
                GL.glVertex3f(speed.x,speed.y,-1.1)
                GL.glEnd()
                
            self.infoScreen.draw()

class Layer:
    """ Класс слоя. Клеточный автомат, который воздейсвтует на объекты """
    def __init__(self,w = -1,h = -1, min = 0,max = 1,type = 'none'):
        """ Инициализирует слой;"""
        global width, height
        if w == -1:
            w = width
        if h == -1:
            h = height
        self.width = w
        self.height = h
        if type == 'none':
            self.layer = None
        elif type == 'petri': #checked
            self.layer = noise.perlin(w,h)
            self.layer = [[self.layer[y][x]*(max-min)+min for x in range(w)] for y in range(h)]
        elif type == 'ones': #checked
            self.layer = noise.ones(w,h)
        elif type == 'lines':
            self.layer = [[int(x>y)*(max-min)+min for x in range(w)] for y in range(h)]

    def get_under(self,pos,R,func,*args): #checked 
        """ Даёт список ссылок на клетки слоя, которые находятся под окружностью радиуса R."""
        _R = int(round(R))
        dx = int(pos[0])
        dy = int(pos[1])
        for x,y in self._get_under[_R-1]:
            x += dx
            x %= self.width
            y += dy
            y %= self.height
            func(x,y,*args)

    def get_under_summ(self,pos,R):
        s = [0]
        def _summ(x,y,s,self):
            s[0] += self.layer[y][x]
        self.get_under(pos,R,_summ,s,self)
        return s[0]

    def _impact_layer(self,layer):
        """ Воздействует на другой слой, в том числе и на себя """
        pass

    def step(self,layers):
        """ Шагает """
        for layer in layers:
            self._impact_layer(layer)

class LayerObjects(Layer):
    """ Необычный класс, который вмещает в себя всех живых существ и делает вид, что он обычный класс \n
    Содержит в себе, в отличие от обычного класса слоя, не слой WxH, а объекты, которые взаимодействуют со средой"""
    def __init__(self,w = -1,h = -1):
        global width, height
        if w == -1:
            w = width
        if h == -1:
            h = height
        self.width = w
        self.height = h
        self._objs = {}
        self._colliding = [[],[]] #сталкивающиеся объекты на данный момент (которые были в прошлый ход, которые сейчас)
        self._attacked = {}
    def _generate_id(self):
        """ Генерирует уникальный id. В будущем переделать """
        return random.randint(0,10000000)

    def create_obj(self,obj):
        """ Создаёт объект """
        obj._id = self._generate_id()
        self._objs.update({obj._id:obj})

    def get_objs(self):
        """ Возвращает список объектов """
        return list(self._objs.values())

    def get_obj_by_id(self,id):
        """ Возвращает объект по id """
        return self._objs.get(id,None)

    def delete_obj_by_id(self,id):
        """ Удаляет объект по id """
        try:
            self._objs.pop(id)
        except KeyError:
            return 1
        return 0

    def _impact_self(self,layer):
        """ Для каждого объекта внутри себя заставляет его обработать слой """
        for obj in self.get_objs():
            if obj.status == 2:
                self.create_obj(obj.create_child())
                obj._energy *= 0.6 # должно отняться ровно 40% за раз FACTOR
                obj._change_state()
            if obj.status == -1:
                self.delete_obj_by_id(obj._id) # удаляем из списка ВОЗМОЖНОЕ МЕСТО ДЛЯ УТЕЧКИ ПАМЯТИ
            obj.step(layer)

    def _eat(self,obj1,obj2):
        """ Поедание """
        if (obj2.status <= 0) and (obj1.status > 0):
            write_inf("Eat: %d --> %d" %(obj2._id,obj1._id))
            obj2.mass -= 1 * dt # За один тик убавляется 1 жизнь
            if obj2._change_state() != -1:
                obj2.radius = sqrt(obj2.mass)
            obj1.energy(obj2._max_energy * obj2._strong) #FACTOR добвляем энергии поедающему

    def _collision(self,id1,id2,inside):
        """ Просчитывает скорости объектов
        РАсчёт коэффициента: 1.1"""
        obj1 = self.get_obj_by_id(id1)
        obj2 = self.get_obj_by_id(id2)
        if (obj1 != None) and (obj2 != None):
            p1 = obj1.pos[0]
            v1 = obj1.speed[0]
            w1 = obj1.speed[1]
            m1 = obj1.mass
            p2 = obj2.pos[0]
            v2 = obj2.speed[0]
            w2 = obj2.speed[1]
            m2 = obj2.mass
            #получаем вектор-прямую, на которую будут откладываться взаимодействующие вектора
            phi = Vector(p2.x - p1.x, p2.y - p1.y,isPolar = False).phi #угол линии с OX, соединяющий центры

            # Меняем скорости, если они столкнулись только что
            if (obj1.status > 0) and (obj2.status > 0):
                if self._colliding[0].count(getpair(id1,id2)) == 0:
                    def _get(m1,m2,v1,v2):
                        """ уравнение полученных скоростей, чтобы по 10 раз не писать одно и то же """ 
                        return (v1 * (m1 - m2) + v2 * 2 * m2) / (m1 + m2)
                    v1.phi -= phi # поворот так, что линия есть OX 
                    v2.phi -= phi
                    _v1 = v1.copy()
                    _v2 = v2.copy()
                    V1before = _v1.copy()
                    V2before = _v2.copy()
                    _v1.x = _get(m1,m2,v1.x,v2.x) # взаимодействуют
                    _v2.x = _get(m2,m1,v2.x,v1.x)

                    dv1 = abs((_v1 - V1before).x)
                    dv2 = abs((_v2 - V2before).x)
                    _v1.phi += phi #поворачиваем обратно
                    _v2.phi += phi
                    # Применяем скорости
                    obj1.speed[0] = _v1
                    obj2.speed[0] = _v2

                    mass12 = obj2.mass / obj1.mass
                    obj1.speed[1] -= (w2 * mass12 + w1) / 10 # сохраняем импульс и передаём 1/10 от угловой скорости FACTOR
                    obj2.speed[1] -= (w1 / mass12 + w2) / 10

                    if dv1>obj1.get_strong():
                        obj1.energy(-(dv1-obj1.get_strong()))
                    if dv2/m2>obj2.get_strong():
                        obj2.energy(-(dv2-obj2.get_strong()))
                else:
                    F = Vector(inside,phi,isPolar = True)
                    obj1._add_accel([-F,0]) #добавляем ускорение
                    obj2._add_accel([F,0])
            elif (obj1.status > 0) or (obj2.status > 0):
                str = obj1.get_strong() * (obj1.status > 0)  + obj2.get_strong() * (obj2.status > 0)
                obj1._add_accel((Vector(str,phi,isPolar = True),0))
                obj2._add_accel((Vector(- str,phi,isPolar = True),0))

            # Съедает
            for a,b in [(obj1,obj2),(obj2,obj1)]:
                self._eat(a,b)

    def collision(self):
        """ Определяет, сталкиваются ли объекты """
        _objs = self.get_objs()
        lenght = len(_objs)
        for i in range(lenght):
            for j in range(i+1,lenght):
                inside = (get_min_distance((self.width,self.height),_objs[i].pos[0],_objs[j].pos[0]) - (_objs[i].radius+_objs[j].radius)) / 2
                if inside < 0:
                    self._collision(_objs[i]._id,_objs[j]._id,-inside)
                    self._colliding[1].append(getpair(_objs[i]._id,_objs[j]._id)) # добавляет сталкивающиеся объекты в новый, чтобы на следующем шаге показать, что они уже сталкивались и надо их посильнее оттолкнуть
        self._colliding[0] = copy.copy(self._colliding[1])
        self._colliding[1] = []

    def _attack(self,a,b):
        #Расчёт того, насколько глубоко проник удар I.3
        OO = Vector(b.pos[0].x-a.pos[0].x, b.pos[0].y-a.pos[0].y, isPolar = False) # Вектор, соединяющий центры ботов
        at = Vector(a.radius + a._attack * a._attack_range * a.get_strong() * a.radius, a.pos[1], isPolar = True) #вектор атаки
        if abs(abs_angl(at.phi - OO.phi)) < pi / 2:
            ah = Vector(cos(at.phi - OO.phi)*OO.r, at.phi, isPolar = True) # вектор до высоты, опущенной из центра b
            f_at = at.r-ah.r # первая часть проникновения
            OH = (ah - OO).r # длинна высота из центра b
            if OH < b.radius:
                BH = sqrt(b.radius*b.radius - OH*OH)
                f_at += BH
                if f_at > 0:
                    write_inf("Attack: %d --> %d, by Str %.3f" %(a._id,b._id,f_at))
                    if b.status != 0:
                        f_at *= a._attack * a.get_strong() / a.radius #вычисляем силу атаки
                        if BH > 0.001: # вычисляем угловое ускорение
                            omega = f_at * sin(atan(OH/BH)) # учитываем энергию a и силу удара
                        else:
                            omega = 0
                        b.energy (- f_at)
                        b._add_accel((OO.one() * f_at, omega)) #отталкиваем и крутим противника
                        return f_at
        return 0

    def attack(self):
        _objs = self.get_objs()
        # lenght = len(_objs)
        self._attacked = {}
        for a in _objs:
            for b in _objs:
                if (a != b):
                    inside = (get_min_distance((self.width,self.height),a.pos[0],b.pos[0]) - (a.radius+b.radius)) / 2
                    if inside < a._attack_range:
                        attack = self._attack(a,b) # Атакует a
                        self._attacked.update({b._id:attack})


    def step(self,layers):
        for layer in layers:
            self._impact_self(layer)
        self.collision()
        self.attack()

class LayerViscosity(Layer):
    """ Замедляет объекты. Работает по принципу больше скорость -- больше сила трения
    #FIXME: не замедляет (общий импульс системы и скорость не уменьшаются, если dt > 0.3 """
    def __init__(self,**args):
        Layer.__init__(self,max = 0.1, type = 'none', **args)
        self.mu = args.get('mu',0.1) #коэффициент трения FACTOR

    def _impact_obj(self,obj):
        """ Как слой с замедлениями влияет на объект, зависит от времени """
        ds = obj.mass * self.mu
        obj._add_accel([- obj.speed[0] * ds, - obj.speed[1] * ds])

    def _impact_layer(self,layer):
        if layer.__class__ == LayerObjects:
            for obj in layer.get_objs():
                self._impact_obj(obj)

class Mind:
    """ (Тестовый) Суперкласс разума. На вход подаются данные с датчиков, на выход -- действия """
    def __init__(self,**init):
        pass

    def create_child(self):
        return copy.deepcopy(self)

    def step(self,args):
        move = (sin(args.get("energy",0) / args.get("maxenergy",1)),random.random()-0.5)
        # move = (1,0)
        attack = random.random() # -- сила удара
        # attack = 1
        ferromons = (1,0,0,0) # 2^4 = 16 ферромонов, в данный момент он отдаёт ферромон 1000
        return {"move":move, "attack":attack, "ferromons": ferromons}

class Mind_const(Mind):
    def __init__(self,**init):
        self.mvnot = init.get("mvnot",(0.05,0.001))
        self.angvel = init.get("angvel",0.05)

    def create_child(self):
        return Mind_const(
            mvnot = ( get_mutation(self.mvnot[0]), get_mutation(self.mvnot[1]) ),
            angvel = get_mutation(self.angvel)
            )

    def step(self,args):
        vis = args.get("vision",[0,0,0,0,0,0,0])
        n = 0
        for x in range(7):
            if vis[n] < vis[x]:
                n = x
        if vis[n] < 0.1: # если ничего нет поблизсти
            move = self.mvnot # крутимся, если ничего не видим
        else:
            move = (vis[n],(n-3) * self.angvel ) # подбираемся к жертве
        attack = 0
        if vis[n] > 0.98    :
            attack = 1
        write_inf(str(self.mvnot)+str(self.angvel))
        return {"move":move, "attack":attack}


class Sensor:
    """ Суперкласс сенсора. На вход -- слои, которые он обрабатывает и выдаёт информацию """
    def __init__(self,obj):
        self.output = None # здесь должен быть ответ
        self.obj = obj

    def __call__(self,layers):
        pass

class Eyes(Sensor):
    def __init__(self,*args1,**args2):
        Sensor.__init__(self,*args1,**args2)
        self.focus = 50 #оптимальное зрение

    def __call__(self,layers):
        self.conversion = self._get_conversion() 
        for layer in layers:
            if layer.__class__ == LayerObjects:
                return self._get_angles((layer.width,layer.height),layer.get_objs())

    def _get_conversion(self):
        f = self.focus
        def getr(r):
            # return r / (100*f*(f-100)) * ( (f-50) * r - f * f + 5000 ) + 1 да вы ебанитесь
            if r > f:
                return 0.5 * (r-100)/(f-100.)/2.
            else:
                return 1. - r/f/2.
        return getr

    def _change_focus(self,r):
        self.focus = self.focus * (1 - dt / 20) + r * dt / 20

    def _get_angles(self,param,objects):
        sobj = self.obj
        w,h = param
        X,Y = sobj.pos[0].x+w, sobj.pos[0].y+h
        radius = sobj.radius
        sphi = sobj.pos[1] # направление "взгляда"
        allowed = []
        for y in range(3): #составляем список разрешённых
            for x in range(3):
                if (x == 1 == y) or (abs(abs_angl(field_angle[y][x]-sphi)) <= 0.55 * pi): # ~ 105/2 + 45
                    allowed.append([x,y])

        # pdb.set_trace()
        out = [0,0,0,0,0,0,0] # 0 -- за пределами видимости r = 100, 0.5 - в фокусе r = f, 1 -- близко r = 0
        r = [100,100,100,100,100,100,100] # расстояния
        for (dx,dy) in allowed:
            for obj in objects:
                if obj != sobj:
                    v = Vector( obj.pos[0].x+w*dx - X, obj.pos[0].y+h*dy - Y)
                    n = abs_angl(v.phi-sphi) / (0.083 * pi) # 15 градусов
                    if -3 <= n <= 3:
                        n += 3.5 # 0.5 для округления
                        n = int(n)
                        r[n] = min(r[n],v.r-obj.radius-radius)
                        # write_inf(str(dx)+" "+str(dy)+" %.3f" %v.r)

        self._change_focus(r[3])

        # write_inf(str(r)+" %.3f" %self.focus)

        return [self.conversion(a + (a-self.focus) * random.normalvariate(0,0.2) ) for a in r]


        # s = ["%3d" %(x + (x - self.focus) * random.normalvariate(0,0.2)) for x in r]

        # for y in reversed(range(3)):
        #     s = str(y)+':'
        #     for x in range(3):
        #         if [x,y] in allowed:
        #             s += 'X'
        #         else:
        #             s += 'O'
        #     write_inf(s)        
        # write_inf(str(allowed)+str(sphi/pi*180))

class Object: 
    """ Суперкласс объекта. Служит основой для других классов объектов. Статус: 0 -- мёртв (<0% энергии), 1 -- в спячке (<10% энергии), 2 -- жив"""
    def __init__(self
            ,pos:'местоположение' = (0,0,0) # x,y; угол
            ,mind:'Мозги' = Mind()
            ,energy:'Энергия' = 0.1
            ,maxenergy:'Максимальная энергия' = 1
            ,speed:'Скорость' = (0,0,0) # мощь, угол; угловая скорость
            ,accelerate:'Ускорение' = (0,0) # мощь, угловое ускорение
            ,radius:'радиус объекта' = 1
            ,strong:'Защита и сила; от неё зависят многие параметры' = 1
            ,attack_range: 'Дальность атаки' = 1
            ,id : 'Идентификатор объекта' = 0
            ,**args):
        self.pos = [Vector(*pos[0:2]), pos[2]] #положение -- координата + угол
        self._energy = energy # Текущая энергия
        self._max_energy = maxenergy # максимальная энергия
        self.speed = [Vector(*speed[0:2]), speed[2]] # начальные скорости
        self._accel = [Vector(accelerate[0],speed[2]),accelerate[1]] # ускорение
        self.radius = radius # радиус
        self.mass = radius ** 2 # масса
        self.status = 1 # статус
        self._strong = strong #Защита и сила
        self._mind = mind # Мозги
        self._attack = 0 #сила атаки
        self._attack_range = attack_range #дальность атаки
        self._id = id
        self.age = 0

    def _move(self): 
        """ Передвигает объект на вектор """
        self.pos[1] += self.speed[1] * dt # угловая скорость
        self.pos[1] %= 2*pi

        self.pos[0] += self.speed[0] * dt#поступательная скорость
        self.pos[0].x %= width
        self.pos[0].y %= height

        # Движение означает, что шаг завершён, поэтому
        self.age += 1

    def get_strong(self):
        """ Возвращает текущую силу, которая зависит от многих параметров. Чем больше максимальная энергия -- тем сильнее организм, поэтому energy/maxenergy * maxenergy = energy """
        return self._energy*self._strong

    def _get_lifestate(self):
        return self._energy / self._max_energy

    def apply_accel(self): 
        """ Применить силу к объекту
        Передаёт мощность, с которой двигаться и крутиться
         accel - % от силы, omega [-1..1] -- часть от угла pi, угловое ускорение
         """
        global dt, width, height
        accel, omega = self._accel

        k = dt / self.mass 

        self.speed[1] += omega * k / pi / 10 #применяем угловое ускорение FACTOR

        self.speed[0] += accel * k # применяем линейную скорости

        self._accel = [vect.null(),0]

    def _add_accel(self,accel):
        """ Добавляет ускорение, низкоуровнево"""
        self._accel[0] += accel[0]
        self._accel[1] += accel[1]

    def energy(self,de): # зависит от времени
        """ Управление энергией """
        if self.get_strong() > 0.0001: # если он более-менее живой
            if de > 0: # и получает энергию
                self._energy += de * dt / self.mass # чем больше масса -- тем тяжелее прокормить
            else:
                self._energy += de * dt / self.get_strong() / self.mass # но чем больше масса и стронг -- тем лучше защита
        else:
            self._energy = 0

        if (de < 0) and (self.status > 1):
            self._strong += -de*dt / 100 # FACTOR увеличивается выносливость
            self._max_energy += -de * dt / self._max_energy / 100 / self.mass # FACTOR что бы увеличить max_energy на 0.01 требуется что бы нанесли увечья равные максимальной энергии
        elif self._energy < 0:
            self._energy = 0

        self._change_state()

    def get_pos(self):
        return self.pos[0].x,self.pos[0].y

    def get_pos_screen(self):
        return self.pos[0].x * k_screen, self.pos[0].y * k_screen

    def _change_state(self):
        """ Изменяет статус в зависимости от параметров """
        self.status = 2 # Готов поделиться
        if self._energy <= 0:
            self.status = 0
        elif self._energy < self._max_energy: 
            self.status = 1
        if self.mass <= 0.1: #FACTOR
            self.status = -1
        return self.status

    def _impact_self(self):
        """ Шагает"""
        self._move()

    def _impact_layer(self,layer):
        if layer.__class__ == LayerObjects:
            self._impact_self()

    def create_child(self):
        """ Возвращает ребёнка """
        pass

    def step(self,layer):
        self._impact_layer(layer)        

    def __str__(self):
        return "Pos: (%3d,%3d;%3.f); Pow: %.2f/%.2f; Sp: [w:%2.4f; a:%3.f om:%3.f] M:%3d [A:%1.4f Str: %1.4f] St: %d" %(self.pos[0].x,self.pos[0].y,(self.pos[1]/pi*180),self._energy,self._max_energy,self.speed[0].r,(self.speed[0].phi/pi*180),(self.speed[1]/pi*180),self.mass,self._attack * self._attack_range * self.get_strong(),self._strong,self.status)

class ObjectBot(Object):
    def __init__(self,**args):
        Object.__init__(self,**args)
        self.eyes = Eyes(self)

    def add_accel(self,usk):
        """ Добавляет ускорение по повороту """
        accel, omega = usk
        self._add_accel([Vector(accel,self.pos[1],isPolar = True),omega])

    def _impact_self(self):
        """ Вызывается, когда Родительский слой воздействует сам на себя """
        global layers
        if self.status > 0: # Осторожно! РАБОТА МОЗГА
            state = {"energy":self._energy, "maxenergy": self._max_energy, "radius": self.radius, "vision":self.eyes(layers),"ferromons":(0,0,0,0)}
            ret =  self._mind.step(state)
            #Обработка результатов мозга
            accel = [x for x in ret.get("move",(0,0))]
            self.energy(- (abs(accel[0]) + abs(accel[1])) / 60) #вычитает энергию за передвижение. При k=1 и k_screen = 1, m=5, m_e = 1, e = 1 пройдёт 100 клеток
            accel = [x  * self.get_strong() for x in accel]
            self.add_accel(accel) # применяет ускорение

            self._attack = ret.get("attack",0) # забирает силу атаки
            self.energy(- self._attack / 60) # FACTOR забирает энергию на работу атаки FACTOR: При силе удара 1 и длинне 1(R) при жизни 1 он способен бить 490*k тиков не переставая 
            # применение
        else:
            self._attack = 0
        self.apply_accel()
        self._move()

        if (self.status == 0) or (self.status == -1):
            self.mass -= dt / 10  # за один тик 1/5 е.м.
            if self._change_state() != -1:
                self.radius = sqrt(self.mass)

        #self._attack = 0 # обнуляем

    def _impact_layer(self,layer):
        """ Воздействует на слои """
        if layer.__class__ == LayerObjects:
            self._impact_self()
        #---------------#

    def create_child(self):
        """ Возвращает ребёнка """
        # Вычисляем позицию
        phi = 2 * pi * random.random()
        pos = self.get_pos()
        pos = (2.5 * self.radius * cos(phi) + pos[0], 2.5 * self.radius * sin(phi) + pos[1],phi)
        speed = (cos(phi), sin(phi), 0)
        return ObjectBot(
            pos = pos
            ,mind = self._mind.create_child()
            ,initmnd = ()
            ,energy = self._energy * 0.4
            ,maxenergy = get_mutation(self._max_energy) #FACTOR
            ,speed = speed # мощь, угол; угловая скорость
            ,accelerate = (0,0) # мощь, угловое ускорение
            ,radius = get_mutation(self.radius) #FACTOR
            ,strong = get_mutation(self._strong) #FACTOR
            ,attack_range = get_mutation(self._attack_range) #FACTOR
            ,id = 0
            )

    def step(self,layer):
        """ Вызывает _impact_layer """
        self._impact_layer(layer)

# ========================== PROGRAMM ============================

def step(layers):
    for layer in layers:
        layer.step(layers)

def info():
    h = height - len(layer_obj._objs) - 2
    for obj in layer_obj._objs:
        write_inf(str(obj)+' ')
        h += 1
    write_int('tick: %d; time: %.2f; half fps: %.2f; last fps: %.2f' % (tick,t2-t1,fps,last_fpc))

def tests(test,layer):
    if test == None:
        # for i in range(random.randint(1,100)):
        for i in range(50):
            maxenergy = random.random() + 0.5
            layer.create_obj(ObjectBot(pos = (random.random()*width,random.random()*height,random.random()*2*pi),energy = random.random() * maxenergy,maxenergy = maxenergy,radius = random.random()*2+1))
    elif test == '1':
        layer.create_obj(ObjectBot( pos = (0,40,0),radius = 5,speed = (1,0,0),energy = 0.9 ))
        layer.create_obj(ObjectBot( pos = (40,40,0),radius = 5,speed = (-1,0,-1), energy = 0.9 ))
        layer.create_obj(ObjectBot( pos = (20,40,0),radius = 5,speed = (0,0,0), energy = 0.9 ))
    elif test == '2':
        layer.create_obj(ObjectBot( pos = (0,40,0),radius = 5,speed = (2,0,1),energy = 0.99 ))
        layer.create_obj(ObjectBot( pos = (20,40,0),radius = 5,speed = (0,0,0), energy = 0.99 ))
    elif test == '3':
        layer.create_obj(ObjectBot( pos = (0,40,0),radius = 5,speed = (1,0,0),energy = 0.3 ))
    elif test == '4':
        layer.create_obj(ObjectBot( pos = (0,40,0),radius = 5,speed = (-1,0,0),energy = 0.4 ))
    elif test == '5':
        layer.create_obj(ObjectBot( pos = (10,10,0),radius = 5,speed = (1,0,0),energy = 0.5 ))
        layer.create_obj(ObjectBot( pos = (30,30,0),radius = 5,speed = (0,0,0), energy = 0.9 ))
    elif test == '6':
        layer.create_obj(ObjectBot( pos = (30,40,0),radius = 5,speed = (1,0,1),energy = 0.6 ))
        layer.create_obj(ObjectBot( pos = (4,45,0),radius = 5,speed = (0,0,0), energy = 0.9 ))
    elif test == '7':
        layer.create_obj(ObjectBot( pos = (30,40,0),radius = 5,speed = (6,0,1),energy = 0.7 ))
        layer.create_obj(ObjectBot( pos = (4,45,0),radius = 5,speed = (-1,0,0), energy = 0.9 ))
    elif test == '8':
        layer.create_obj(ObjectBot( pos = (30,40,0),radius = 4,speed = (1,0,1),energy = 0.8 ))
        layer.create_obj(ObjectBot( pos = (4,45,0),radius = 5,speed = (-0.5,0,0), energy = 0.9 ))
    elif test == '9':
        layer.create_obj(ObjectBot( pos = (10,20,0),radius = 4,speed = (1,1,1),energy = 0.9 ))
        layer.create_obj(ObjectBot( pos = (10,40,0),radius = 5,speed = (1,-1,0), energy = 0.9 ))
    elif test == '10':
        layer.create_obj(ObjectBot( pos = (10,35,0),radius = 4,speed = (0,0,0),energy = 0.2 ))
        layer.create_obj(ObjectBot( pos = (10,40,0),radius = 5,speed = (0,0,0), energy = 0.9 ))
    elif test == '11':
        layer.create_obj(ObjectBot( pos = (10,35,0),radius = 4,speed = (0,1,0),energy = 0.2 ))
        layer.create_obj(ObjectBot( pos = (10,40,0),radius = 5,speed = (0,1,0), energy = 0.9 ))
    elif test == '12':
        layer.create_obj(ObjectBot( pos = (10,35,0),radius = 4,speed = (0,1,0),energy = 0.9 ))
        layer.create_obj(ObjectBot( pos = (10,40,0),radius = 5,speed = (0,-1,0), energy = 0.9 ))
    elif test == '13':
        layer.create_obj(ObjectBot( pos = (13,38,pi*2/3),radius = 4,speed = (0,1,0),energy = 0.9, maxenergy = 1 ))
        layer.create_obj(ObjectBot( pos = (10,44,0),radius = 5,speed = (0,-1,0), energy = 0.9))
    elif test == '14':
        layer.create_obj(ObjectBot( pos = (13,38,pi/2),radius = 4,speed = (0,1,0),energy = 9.9,strong = 1.5,maxenergy = 10))
        layer.create_obj(ObjectBot( pos = (10,44,0),radius = 5,speed = (0,-1,0), energy = 0.9))
    elif test == '15':
        layer.create_obj(ObjectBot( pos = (25,38,0),radius = 5,speed = (1,0,0),energy = 0.9,strong = 1,maxenergy = 1))
        layer.create_obj(ObjectBot( pos = (30,38,pi),radius = 5,speed = (-1,0,0), strong = 1, energy = 0.9))
    elif test == '16':
        layer.create_obj(ObjectBot( pos = (5,38,0),radius = 4,speed = (1,0,0),energy = 0.9,strong = 1,maxenergy = 1))
        layer.create_obj(ObjectBot( pos = (30,38,0),radius = 2,speed = (-1,0,0), strong = 1, energy = 0.9))
    elif test == '17':
        layer.create_obj(ObjectBot( pos = (0,38,0),radius = 2,speed = (0,0,0),energy = 1,strong = 1,maxenergy = 1))
    elif test == '18':
        layer.create_obj(ObjectBot( pos = (10,50,pi/2),radius = 1,speed = (0,0,0),energy = 1,strong = 1,maxenergy = 1))
    elif test == '19':
        layer.create_obj(ObjectBot( pos = (0,0,-pi/2),radius = 10,speed = (0,0,0),energy = 0,strong = 1,maxenergy = 1))
        layer.create_obj(ObjectBot( pos = (100,100,pi/2),radius = 10,speed = (0,0,0),energy = 0,strong = 1,maxenergy = 1))
    elif test == '20':
        layer.create_obj(ObjectBot( pos = (38,20,pi/2),radius = 5,speed = (0,1,0),energy = 0.9,strong = 1,maxenergy = 1))
        layer.create_obj(ObjectBot( pos = (38,30,-pi/2),radius = 5,speed = (0,-1,0), strong = 1, energy = 0.9))
    elif test == '21':
        layer.create_obj(ObjectBot( pos = (0,20,0),radius = 5,speed = (1,0,0),energy = 0.9,strong = 1,maxenergy = 1))
        layer.create_obj(ObjectBot( pos = (50,20,-pi),radius = 5,speed = (-1,0,0),energy = 0.9,strong = 1,maxenergy = 1))
    elif test == '22':
        layer.create_obj(ObjectBot( pos = (0,20,random.random()*2*pi),radius = 5,speed = (1,0,0.01),energy = 0.9,strong = 1,maxenergy = 1))
    elif test == '23':
        layer.create_obj(ObjectBot( pos = (0,20,random.random()*2*pi),radius = 5,speed = (0,0,0.1),energy = 0.9,strong = 1,maxenergy = 1))
        layer.create_obj(ObjectBot( pos = (30,20,random.random()*2*pi),radius = 5,speed = (0,0,0),energy = 0.9,strong = 1,maxenergy = 1))
    elif test == '24':
        for x in range(50):
            layer.create_obj(ObjectBot( pos = (random.randint(0,100),random.randint(0,100),random.random()*2*pi),radius = 2 + random.random()*2,speed = (0,0,0),energy = 0.4,strong = 1,maxenergy = 1,mind = Mind_const()))





def field_params(real_width,real_height):
    global width,height,k_screen
    width = 100
    height = 100
    if real_width > real_height:
        height = width / real_width * real_height
    else:
        width = height / real_height * real_width
    k_screen = real_width / width


print("Parse...")
parser = argparse.ArgumentParser(description='Evoluo prealpha')
parser.add_argument("-s","--screen",help="Type output screen:\n curses - curses screen \n screen -- no output")
parser.add_argument("-d","--deltat",help="Set interval of the tick. ")
parser.add_argument("-t","--test",help="Number of test. ")
args = parser.parse_args()
print("Init...")
if args.deltat == None:
    dt = 0.1
else:
    dt = float(args.deltat) # сек^-1
width = 100
height = 100
# field param
print("Layers:")
layer_viscosity = LayerViscosity()
print("Viscosity")
layer_obj = LayerObjects()
tests(args.test,layer_obj)
print("Object")
layers = [layer_viscosity,layer_obj] # layer_obj должен быть всегда в конце, что бы объекты двигались, когда они уже 
print("Ok")

ch = []
tick = 0
t3 = t2 = t1 = time_.time()
last_fps = fps = 0
last_tick = 0
k_mutation = 0.05
Informations = []

if __name__ == '__main__':
    print("Init Screen and start main loop...")

    if args.screen == 'curses':
        print("Start screen...")
        ScreenCursesOut (step,layers)
    elif args.screen == 'opengl':
        print("Start screen...")
        ScreenOpenGL(step,layers)
    else:
        ScreenStandart(step,layers)

    print("dfdfff")

# if __name__ == '__main__':

    

#     while ch != 27:
#         if ch == 105: # i
#             flags["drawinfo"] += 1
#             flags["drawinfo"] %= 2
#         elif ch == 112: # p
#             flags["run"] += 1
#             flags["run"] %= 2
#             screen._scr.nodelay(flags['run'])

#         if flags['run']:
#             screen.clear()
#             if flags["drawinfo"]:
#                 sum = 0
#                 for obj in layer_obj.get_objs():
#                     screen.write_inf(str(obj)+' ')
#                     sum += obj.speed[0].r*obj.mass
#                 screen.write_inf('Общий импульс системы:'+str(sum))
#                 screen.write_inf('tick: %d; Gtime: %.2f; half fps: %.2f; last fps: %.2f' % (tick,tick*dt,fps,last_fps))
            
#             step(layers) # Самое главное
#             tick += 1
#             t2 = time_.time()
#             if (tick - last_tick) - last_fps > 0.:
#                 last_fps = (tick - last_tick) / (t2 - t3)
#                 last_tick = tick 
#                 t3 = t2
#             fps = 0
#             if t2 != t1:
#                 fps = tick / (t2-t1)

#             screen.draw(layer_obj)
#             screen.update()

#         # if args.screen != 'curses':
#             # if tick > 2000:
#                  # break
#         ch = screen.getch()
#         #screen.scr.addstr(22,0,str(c))
#     del screen
#     print("===========================================\nTicks = %d, half FPS: %.3f, last FPS: %.3f\n Width:%d, Height:%d" %(tick,fps,last_fps,width,height))
