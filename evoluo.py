#!/usr/bin/env python3
#-*- coding:utf-8 -*-
""" Программа-симулятор эволюции. Тестовая версия.
Основные положения:
В самой программе и в визуализации вывод будет, скорее всего, начиная с нижней левой точки как (0,0) 
Итак, что вообще и как происходит.
В главном цикле вызывается функция step(), которая перебирает все слои и отдаёт им массив со всеми слоями.
"""
import random,math,pdb,noise,argparse,copy
import time as time_
import vect
from vect import Vector

def getpair(a,b): #костыль для быстродействия
    if a>b:
        return [b,a]
    else:
        return [a,b]

def get_min_distance(param,p1,p2): #проверена
    """ Возвращает минимальное расстояние между двумя точками на поле """
    _p1 = [p1.x, p1.y]
    _p2 = [p2.x, p2.y]
    # _p1[0] += math.trunc((p2[0] - p1[0]) / (0.5*param[0])) * param[0]
    # _p1[1] += math.trunc((p2[1] - p1[1]) / (0.5*param[1])) * param[1]
    _r = lambda x,y: math.sqrt(x*x + y*y)
    return _r(_p1[0] - _p2[0] + math.trunc((_p2[0] - _p1[0]) / (0.5*param[0])) * param[0]
        ,_p1[1] - _p2[1] + math.trunc((_p2[1] - _p1[1]) / (0.5*param[1])) * param[1])

def simm_r(param:'разрешение экрана',p1,p2:'координаты точек',func,deep:'глубина' = 1): #проверена
    """ Получает 'симметричное' расстояние, т.е. если бы вокруг основного экрана был бы квадрат со стороной (1+deep*2) и считает для этого какой либо коэффициент через функцию, где функция принимает на вход расстояние от точки до точки. """
    def _r(a,b):
        return ((a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2) ** 0.5
    w = param[0]
    h = param[1]
    X = (deep*w)+p1[0]
    Y = (deep*h)+p1[1]
    ans = 0
    for y in range(1+2*deep):
        for x in range(1+2*deep):
            ans+=func(_r((X,Y),(p2[0]+w*x,p2[1]+h*y)))
    return ans

def get_distribution(param,f0):
    """ Возвращает функцию с распределением таким, что при deep = 0 и dx = w/2 dy = h/2 даёт 0.1, а при r = 0 даёт 1, формула 1/(x^alpha+1).
    Является генератором функций.
    Пока особо не нужна, но понадобится при создании солнечного света"""
    r = (param[0] ** 2 + param[1] ** 2) ** 0.5
    alpha = math.log(1/f0-1)/math.log(r)
    def _dist(x):
        return 1/(x ** alpha + 1)
    return _dist

def write_arr(arr):
    for a in arr:
        print(end='[')
        for el in a:
            print("%.2f" %el,end=', ')
        print(']')

def _init_get_under():
    """ Раньше эта функция жила в Layer, но так как она жутко ресурсозатратная, было решено вынести её в отдельное место и просчитать с самого начала, ибо уменьшение производительности в 30 раз меня огорчило.
    !!! заполнять вписанный квадрат в окружность, а всё оставшееся считать, проверить на скорость"""
    global width,height,screen
    maxR = round(min(width/2,height/2))
    _get_under = [[] for r in range(maxR)]

    for R in range(1,maxR+1):
        for y in range(-R,+R+1):
            for x in range(-R,+R+1):
                if get_min_distance((width,height),(x,y),(0,0)) <= R:
                    _get_under[R-1].append([x,y])
    return _get_under

def _init_get_under_new():
    global width,height,screen
    maxR = round(min(width/2,height/2))
    _get_under = [[] for r in range(maxR)]
    v_0 = Vector(0,0)
    for R in range(1,maxR+1):
        _quart = []
        for x in range(1,R+1):
            for y in range(1,R+1):
                if (x + y >= R):
                    if get_min_distance((width,height),Vector(x,y),v_0) <= R:
                        _quart.append([x,y])
                else:
                    _quart.append([x,y])

        _get_under[R-1] = (_quart
                        + [[-x,y] for x,y in _quart]
                        + [[x,-y] for x,y in _quart]
                        + [[-x,-y] for x,y in _quart]
                        + [[x,0] for x in range(-R,R+1)]
                        + [[0,y] for y in range(-R,R+1) if y != 0])
        screen.write((0,0),"%d" % (R))
        screen.update()
    return _get_under

class Screen:
    """ Суперкласс экрана. Служит основой для других классов """
    def __init__(self):
        """ Инициализирует экран """
        self.type = "Default Screen"
        self._scr = None
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
    def stop(self):
        """ Останавливает экран """
        pass
    def getch(self):
        return 0
    def __del__(self):
        """ Деструктор класса """
        pass

import curses

class Curses(Screen):
    def __init__(self):
        global width, height
        self.type = "Curses Screen"
        self._scr = curses.initscr()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_WHITE)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_WHITE)
        curses.noecho()
        curses.cbreak()
        self._scr.keypad(1)
        self._scr.clear()
        self._scr.nodelay(1)
        height,width = self._scr.getmaxyx()

    def line(self,pos,vect,color):
        def _from(a,b):
            if a > b:
                return reversed(range(b,a+1))
            else:
                return range(a,b+1)
        x = [0,0]
        y = [0,0]

        x[0], y[0] = pos.x, pos.y
        x[1], y[1] = vect.x + pos.x, vect.y + pos.y

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
        
    def draw(self,layer):
        if layer.__class__ == LayerObjects:
            for obj in layer.get_objs():

                def _wrt(x,y,obj):
                    self.write((x,y)," ",curses.A_REVERSE)

                layer.get_under(obj.pos,obj.radius,_wrt,obj) # вырисовываем круг
                self.line(obj.pos[0],Vector(obj.radius+obj._attack_range*obj._attack,obj.pos[1],isPolar = True),2)
                self.line(obj.pos[0],obj.speed[0] * 3,1)
                self.write((int(obj.pos[0].x),int(obj.pos[0].y)),"%d" %(obj._get_lifestate()*10),curses.A_REVERSE)

                # сделать вывод скорости и ускорения
        elif layer.__class__ == LayerViscosity:
            pass

    def update(self):
        self._scr.refresh()

    def clear(self):
        for y in range(height):
                self.write((0,y),' ' * width)
        #self._scr.clear()

    def write(self,pos,str,*attr):
        self._scr.addstr(pos[1] % height,pos[0] % width,str,*attr)

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
        """ Даёт список ссылок на клетки слоя, которые находятся под окружностью радиуса R.
        На будущее: есть возможность оптимизации, вида вынести все эти расчёты во вне, например массивами размера RxR, где показывается, брать или не брать эту клетку. Переделать в иттератор!"""
        _R = int(round(R))
        dx = int(pos[0].x)
        dy = int(pos[0].y)
        for x,y in _get_under[_R-1]:
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

    def _create_obj(self,obj):
        """ Создаёт объект """
        self._objs.update({random.randint(0,10000000):obj})

    def get_objs(self):
        return list(self._objs.values())

    def get_obj_by_id(self,id):
        return self._objs.get(id,None)

    def _impact_layer(self,layer):
        """ Для каждого объекта внутри себя заставляет его обработать слой """
        for obj in self.get_objs():
            obj.step(layer)

    def _collision(self,i,k,inside):
        """ Просчитывает скорости объектов
        РАсчёт коэффициента: 1.1"""
        _objs = self.get_objs()
        p1 = _objs[i].pos[0]
        v1 = _objs[i].speed[0]
        w1 = _objs[i].speed[1]
        m1 = _objs[i].mass
        p2 = _objs[k].pos[0]
        v2 = _objs[k].speed[0]
        w2 = _objs[k].speed[1]
        m2 = _objs[k].mass
        #получаем вектор-прямую, на которую будут откладываться взаимодействующие вектора
        phi = Vector(p2.x - p1.x, p2.y - p1.y,isPolar = False).phi #угол линии с OX, соединяющий центры

        # Меняем скорости, если они столкнулись только что
        if self._colliding[0].count(getpair(i,k)) == 0:
            def _get(m1,m2,v1,v2):
                """ уравнение полученных скоростей, чтобы по 10 раз не писать одно и то же """ 
                return (v1 * (m1 - m2) + v2 * 2 * m2) / (m1 + m2)
            v1.phi -= phi # поворот так, что линия есть OX 
            v2.phi -= phi
            _v1 = v1.copy()
            _v2 = v2.copy()
            V1before = _v1.x
            V2before = _v2.x
            V1after = _v1.x = _get(m1,m2,v1.x,v2.x) # взаимодействуют
            V2after = _v2.x = _get(m2,m1,v2.x,v1.x)
            _v1.phi += phi #поворачиваем обратно
            _v2.phi += phi
            _objs[i].speed[0] = _v1
            _objs[k].speed[0] = _v2

            mass12 = _objs[k].mass / _objs[i].mass
            _objs[i].speed[1] -= (w2 * mass12 + w1) / 10 # сохраняем импульс и передаём 1/10 от угловой скорости FACTOR
            _objs[k].speed[1] -= (w1 / mass12 + w2) / 10

            if (V1after-V1before)/m1>_objs[i].get_strong():
                _objs[i].energy(-((V1after-V1before)-_objs[i].get_strong()/m1))
            if (V2after-V2before)/m2>_objs[k].get_strong():
                _objs[k].energy(-((V2after-V2before)-_objs[k].get_strong()/m2))
        else:
            F = Vector(inside,phi,isPolar = True)
            _objs[i]._add_accel([-F,0]) #добавляем ускорение
            _objs[k]._add_accel([F,0])

    def collision(self):
        """ Определяет, сталкиваются ли объекты """
        _objs = self.get_objs()
        lenght = len(_objs)
        for i in range(lenght):
            for j in range(i+1,lenght):
                inside = (get_min_distance((self.width,self.height),_objs[i].pos[0],_objs[j].pos[0]) - (_objs[i].radius+_objs[j].radius)) / 2
                if inside < 0:
                    self._collision(i,j,-inside)
                    self._colliding[1].append(getpair(i,j)) # добавляет сталкивающиеся объекты в новый, чтобы на следующем шаге показать, что они уже сталкивались и надо их посильнее оттолкнуть
        self._colliding[0] = copy.copy(self._colliding[1])
        self._colliding[1] = []

    def _attack(self,a,b):
        #Расчёт того, насколько глубоко проник удар I.3
        OO = Vector(b.pos[0].x-a.pos[0].x, b.pos[0].y-a.pos[0].y, isPolar = False) # Вектор, соединяющий центры ботов
        at = Vector(a.radius + a._attack * a._attack_range * a.get_strong(), a.pos[1], isPolar = True) #вектор атаки
        phi = OO.phi - at.phi # угол
        ah = Vector(math.cos(phi)*OO.r, at.phi, isPolar = True) # вектор до высоты, опущенной из центра b
        f_at = at.r-ah.r # первая часть проникновения
        OH = (OO - ah).r # длинна высота из центра b
        if OH < b.radius:
            BH = math.sqrt(b.radius*b.radius - OH*OH)
            f_at += BH
            if f_at > 0:
                # pdb.set_trace()
                f_at *= a._attack * a.get_strong() #вычисляем силу атаки, заглушка
                if BH > 0.001: # вычисляем угловое ускорение
                    omega = f_at * math.sin(math.atan(OH/BH)) # учитываем энергию a и силу удара
                else:
                    omega = 0
                b.energy (- f_at)
                b._add_accel((OO.one() * f_at, omega)) #отталкиваем и крутим противника


    def attack(self):
        _objs = self.get_objs()
        lenght = len(_objs)
        for a in _objs:
            for b in _objs:
                if (a != b):
                    inside = (get_min_distance((self.width,self.height),a.pos[0],b.pos[0]) - (a.radius+b.radius)) / 2
                    if inside < a._attack_range:
                        self._attack(a,b) # Атакует a


    def step(self,layers):
        for layer in layers:
            self._impact_layer(layer)
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

    def _impact_self(self):
        """ Как слой изменяется """
        pass # да никак

    def _impact_layer(self,layer):
        if layer.__class__ == LayerObjects:
            for obj in layer.get_objs():
                self._impact_obj(obj)
        elif layer.__class__ == LayerViscosity:
            self._impact_self()


class Mind:
    """ (Тестовый) Суперкласс разума. На вход подаются данные с датчиков, на выход -- действия """
    def __init__(self,init):
        pass
    def step(self,args):
        move = (math.sin(args.get("energy",0) / args.get("maxenergy",1)),0)
        #move = (0,0)
        attack = 0.1 # -- сила удара
        ferromons = (1,0,0,0) # 2^4 = 16 ферромонов, в данный момент он отдаёт ферромон 1000
        return {"move":move, "attack":attack, "ferromons": ferromons}


class Object: 
    """ Суперкласс объекта. Служит основой для других классов объектов. Статус: 0 -- мёртв (<0% энергии), 1 -- в спячке (<10% энергии), 2 -- жив \n
    !!!!Переписать: то что нужно, оставить здесь, всё остальное вынести в уже класс ObjectBot, который будет прототипом бота. Здесь же учесть нужно лишь _общие_ моменты объектов, и не засорять, т.к. у нас будут ещё и твёрдые вещества, и всё что угодно"""
    def __init__(self
            ,pos:'местоположение' = (0,0,0) # x,y; угол
            ,mind:'Мозги' = Mind
            ,initmnd:'Данные для инициализации мозгов' = ()
            ,energy:'Энергия' = 0.1
            ,maxenergy:'Максимальная энергия' = 1
            ,speed:'Скорость' = (0,0,0) # мощь, угол; угловая скорость
            ,accelerate:'Ускорение' = (0,0) # мощь, угловое ускорение
            ,radius:'радиус объекта' = 1
            ,strong:'Защита и сила; от неё зависят многие параметры' = 1
            ,attack_range: 'Дальность атаки' = 1
            ,**args):
        self.pos = [Vector(*pos[0:2]), pos[2]] #положение -- координата + угол
        self._energy = energy # Текущая энергия
        self._max_energy = maxenergy # максимальная энергия
        self.speed = [Vector(*speed[0:2]), speed[2]] # начальные скорости
        self._accel = [Vector(accelerate[0],speed[2]),accelerate[1]] # ускорение
        self.radius = radius # радиус
        self.mass = radius ** 2 # масса
        self.status = 2 # статус
        self._strong = strong #Защита и сила
        self.mind = mind(initmnd) # Мозги
        self._attack = 0 #сила атаки
        self._attack_range = attack_range #дальность атаки

    def _move(self): 
        """ Передвигает объект на вектор """
        self.pos[1] += self.speed[1] * dt # угловая скорость
        self.pos[1] %= 2*math.pi

        self.pos[0] += self.speed[0] * dt #поступательная скорость
        self.pos[0].x %= width
        self.pos[0].y %= height

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

        self.speed[1] += omega * k / math.pi / 10 #применяем угловое ускорение FACTOR

        self.speed[0] += accel * k # применяем линейную скорости

        self._accel = [vect.null(),0]

    def _add_accel(self,accel):
        """ Добавляет ускорение, низкоуровнево"""
        self._accel[0] += accel[0]
        self._accel[1] += accel[1]

    def energy(self,de): # зависит от времени
        """ Управление энергией """
        self._energy += de * dt
        if (de < 0) and (self.status > 1):
            self._strong += -de*dt / 100 # FACTOR
        return_energy = 0
        if self._energy > self._max_energy:
            return_energy = self._energy - self._max_energy 
            self._energy = self._max_energy
        elif self._energy < 0:
            return_energy = -self._energy
            self._energy = 0
        return return_energy

    def get_pos(self):
        return self.pos[0].x,self.pos[0].y

    def _change_state(self):
        """ Изменяет статус в зависимости от параметров """
        self.status = 2
        if self._energy < self._max_energy*0.05: #FACTOR
            self.status = 0
        elif self._energy < self._max_energy*0.1 : #FACTOR
            self.status = 1

    def _impact_self(self):
        """ Шагает"""
        self._move()
        self._change_state()

    def _impact_layer(self,layer):
        if layer.__class__ == LayerObjects:
            self._impact_self()

    def step(self,layer):
        self._impact_layer(layer)

    def __str__(self):
        return "Pos: (%3d,%3d;%3.f); Pow: %.2f/%.2f; Sp: [w:%2.4f; a:%3.f om:%3.f] M:%3d [A:%1.4f Str: %1.4f] St: %d" %(self.pos[0].x,self.pos[0].y,(self.pos[1]/math.pi*180),self._energy,self._max_energy,self.speed[0].r,(self.speed[0].phi/math.pi*180),(self.speed[1]/math.pi*180),self.mass,self._attack * self._attack_range * self.get_strong(),self._strong,self.status)


class ObjectBot(Object):
    def add_accel(self,usk):
        """ Добавляет ускорение по повороту """
        accel, omega = usk
        self._add_accel([Vector(accel,self.pos[1],isPolar = True),omega])

    def _impact_self(self):
        """ Вызывается, когда Родительский слой воздействует сам на себя """
        if self.status == 2: # Осторожно! РАБОТА МОЗГА
            state = {"energy":self._energy, "maxenergy": self._max_energy, "radius": self.radius, "vision":[0,0,0,0,0,0,0],"ferromons":(0,0,0,0)}
            ret =  self.mind.step(state)
        else:
            ret = {}
        #Обработка результатов мозга
        accel = [x * self.get_strong() for x in ret.get("move",(0,0))]
        self.add_accel(accel) # применяет ускорение
        self.energy(- (abs(accel[0]) + abs(accel[1])) / area) #вычитает энергию в зависимости от площади
        self._attack = ret.get("attack",0) # забирает силу атаки
        self.energy(- self._attack * self._attack_range * self.get_strong() / 10) #забирает энергию на работу атаки FACTOR: При силе удара 1 и длинне 1 при жизни 1 он способен бить 10 тиков не переставая 
        # применение
        self.apply_accel()
        self._move()
        self._change_state()

        #self._attack = 0 # обнуляем

    def _impact_layer(self,layer):
        """ Воздействует на слои """
        if layer.__class__ == LayerViscosity:
            pass # никак
        #---------------#
        elif layer.__class__ == LayerObjects:
            self._impact_self()
        #---------------#

    def step(self,layer):
        """ Вызывает _impact_layer """
        self._impact_layer(layer)


def step(layers):
    for layer in layers:
        layer.step(layers)

def info():
    h = height - len(layer_obj._objs) - 2
    for obj in layer_obj._objs:
        screen.write((0,h),str(obj)+' ')
        h += 1
    screen.write((0,h),'tick: %d; time: %.2f; half fps: %.2f; last fps: %.2f' % (tick,t2-t1,fps,last_fpc))

width = 100
height = 100

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evoluo prealpha')
    parser.add_argument("-s","--screen",help="Type output screen:\n curses - curses screen \n screen -- no output")
    parser.add_argument("-d","--deltat",help="Set interval of the tick. ")
    parser.add_argument("-t","--test",help="Number of test. ")
    args = parser.parse_args()

    #Коэффициент шага
    if args.deltat == None:
        dt = 0.1
    else:
        dt = float(args.deltat) # сек^-1


    flags = {"drawinfo":1,"run":1}

    if args.screen == 'curses':
        screen = Curses()
    elif args.screen == "tkinter":
        screen = ScreenTkinter()
    else:
        screen = Screen()

    height -= 1

    area = width * height
    _get_under = _init_get_under_new()

    layer_obj = LayerObjects()

    if args.test == None:
        for i in range(random.randint(1,50)):
            layer_obj._create_obj(ObjectBot(pos = (random.randint(0,width-1),random.randint(0,height-1),random.random()),energy = random.random(),maxenergy = random.random(),radius = random.random()*7+1))
    elif args.test == '1':
        layer_obj._create_obj(ObjectBot( pos = (0,40,0),radius = 5,speed = (1,0,0),energy = 0.9 ))
        layer_obj._create_obj(ObjectBot( pos = (40,40,0),radius = 5,speed = (-1,0,-1), energy = 0.9 ))
        layer_obj._create_obj(ObjectBot( pos = (20,40,0),radius = 5,speed = (0,0,0), energy = 0.9 ))
    elif args.test == '2':
        layer_obj._create_obj(ObjectBot( pos = (0,40,0),radius = 5,speed = (2,0,1),energy = 0.99 ))
        layer_obj._create_obj(ObjectBot( pos = (20,40,0),radius = 5,speed = (0,0,0), energy = 0.99 ))
    elif args.test == '3':
        layer_obj._create_obj(ObjectBot( pos = (0,40,0),radius = 5,speed = (1,0,0),energy = 0.3 ))
    elif args.test == '4':
        layer_obj._create_obj(ObjectBot( pos = (0,40,0),radius = 5,speed = (-1,0,0),energy = 0.4 ))
    elif args.test == '5':
        layer_obj._create_obj(ObjectBot( pos = (10,10,0),radius = 5,speed = (1,0,0),energy = 0.5 ))
        layer_obj._create_obj(ObjectBot( pos = (30,30,0),radius = 5,speed = (0,0,0), energy = 0.9 ))
    elif args.test == '6':
        layer_obj._create_obj(ObjectBot( pos = (30,40,0),radius = 5,speed = (1,0,1),energy = 0.6 ))
        layer_obj._create_obj(ObjectBot( pos = (4,45,0),radius = 5,speed = (0,0,0), energy = 0.9 ))
    elif args.test == '7':
        layer_obj._create_obj(ObjectBot( pos = (30,40,0),radius = 5,speed = (6,0,1),energy = 0.7 ))
        layer_obj._create_obj(ObjectBot( pos = (4,45,0),radius = 5,speed = (-1,0,0), energy = 0.9 ))
    elif args.test == '8':
        layer_obj._create_obj(ObjectBot( pos = (30,40,0),radius = 4,speed = (1,0,1),energy = 0.8 ))
        layer_obj._create_obj(ObjectBot( pos = (4,45,0),radius = 5,speed = (-0.5,0,0), energy = 0.9 ))
    elif args.test == '9':
        layer_obj._create_obj(ObjectBot( pos = (10,20,0),radius = 4,speed = (1,1,1),energy = 0.9 ))
        layer_obj._create_obj(ObjectBot( pos = (10,40,0),radius = 5,speed = (1,-1,0), energy = 0.9 ))
    elif args.test == '10':
        layer_obj._create_obj(ObjectBot( pos = (10,35,0),radius = 4,speed = (0,0,0),energy = 0.2 ))
        layer_obj._create_obj(ObjectBot( pos = (10,40,0),radius = 5,speed = (0,0,0), energy = 0.9 ))
    elif args.test == '11':
        layer_obj._create_obj(ObjectBot( pos = (10,35,0),radius = 4,speed = (0,1,0),energy = 0.2 ))
        layer_obj._create_obj(ObjectBot( pos = (10,40,0),radius = 5,speed = (0,1,0), energy = 0.9 ))
    elif args.test == '12':
        layer_obj._create_obj(ObjectBot( pos = (10,35,0),radius = 4,speed = (0,1,0),energy = 0.9 ))
        layer_obj._create_obj(ObjectBot( pos = (10,40,0),radius = 5,speed = (0,-1,0), energy = 0.9 ))
    elif args.test == '13':
        layer_obj._create_obj(ObjectBot( pos = (13,38,math.pi*2/3),radius = 4,speed = (0,1,0),energy = 0.9, maxenergy = 1 ))
        layer_obj._create_obj(ObjectBot( pos = (10,44,0),radius = 5,speed = (0,-1,0), energy = 0.9))
    elif args.test == '14':
        layer_obj._create_obj(ObjectBot( pos = (13,38,math.pi/2),radius = 4,speed = (0,1,0),energy = 9.9,strong = 1.5,maxenergy = 10))
        layer_obj._create_obj(ObjectBot( pos = (10,44,0),radius = 5,speed = (0,-1,0), energy = 0.9))

    layer_viscosity = LayerViscosity()

    layers = [layer_viscosity,layer_obj] # layer_obj должен быть всегда в конце, что бы объекты двигались, когда они уже 
    ch = 0 
    tick = 0
    t3 = t2 = t1 = time_.time()
    last_fps = fps = 0
    last_tick = 0
    h0 = height - len(layer_obj._objs) - 2
    while ch != 27:
        if ch == 105: # i
            flags["drawinfo"] += 1
            flags["drawinfo"] %= 2
        elif ch == 112: # p
            flags["run"] += 1
            flags["run"] %= 2
            screen._scr.nodelay(flags['run'])

        screen.clear()
        if flags["drawinfo"]:
            h = h0
            sum = 0
            for obj in layer_obj.get_objs():
                screen.write((0,h),str(obj)+' ')
                h += 1
                sum += obj.speed[0].r*obj.mass
            screen.write((0,h),'Общий импульс системы:'+str(sum))
            screen.write((0,h+1),'tick: %d; Gtime: %.2f; half fps: %.2f; last fps: %.2f' % (tick,tick*dt,fps,last_fps))

        screen.draw(layer_obj)

        if flags['run']:
            step(layers) # Самое главное
            tick += 1
            t2 = time_.time()
            if (tick - last_tick) - last_fps > 0.:
                last_fps = (tick - last_tick) / (t2 - t3)
                last_tick = tick 
                t3 = t2
            fps = 0
            if t2 != t1:
                fps = tick / (t2-t1)

            screen.update()

        if args.screen != 'curses':
            if tick > 2000:
                 break
        ch = screen.getch()
        #screen.scr.addstr(22,0,str(c))
    del screen
    print("===========================================\nTicks = %d, half FPS: %.3f, last FPS: %.3f\n Width:%d, Height:%d" %(tick,fps,last_fps,width,height))
