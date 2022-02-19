
import matplotlib.pyplot as plt

class ViZDoomMap():
    def __init__(self):
        self.vertices = Vertices()
        self.rooms = {}

        self._setRooms()

    def _setRooms(self):
        self.rooms['10'] = self.vertices.getVertices(1,2,3,5)
        self.rooms['11'] = self.vertices.getVertices(13,7,11,9)
        self.rooms['12'] = self.vertices.getVertices(68,8,35,69)
        self.rooms['13'] = self.vertices.getVertices(34,39,36,37)
        self.rooms['14'] = self.vertices.getVertices(71,38,99,72)
        self.rooms['15'] = self.vertices.getVertices(99,73,70,74)
        self.rooms['16'] = self.vertices.getVertices(10,76,75,22)
        self.rooms['17'] = self.vertices.getVertices(25,24,21,23)
        self.rooms['18'] = self.vertices.getVertices(65,30,26,64)
        self.rooms['19'] = self.vertices.getVertices(67,16,12,66)
        self.rooms['20'] = self.vertices.getVertices(19,17,20,15)
        self.rooms['21'] = self.vertices.getVertices(18,63,62,28)
        self.rooms['22'] = self.vertices.getVertices(33,31,27,29)
        self.rooms['23'] = self.vertices.getVertices(32,60,59,41)
        self.rooms['24'] = self.vertices.getVertices(45,44,40,42)
        self.rooms['25'] = self.vertices.getVertices(56,43,51,55)
        self.rooms['26'] = self.vertices.getVertices(47,50,46,48)
        self.rooms['vest'] = self.vertices.getVertices(53,49,52,54)
        self.rooms['corridor10-11'] = self.vertices.getVertices(4,6,57,14)

    def getRoom(self,x,y):
        x = int(x)
        y = int(y)

        if x == 240 and y == -176:
            roomID = 10
        elif x == 464 and y == -176:
            roomID = 11
        elif x == 470 and y == -321:
            roomID = 12
        elif x == 466 and y == -466:
            roomID = 13
        elif x == 460 and y == -596:
            roomID = 14
        elif x == 543  and y == -672:
            roomID = 15
        elif x == 575 and y == -171:
            roomID = 16
        elif x == 686 and y == -173:
            roomID = 17
        elif x == 686 and y == -62:
            roomID = 18
        elif x == 464 and y == -60:
            roomID = 19
        elif x == 460 and y == 56:
            roomID = 20
        elif x == 571 and y == 54:
            roomID = 21
        elif x == 694 and y == 58:
            roomID = 22
        elif x == 871 and y == 54:
            roomID = 23
        elif x == 1043 and y == 60:
            roomID = 24
        elif x == 1039 and y == -61:
            roomID = 25
        elif x == 1039 and y == -169:
            roomID = 26
        else:
            roomID = 27

        return roomID

    def getRoomLines(self,roomID):
        ne,no,se,so = self._getNSEO(roomID)
        return (no[0], ne[0], se[0], so[0], no[0]) , (no[1], ne[1], se[1], so[1], no[1] )

    def _getNSEO(self,roomID):
        v1,v2,v3,v4 = self.rooms[str(roomID)]

        # Norte-Este
        ne = v1
        for v in [v2,v3,v4]:
            x,y = v
            if x >= ne[0] and y >= ne[1]:
                ne = v
        # Norte-Oeste
        no = v1
        for v in [v2,v3,v4]:
            x,y = v
            if x <= no[0] and y >= no[1]:
                no = v
        # Sur-Este
        se = v1
        for v in [v2,v3,v4]:
            x,y = v
            if x >= se[0] and y <= se[1]:
                se = v
        # Sur-Oeste
        so = v1
        for v in [v2,v3,v4]:
            x,y = v
            if x <= so[0] and y <= so[1]:
                so = v
        return ne,no,se,so

    def plotViZDoom_map(self):
        vizdoom_map = plt.figure(figsize=(10,10))
        # mapa = ViZDoomMap()

        x_map = []
        y_map = []
        for roomID,room_vertices in self.rooms.items():
            for vertices in room_vertices:
                x_map.append(vertices[0])
                y_map.append(vertices[1])
            lines = self.getRoomLines(roomID)
            plt.plot(lines[0],lines[1],color='black')
        # plot all vertices
        plt.plot(x_map, y_map,'.',color='black')
        return plt

    def pointInRoom(self,roomID,point):
        # v1,v2,v3,v4 = self.rooms[str(roomID)]
        ne,no,se,so = self._getNSEO(roomID)
        x, y = point[0],point[1]
        x_min = no[0]
        x_max = ne[0]
        y_min = se[1]
        y_max = ne[1]

        # check if point is inside the area limited by those points
        if (x >= x_min and x<=x_max) and (y >= y_min and y<=y_max):
            return True
        return False

class Vertices():
    def __init__(self):
        self.vertices = {}
        self._fill_vertices()

    def getVertices(self,v1,v2,v3,v4):
        return self.vertices[str(v1)],self.vertices[str(v2)],self.vertices[str(v3)],self.vertices[str(v4)]

    def getAllVertices(self):
        return self.vertices

    def _fill_vertices(self):
        # room10
        self.vertices['1'] = [160,-256]
        self.vertices['2'] = [160,-96]
        self.vertices['3'] = [320,-96]
        self.vertices['5'] = [320,-256]

        # corridor 10-11
        self.vertices['4'] = [320,-144]
        self.vertices['6'] = [320,-208]
        self.vertices['57'] = [384,-144]
        self.vertices['14'] = [384,-208]

        # room 11
        self.vertices['13'] = [384,-256]
        self.vertices['7'] = [544,-256]
        self.vertices['11'] = [384,-96]
        self.vertices['9'] = [544,-96]

        # room 12
        self.vertices['68'] = [432,-256]
        self.vertices['8'] = [496,-256]
        self.vertices['35'] = [432,-384]
        self.vertices['69'] = [496,-384]

        # room 13
        self.vertices['34'] = [384,-384]
        self.vertices['39'] = [544,-384]
        self.vertices['36'] = [384,-544]
        self.vertices['37'] = [544,-544]

        # room 14
        self.vertices['71'] = [432,-544]
        self.vertices['38'] = [496,-544]
        self.vertices['99'] = [432,-640]
        self.vertices['72'] = [496,-640]

        # room 15
        self.vertices['99'] = [432,-640]
        self.vertices['73'] = [576,-640]
        self.vertices['70'] = [432,-704]
        self.vertices['74'] = [576,-704]

        # room 16
        self.vertices['10'] = [544,-144]
        self.vertices['76'] = [608,-144]
        self.vertices['75'] = [544,-208]
        self.vertices['22'] = [608,-208]

        # room 17
        self.vertices['25'] = [608,-96]
        self.vertices['24'] = [768,-96]
        self.vertices['21'] = [608,-256]
        self.vertices['23'] = [768,-256]

        # room 18
        self.vertices['65'] = [656,-32]
        self.vertices['30'] = [720,-32]
        self.vertices['26'] = [656,-96]
        self.vertices['64'] = [720,-96]

        # room 19
        self.vertices['67'] = [432,-32]
        self.vertices['16'] = [496,-32]
        self.vertices['12'] = [432,-96]
        self.vertices['66'] = [496,-96]

        # room 20
        self.vertices['19'] = [384,128]
        self.vertices['17'] = [544,128]
        self.vertices['20'] = [384,-32]
        self.vertices['15'] = [544,-32]

        # room 21
        self.vertices['18'] = [544,80]
        self.vertices['63'] = [608,80]
        self.vertices['62'] = [544,16]
        self.vertices['28'] = [608,16]

        # room 22
        self.vertices['33'] = [608,128]
        self.vertices['31'] = [768,128]
        self.vertices['27'] = [608,-32]
        self.vertices['29'] = [768,-32]

        # room 23
        self.vertices['32'] = [768,80]
        self.vertices['60'] = [960,80]
        self.vertices['59'] = [768,16]
        self.vertices['41'] = [960,16]

        # room 24
        self.vertices['45'] = [960,128]
        self.vertices['44'] = [1120,128]
        self.vertices['40'] = [960,-32]
        self.vertices['42'] = [1120,-32]

        # room 25
        self.vertices['56'] = [1008,-32]
        self.vertices['43'] = [1072,-32]
        self.vertices['51'] = [1008,-96]
        self.vertices['55'] = [1072,-96]

        # room 26
        self.vertices['47'] = [960,-96]
        self.vertices['50'] = [1120,-96]
        self.vertices['46'] = [960,-256]
        self.vertices['48'] = [1120,-256]

        # vest room
        self.vertices['53'] = [992,-256]
        self.vertices['49'] = [1088,-256]
        self.vertices['52'] = [992,-416]
        self.vertices['54'] = [1088,-416]

def getRoomsFromSettings(setting):
    if setting=='dense':
        return [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    elif setting=='sparse':
        return [10]
    else:
        return [15]
