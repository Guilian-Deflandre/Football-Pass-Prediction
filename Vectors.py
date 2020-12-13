import math

def dot(v,w):
    """
    Compute a matrix containing th distances between each players.

    PARAMETERS
    pass_: array[1, 47]
        The information of a pass (x and y coordinate of each players, id of
        the sender, time in ms since the half start and the if of the pass).

    RETURN
    distance_matrix: array[22, 22]
        The matrix of distances between each players.
    """
    x,y = v
    X,Y = w
    return x*X + y*Y

def norm(v):
    x,y = v
    return math.sqrt(x*x + y*y)

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def unit(v):
    x,y = v
    mag = norm(v)
    return (x/mag, y/mag)

def distance(p0,p1):
    return norm(vector(p0,p1))

def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)

def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)
