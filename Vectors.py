import math

def dot(v,w):
    """
    Compute the dot product of vectors v and w.

    PARAMETERS
    v, w: (x, y) ((float, float))
        2D components of vectors v, w.

    RETURN
    dot : float
        The dot product of v and w.
    """
    x,y = v
    X,Y = w
    dot = x*X + y*Y
    return dot

def norm(v):
    """
    Compute the Euclidean norm of vector v.

    PARAMETERS
    v: (x, y) ((float, float))
        2D components of vectors v.

    RETURN
    norm : float
        The Euclidean norm of v.
    """
    x,y = v
    norm = math.sqrt(x*x + y*y)
    return norm

def vector(a,b):
    """
    Compute the 2D components of vector a->b.

    PARAMETERS
    a, b: (x, y) ((float, float))
        2D coordinates of points a,b.

    RETURN
    vector : (v1, v2) ((float, float))
        2D components of vector a->b.
    """
    x,y = a
    X,Y = b
    vector = (X-x, Y-y)
    return vector

def unit(v):
    """
    Transform v into a unity-norm vector.

    PARAMETERS
    v: (x, y) ((float, float))
        2D components of vector v.

    RETURN
    unit : (x, y) ((float, float))
        2D components of unit vector.
    """
    x,y = v
    mag = norm(v)
    unit = (x/mag, y/mag)
    return unit

def distance(p0,p1):
    """
    Compute the 2D Euclidean distance between points p0 and p1.

    PARAMETERS
    p0, p1: (x, y) ((float, float))
        2D coordinates of points p0,p1.

    RETURN
    distance : float
        2D Euclidean norm of vector p0->p1.
    """
    distance = norm(vector(p0,p1))
    return distance

def scale(v,sc):
    """
    Scale vector v by scaling factor sc.

    PARAMETERS
    v: (x, y) ((float, float))
        2D components of vector v.
    sc: float
        Scaling factor.

    RETURN
    scaled_vector : (x, y) ((float, float))
        2D components of scaled vector v*sc.
    """
    x,y = v
    scaled_vector = (x * sc, y * sc)
    return scaled_vector

def add(v,w):
    """
    Add vectors v and w.

    PARAMETERS
    v, w: (x, y) ((float, float))
        2D components of vectors v, w.

    RETURN
    addition : (x, y) ((float, float))
        2D components of vector v+w.
    """
    x,y = v
    X,Y = w
    addition = (x+X, y+Y)
    return addition
