"""
***********************************************************************
Assembly         : 
Author           : duguguang
Created          : 08-18-2023

Last Modified By : duguguang
Last Modified On : 08-18-2023
***********************************************************************
<copyright file="Utility.h" company="Nokov">
    Copyright (c) Nokov. All rights reserved.
</copyright>
<summary>Auxiliary function header file, provide external calculation of speed, acceleration, angle support</summary>
***********************************************************************
"""

import math
from collections import deque

class Point:
    """
    Represents a point in 3D space.
    """
    def __init__(self, x=0, y=0, z=0, name=""):
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def __str__(self):
        return f"name:{self.name}\tx:{self.x}\ty:{self.y}\tz:{self.z}\t\n"

    def __repr__(self):
        return str(self)

class Vel:
    """
    Represents velocity in 3D space.
    """
    def __init__(self, Vx=0, Vy=0, Vz=0, Vr=0):
        self.Vx = Vx
        self.Vy = Vy
        self.Vz = Vz
        self.Vr = Vr

    def __str__(self):
        return f"Vx:{self.Vx}\tVy:{self.Vy}\tVz:{self.Vz}\tVr:{self.Vr}\n"

    def __repr__(self):
        return str(self)

class Accel:
    """
    Represents acceleration in 3D space.
    """
    def __init__(self, Ax=0, Ay=0, Az=0, Ar=0):
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.Ar = Ar

    def __str__(self):
        return f"Ax:{self.Ax}\tAy:{self.Ay}\tAz:{self.Az}\tAr:{self.Ar}\n"

    def __repr__(self):
        return str(self)

class CalculateMethod:
    """
    Base class for calculating velocity and acceleration.
    """
    def __init__(self, FR, FF):
        self.Points = None
        self.FPS = FR
        self.FrameFactor = FF

    def get_frame_factor(self):
        """
        Get the frame factor.
        """
        return self.FrameFactor

    def get_frame_fps(self):
        """
        Get the frame FPS.
        """
        return self.FPS

    def try_to_calculate(self, points):
        """
        Try to calculate velocity or acceleration using the given points.
        """
        if points is None:
            return 1

        self.Points = points

        if self.FPS <= 0 or self.FPS >= 400:
            return 1

        return self.calculate()

    def calculate(self):
        """
        Calculate velocity or acceleration.
        """
        pass

class CalculateVelocity(CalculateMethod):
    """
    Class for calculating velocity using multiple frames.
    """
    def __init__(self, FR, FF):
        super().__init__(FR, FF)

    def calculate(self):
        """
        Calculate velocity using multiple frames.
        """
        TR = self.FrameFactor // 2

        vel = Vel()
        vel.Vx = self.FPS * (self.Points[TR * 2].x - self.Points[0].x) / (2 * TR)
        vel.Vy = self.FPS * (self.Points[TR * 2].y - self.Points[0].y) / (2 * TR)
        vel.Vz = self.FPS * (self.Points[TR * 2].z - self.Points[0].z) / (2 * TR)
        vel.Vr = math.sqrt(vel.Vx * vel.Vx + vel.Vy * vel.Vy + vel.Vz * vel.Vz)

        return vel

class CalculateVelocityByTwoFrame(CalculateMethod):
    """
    Class for calculating velocity using two frames.
    """
    def __init__(self, FR):
        super().__init__(FR, 2)

    def calculate(self):
        """
        Calculate velocity using two frames.
        """
        vel = Vel()
        vel.Vx = self.FPS * (self.Points[1].x - self.Points[0].x)
        vel.Vy = self.FPS * (self.Points[1].y - self.Points[0].y)
        vel.Vz = self.FPS * (self.Points[1].z - self.Points[0].z)
        vel.Vr = math.sqrt(vel.Vx * vel.Vx + vel.Vy * vel.Vy + vel.Vz * vel.Vz)

        return vel

class CalculateAcceleration(CalculateMethod):
    """
    Class for calculating acceleration using multiple frames.
    """
    def __init__(self, FR, FF):
        super().__init__(FR, FF)

    def calculate(self):
        """
        Calculate acceleration using multiple frames.
        """
        TR = self.FrameFactor // 2

        accel = Accel()
        accel.Ax = self.FPS * self.FPS * (self.Points[TR * 2].x - 2 * self.Points[TR].x + self.Points[0].x) / (TR * TR)
        accel.Ay = self.FPS * self.FPS * (self.Points[TR * 2].y - 2 * self.Points[TR].y + self.Points[0].y) / (TR * TR)
        accel.Az = self.FPS * self.FPS * (self.Points[TR * 2].z - 2 * self.Points[TR].z + self.Points[0].z) / (TR * TR)
        accel.Ar = math.sqrt(accel.Ax * accel.Ax + accel.Ay * accel.Ay + accel.Az * accel.Az)

        return accel

class SlideFrameArray:
    """
    Class for sliding frame array.
    """
    def __init__(self):
        self._list = deque()

    def clear(self):
        """
        Clear the frame array.
        """
        self._list.clear()

    def cache(self, point):
        """
        Cache a point in the frame array.
        """
        self._list.append(point)
        return len(self._list)

    def cache_xyz(self, x, y, z):
        """
        Cache a point with x, y, z coordinates in the frame array.
        """
        point = Point(x, y, z)
        return self.cache(point)

    def try_to_calculate(self, method):
        """
        Try to calculate velocity or acceleration using the given calculation method.
        """
        if len(self._list) < method.get_frame_factor():
            return None

        points = list(self._list)
        self._list.popleft()
        return method.try_to_calculate(points[0:method.get_frame_factor()])

def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between two lines formed by three points.
    """
    # Calculate the vectors formed by the two lines
    vector1 = (point1.x - point2.x, point1.y - point2.y, point1.z - point2.z)
    vector2 = (point3.x - point2.x, point3.y - point2.y, point3.z - point2.z)
    
    # Calculate the dot product of the two vectors
    dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2]
    
    # Calculate the magnitudes of the two vectors
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2 + vector1[2]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2 + vector2[2]**2)
    
    # Calculate the angle between the two lines using the dot product and magnitudes
    angle = math.acos(dot_product / (magnitude1 * magnitude2))
    
    # Convert the angle from radians to degrees
    angle_degrees = math.degrees(angle)
    
    return angle_degrees