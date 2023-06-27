import numpy as np

def trapezoidal_profile(distance, max_speed, acceleration, deceleration=None, nr_samples=1000, return_segment_distances=False):
    """
    Generate a trapezoidal velocity profile with specified distance, max speed, and acceleration.

    Args:
        distance (float): Total distance to travel.
        max_speed (float): Maximum velocity allowed.
        acceleration (float): Acceleration and deceleration to use.
        deceleration (float): Deceleration to use. Defaults to None, in which case it is assumed to be equal to acceleration.
        return_segment_distances (bool, optional): If return segment distances also. Defaults to False.

    Returns:
        Tuple containing numpy arrays of time, velocity, and position.
    """
    # If no deceleration is specified, assume it's the same as acceleration
    if deceleration is None:
        deceleration = acceleration
    
    # Calculate the time needed to reach max speed during acceleration
    t_ramp_up = max_speed / acceleration
    
    # Calculate the distance covered during acceleration
    d_ramp_up = 0.5 * acceleration * t_ramp_up ** 2
    
    # Calculate the time needed to reach 0 speed during deceleration
    t_ramp_down = max_speed / deceleration
    
    # Calculate the distance covered during deceleration
    d_ramp_down = 0.5 * deceleration * t_ramp_down ** 2
    
    # Calculate the time spent traveling at max speed
    d_const_max = distance - (d_ramp_up + d_ramp_down)
    t_const = d_const_max / max_speed

    # If the distance is too short, adjust the time spent at max speed
    if t_const < 0:
        acceleration = min(acceleration, deceleration)
        deceleration = acceleration
        max_speed = np.sqrt(distance * acceleration)
        t_const = 0
        d_const_max = 0
        t_ramp_up = max_speed / acceleration
        t_ramp_down = max_speed / deceleration
        d_ramp_up = 0.5 * acceleration * t_ramp_up ** 2
        d_ramp_down = 0.5 * deceleration * t_ramp_down ** 2

    # Calculate the total time
    t_total = t_ramp_up + t_const + t_ramp_down

    
    # Generate time, velocity, and position arrays
    t = np.linspace(0, t_total, num=nr_samples)
    v = np.piecewise(t, [t <= t_ramp_up, (t_ramp_up < t) & (t <= t_ramp_up + t_const), t > t_ramp_up + t_const], 
                      [lambda x: acceleration * x, max_speed, lambda x: max_speed - deceleration * (x - t_ramp_up - t_const)])
    p = np.piecewise(t, [t <= t_ramp_up, (t_ramp_up < t) & (t <= t_ramp_up + t_const), t > t_ramp_up + t_const], 
                      [lambda x: 0.5 * acceleration * x ** 2, lambda x: d_ramp_up + max_speed * (x - t_ramp_up), 
                       lambda x: distance - 0.5 * deceleration * (t_total - x) ** 2])
    
    if return_segment_distances:
        return t, v, p, d_ramp_up, d_const_max, d_ramp_down
    return t, v, p
