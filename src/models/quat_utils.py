"""
Author: Matthew Copeland
Email: matthew.copeland@sjsu.edu
Created: 11-9-2025
Last Modified: 11-9-2025

Quaternion mathematics utilities.
All quaternion operations for attitude representation and manipulation

Quaternion convention: q = [q0, q1, q2, q3] = [scalar, vectorX, vectorY, vectorZ]
Represents rotation from inertial frame to body frame
"""

import numpy as np


def quaternion_multiplication(q1: np.ndarray, q2: np.ndarray):
    """
    Multiply two quaternions together

    Args:
        q1: First quaternion [q0, q1, q2, q3]
        q2: Second quaternion [q0, q1, q2, q3]

    Returns:
        Product quaternion [q0, q1, q2, q3]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 + x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray):
    """
    Compute quaternion conjugate or the inverse unit quaternion

    Args:
        q: Quaternion [q0, q1, q2, q3]

    Returns:
        Conjugate [q0, -q1, -q2, -q3]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_norm(q: np.ndarray):
    """
    Normalize quaternion to unit length

    Args:
        q: Quaternion [q0, q1, q2, q3]

    Returns:
        Normalized Quaternion
    """
    norm = np.linalg.norm(q)
    if norm < 1e-5:
        # Return identity quaternion if near zero
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q/norm


def quaternion_error(q_current: np.ndarray, q_target: np.ndarray,
                     degrees: bool = False, print_axis_angle: bool = False):
    """
    Compute attitude error from current to target quaternion and return
    error vector for control (small angle approximation)

    Args:
        q_current: Current atttitude [q0, q1, q2, q3]
        q_target: Target attitude [q0, q1, q2, q3]
        degrees: Default False - Set true if you want quaternion error angle to return in degrees
        print_axis_angle: Default False - Set true if you want to print axis-angle values

    Returns:
        Error vector [ex, ey, ez] in radians and body frame
        """

    q_conj = quaternion_conjugate(q_current)
    q_error = quaternion_multiplication(q_target, q_conj)

    if q_error[0] < 0:
        q_error = -q_error

    angle = 2*np.acos(np.clip(q_error[0], -1.0, 1.0))
    sin_half_angle = np.sin(angle/2)
    if sin_half_angle < 1e-8:
        return np.zeros(3)
    else:
        axis = q_error[1:] / sin_half_angle

    if print_axis_angle:
        print(f'AXIS: {axis}')
        print(f'ANGLE: {angle}')

    error = axis * angle

    if degrees:
        # Return error as degress instead of radians if true
        error = error*180/np.pi

    return error


def quat_from_euler(roll: float, pitch: float, yaw: float, degrees: bool = False):
    """
    Convert Euler angles (3-2-1 sequence) to quaternion
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Args:
        roll: Roll angle (rad) - rotation about X
        pitch: Pitch angle (rad) - rotation about Y
        yaw: Yaw angle (rad) - rotation about Z
        degrees: Default False - Set true if roll, pitch, and yaw are given in degrees

    Returns:
        Quaternion [q0, q1, q2, q3]
    """
    if degrees:
        roll = roll*np.pi/180
        pitch = pitch*np.pi/180
        yaw = yaw*np.pi/180

    cr = np.cos(roll/2)
    sr = np.sin(roll/2)
    cp = np.cos(pitch/2)
    sp = np.sin(pitch/2)
    cy = np.cos(yaw/2)
    sy = np.sin(yaw/2)

    q0 = cr*cp*cy + sr*sp*sy
    q1 = sr*cp*cy - cr*sp*sy
    q2 = cr*sp*cy + sr*cp*sy
    q3 = cr*cp*sy - sr*sp*cy

    return np.array([q0, q1, q2, q3])


def quat_to_euler(q: np.ndarray, degrees: bool = False):
    """
    Convert quaternion to Euler angles (3-2-1 sequence)
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Args:
        q: Quaternion [q0, q1, q2, q3]
        degrees: Default False - Set true if you want euler angles to return in degrees

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    q0, q1, q2, q3 = q

    sinr_cosp = 2 * (q0*q1 + q2*q3)
    cosr_cosp = 1 - 2 * (q1**2 + q2**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (q0*q2 - q3*q1)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2**2 + q3**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        roll = roll*180/np.pi
        pitch = pitch*180/np.pi
        yaw = yaw*180/np.pi

    return roll, pitch, yaw


# ----------------------------------------- EXAMPLE SCRIPTS TO TEST FUNCTIONS -----------------------------------------
if __name__ == "__main__":
    # Example 1: Create quaternion from Euler angles
    print("\n1. Euler to Quaternion:")
    roll, pitch, yaw = [20, 0, 0]
    q = quat_from_euler(roll, pitch, yaw, degrees=True)
    print(f"   Quaternion: {q}")

    # Example 2: Quaternion to Euler angles
    print("\n2. Quaternion to Euler:")
    r, p, y = quat_to_euler(q, degrees=True)
    print(f'   Euler Angles: [{r:.1f}, {p:.1f}, {y:.1f}] [deg]')

    # Example 3: Quaternion error
    print("\n3. Attitude Error Calculation:")
    q_current = quat_from_euler(roll, pitch, yaw, degrees=True)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])  # Identity (no rotation)
    roll_target, pitch_target, yaw_target = quat_to_euler(q_target)
    error = quaternion_error(q_current, q_target, degrees=True)
    print(f"   Current: [{roll} roll, {pitch} pitch, {yaw} yaw]")
    print(f"   Target: [{roll_target} roll, {pitch_target} pitch, {yaw_target} yaw]")
    print(f"   Error Angle: [{error[0]} roll, {error[1]} pitch, {error[2]} yaw]")