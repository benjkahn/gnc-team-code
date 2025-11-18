import pytest as test
import numpy as np
from src.models import quat_utils

    
def test_quaternion_multiply_random():
    q1 = np.array([2,1,0,0])
    q2 = np.array([3,0,4,0])

    actual_output = quat_utils.quaternion_multiplication(q1,q2)
    expected_output = np.array([6,3,8,4])

    assert actual_output.all() == expected_output.all()

