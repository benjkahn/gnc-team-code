import unittest as test
import numpy as np
from models import quat_utils

class Test_Quaternion_Mult(test.TestCase):
    
    def test_quaternion_multiply_random():
        q1 = np.array(2,1,0,0)
        q2 = np.array(3,0,4,0)

        actual_output = quat_utils.quaternion_multiplication(q1,q2)
        expected_output = np.array(6,3,8,4)

        test.assertEqual(actual_output,expected_output)

