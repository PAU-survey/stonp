import unittest
from stonp.plotting import single_plotter, rc_parameters


class testRcParameters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n"+str(cls.__name__))

    def test_no_args(self):
        # Check that no errors are thrown
        try:
            stonp.stacker.rc_parameters()
        except Exception as e:
            assert False, f"Exception raised: {e}"

    def test_bad_arg(self):
        with self.assertRaises(TypeError):
            stonp.stacker.rc_parameters(1)
            stonp.stacker.rc_parameters(1.1)
            stonp.stacker.rc_parameters("a")
            stonp.stacker.rc_parameters(object())

    def test_result(self):
        self.assertIsInstance(stonp.stacker.rc_parameters(), dict)
