#this is a comment  new comment

import unittest

class TestUM(unittest.TestCase):
  
    def setUp(self):
        pass
	       
    def test_numbers(self):
        self.assertEqual( 12, 12)
			    
    def test_strings(self):
        self.assertEqual( 'aaa1' , 'aaa')
					 
if __name__ == '__main__':
    unittest.main()
