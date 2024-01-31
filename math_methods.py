'''
'''

import numpy as np

class Methods:


    @classmethod
    def moving_average(self, data:np.ndarray, window:int) ->np.ndarray:
        '''Applies moving average to data, reducing its resolution.'''
        bx_filter = np.ones(window)/window
        averaged = np.convolve(data, bx_filter, 'same')
        return averaged
    
    @classmethod
    def _isfloat(self, string) ->bool:
        '''Returns "True" if string is interpretable as float.'''
        try:
            float(string)
            return True
        except ValueError:
            return False
        
    @classmethod
    def read_txt(self, filepath) ->list:
        '''Reads data from .txt file.'''
        output = []
        with open(filepath, 'r') as r:
            lines = r.readlines()
            for line in lines:
                line = line.strip()
                if self._isfloat(line):
                    line = float(line)
                output.append(line)
        return output
    
    @classmethod
    def read_wavelength_txt(self, filepath) ->np.ndarray:
        '''Reads wavelengths from txt file, returns an array.'''
        list_output = self.read_txt(filepath)
        array_output = np.array(list_output)
        return array_output
    
    