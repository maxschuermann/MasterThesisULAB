'''
20231122
'''
import logging
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
plt.style.use('_mpl-gallery')
from scipy.signal import find_peaks, medfilt
import pytz
from datetime import datetime
timezone = 'Europe/Zurich'


class AnalysisMethods:

    def __init__(self) -> None:
        pass
    
    @classmethod
    def get_difference(self, spectrum_1, spectrum_2):
        '''Returns spectrum1 - spectrum2.
                Idea: Spectrum(reactands) - Spectrum(products)
            "0" at wavelengths where spectra are equal
            "positive" when the intensity in the product spectrum is lower
            "negative" in areas where the intensity in the product spectrum is higher (for example new compounds)'''
        diff = np.copy(spectrum_1.spectrum) - np.copy(spectrum_2.spectrum)
        return diff
    
    @classmethod
    def get_sum_differences(self, spectra_list:list):
        '''Checks if spectral data is continuos or if output changes over time.
        Calculates sum of the difference of spectra, has to be corrected with noise level!
        Spectra given in this function have to be recorded with the same parameters (integration time).'''
        #check if same parameters?
        differences = []
        for i in range(len(spectra_list)-1):
            diff = self.get_difference(spectrum_1=spectra_list[i], spectrum_2=spectra_list[i+1])
            sum_diff = np.sum(diff)
            differences.append(sum_diff)
        return differences
    
    @classmethod
    def get_minima(self, spectral_data:np.ndarray):
        '''Wrapper for scipy.signal.find_peaks function.'''
        spectral_data *= -1
        minima, peakdata = find_peaks(spectral_data, prominence=0.1)
        return minima, peakdata
    
    @classmethod
    def get_maxima(self, spectral_data:np.ndarray):
        '''Find maxima in spectral data with scipy.signal.find_peaks'''
        maxima, peakdata = find_peaks(spectral_data, prominence=0.1)
        return maxima, peakdata

    
class Visualization:
    def __init__(self) -> None:
        self.wavelengths = None
        pass

    def build_wavelengths(self, wavelengths)->None:
        '''Saves wavelength table to instance of Visualization class.'''
        self.wavelengths = wavelengths

    def build_wavelengths_from_txt(self, filepath)->None:
        '''Saves wavelength table to instance from txt file.'''
        wlngths = Methods.read_wavelength_txt(filepath)
        self.build_wavelengths(wlngths)

    def plot_spectrum(self, spectr, label= 'integration_time', title=None, legend:bool = True):
        '''Single plot of spectrum/spectra in 2D.'''
        #check if more than one spectrum
        fig, ax = plt.subplots(figsize = (8,4))
        if type(spectr) == list:
            for item in spectr:
                ax.plot(self.wavelengths, item.spectrum, label = item.__dict__[label])
        elif type(spectr) != list:
            ax.plot(self.wavelengths, spectr.spectrum, label = spectr.__dict__[label])
        if title is not None:
            plt.title(title)
        if legend:
            plt.legend()
        plt.xlabel('$\lambda$ [nm]')
        plt.ylabel('counts')
        plt.show()

    def plot_spectra_over_time3D(self, spectra:list, times:list)->None:
        '''Plot stuff 3d'''
        x_wavelengths = np.copy(self.wavelengths)
        y_times = np.array(times)
        x_wavelengths, y_times = np.meshgrid(x_wavelengths, y_times)
        z_spectra = []
        for spectrum in spectra:
            if type(spectrum) != np.ndarray:
                spec = spectrum.spectrum
            else:
                spec =spectrum
            z_spectra.append(spec)
        z_spectra = np.array(z_spectra)
        fig, ax = plt.subplots(subplot_kw = {"projection":"3d"}, figsize = (8,8))
        ax.plot_surface(x_wavelengths, y_times, z_spectra, cmap = 'coolwarm', vmin = z_spectra.min()*2)
        ax.set_zlim(np.min(z_spectra), np.max(z_spectra))
        ax.set_xlabel('$\lambda$ [nm]')
        ax.set_ylabel('Time since start [s]')
        plt.show()

    def plot_spectra_over_time_contour(self, spectra:list, times:list, nlevels = 100, wavelengths:np.ndarray = None)->None:
        '''Plot contour'''
        if wavelengths is None:
            x_wavelengths = np.copy(self.wavelengths)
        elif wavelengths is not None:
            x_wavelengths = wavelengths
        y_times = np.array(times)
        x_wavelengths, y_times = np.meshgrid(x_wavelengths, y_times)
        z_spectra = []
        for spectrum in spectra:
            if type(spectrum) !=np.ndarray:
                z_spectra.append(spectrum.spectrum)
            else:
                z_spectra.append(spectrum)
        z_spectra = np.array(z_spectra)
        lvls = np.linspace(np.min(z_spectra), np.max(z_spectra), nlevels)
        fig, ax = plt.subplots(figsize = (8,8))
        ax.contourf(x_wavelengths,y_times,z_spectra, levels = lvls, cmap = 'coolwarm')
        ax.set_xlabel('$\lambda$ [nm]')
        ax.set_ylabel('Time since start [s]')
        plt.show()

    def ths_plot_absorbance(self, spectrum, reference, title:str=None, filepath:str=None, ylim:tuple=None):
        absbance = spectrum.return_absorbance(reference)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.set_xlim(np.min(self.wavelengths), np.max(self.wavelengths))
        if ylim is None:
            ax.set_ylim(np.min(absbance), np.max(absbance))
        else: 
            ax.set_ylim(ylim)
        ax.set_xlabel('$\lambda$ [nm]')
        ax.set_ylabel('Absorbance [Abs]')
        if title is not None:
            plt.title(title)
        ax.plot(self.wavelengths, absbance, color = 'b')
        plt.tight_layout()
        plt.show()
        if filepath is not None:
            fig.savefig(filepath, format = 'svg')
    
    def ths_plot_generic(self, spectrum, title:str=None, filepath:str= None, xlabel:str='$\lambda$ [nm]', ylabel:str='',  color:str='b', ylim:tuple=None, deviation:np.ndarray=None):
        '''Plot spectral data (y) vs wavelengths (x).'''
        plt.figure(figsize=(8,4))
        plt.xlim(np.min(self.wavelengths), np.max(self.wavelengths))
        if ylim is None:
            plt.ylim(np.min(spectrum), np.max(spectrum)*1.2)
        else:
            plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        plt.plot(self.wavelengths, spectrum, color=color)
        if deviation is not None:
            plt.fill_between(self.wavelengths, spectrum-deviation, spectrum+deviation, color=color, alpha = 0.3)
        if filepath is not None:
            plt.savefig(filepath, format= 'svg')
        plt.show()

    def ths_plot_tm_spectrum(self, spectrum, title:str=None, filepath:str=None):
        ydata = spectrum.spectrum
        fig, ax = plt.subplots(figsize=(16,8))
        ax.set_xlim(np.min(self.wavelengths), np.max(self.wavelengths))
        ax.set_ylim(0, np.max(ydata)*1.1)
        ax.set_xlabel('$\lambda$ [nm]')
        ax.set_ylabel('Transmission [a.u.]')
        if title is not None:
            plt.title(title)
        ax.plot(self.wavelengths, ydata, color = 'b')
        plt.show()
        if filepath is not None:
            fig.savefig(filepath, format='svg')


class Methods:
    @classmethod
    def moving_average(self, data:np.ndarray, window:int) ->np.ndarray:
        '''Applies moving average to data, reducing its resolution.'''
        bx_filter = np.ones(window)/window
        averaged = np.convolve(data, bx_filter, 'same')
        return averaged
    # @classmethod
    # def median_filter(self, data:np.ndarray, window_size)
    #     medfil =medfilt(spectrum, window)
    # 
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
    
    @classmethod
    def _find_closest_value_index(self, array:np.ndarray, value:float)->int:
        '''Returns index of closest value in array to value.'''
        arry = np.copy(array)
        idx = (np.abs(arry-value)).argmin()
        return idx
    
    @classmethod
    def _input_ask(self, msg:str)->bool:
        print(msg, '(Y?)')
        ipt = input()
        if 'Y' in ipt:
            return True
        else:
            return False
        
    @classmethod
    def convert_utc_to_human(self, utc):
        utc = datetime.utcfromtimestamp(utc)
        local_time = utc.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(timezone))
        return local_time