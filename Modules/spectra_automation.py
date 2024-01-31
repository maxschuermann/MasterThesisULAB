'''
20231121
'''
import logging
import optics_control_final as opt_ctrl
from spectral_data_fmts import MetaSpectralData, DarkCorrectedSpectralData
class AutoSpectra:
    '''MOSTLY DEPRECATED. measure_until_saturated used for integration time/intensity dependence, adjust integration time functions are maybe useful.'''
    def __init__(self, inttime_stepsize = 0.002, saturation_limit = 1) -> None:
        self.opt_ctrl = opt_ctrl.OpticsControl()
        self.inttime_stepsize = inttime_stepsize
        self.saturation_limit = 2048/100 * saturation_limit

    def connect(self, spectrometer:bool = True, filterwheel:bool = False, lightsource:bool = True, ls_attempts:int=3, ls_retry_delay:float=0.5):
        '''
        Wrapper for OpticsControl.connect()
        Establishes/Checks connection to listed devices.'''
        self.opt_ctrl.connect(spectrometer=spectrometer, filterwheel=filterwheel, lightsource=lightsource, ls_attempts=ls_attempts, ls_retry_delay=ls_retry_delay)

    def disconnect(self, spectrometer:bool = False, filterwheel:bool= False, lightsource:bool= False)->None:
        '''
        Wrapper for OpticsControl.disconnect().
        Disconnects from selected devices.'''
        self.opt_ctrl.disconnect(spectrometer=spectrometer, filterwheel=filterwheel, lightsource=lightsource)

    def measure_until_saturated(self, start_inttime:float = 0.002, inttime_stepsize:float = 0.002, saturation_limit_percent:float = 1, max_measurements:int = 50, measurements_per_step = 1, open_shutter:bool = True, close_shutter:bool = True, filter_pos:int = 0):
        '''Records spectra until the last spectum exceeds the saturation limits.
            Params:
                    start_inttime: integration time [ms] to start measurement
                    inttime_stepsize: size of steps the inttime is increased by [ms]
                    saturation_limit_percent: percentage of pixels that are allowed to be saturated
                    max_measurements: max number of measurements allowed
                    
            Returns:
                    List of integration times used  (last inttime is first inttime that exceeded saturation limit)
                    List of recorded spectra'''
        max_sat_pixels = 2048/100 * saturation_limit_percent
        meas_count = 0
        last_saturation = 0
        inttimes = []
        spectra_out= []
        while last_saturation < max_sat_pixels and meas_count < max_measurements and start_inttime <= 30000:
            spectrum = self.opt_ctrl.capture_spectrum(inttime=start_inttime, nmeas=measurements_per_step, shutter= open_shutter, close_shutter=close_shutter, filter_pos=filter_pos)
            inttimes.append(start_inttime)
            start_inttime += inttime_stepsize
            meas_count += 1
            last_saturation = spectrum[0].return_n_saturated()
            spectra_out.append(spectrum[0])
        return inttimes, spectra_out
    
    def measure_from_inttime_list(self, inttime_list:list, dark:bool = True):
        '''Measures a spectra with given integration times (for example dark references).'''
        spectra_out = []
        shutter = True
        if dark:
            shutter = False
            self.opt_ctrl.lightsource.set_shutter(False)
        for inttime in inttime_list:
            spectrum = self.opt_ctrl.capture_spectrum(inttime=inttime, shutter=shutter)
            spectra_out.append(spectrum[0])
        return spectra_out
    
    def _measure_wrap(self, inttime:float, **kwargs) ->list[MetaSpectralData]:
        '''Wrapper for optics_control's capture_spectrum_function.'''
        sig = self.opt_ctrl.capture_spectrum.__code__
        param_names = sig.co_varnames[:sig.co_argcount-1]
        defaults = self.opt_ctrl.capture_spectrum.__defaults__ or ()
        default_dict = dict(zip(param_names[-len(defaults):], defaults))
        default_dict.update(kwargs)
        output = self.opt_ctrl.capture_spectrum(inttime=inttime, **default_dict)
        return output

    def adjust_inttime_by_division(self, spectrum:MetaSpectralData, desired_value:float = 3500, index:int = 1408) ->float:
        #TRANSMISSION SPECTRA! Calculates new inttime to reach desired value at wavelength. 
        #Default is 3500 cts at 1000 nm, leads to ~10000 at the highest intensity.
        #Use any functions to find matching index for wavelengths
        '''Tries to find new inttime value for measurements from a spectrum.'''
        par = spectrum.spectrum[index] / desired_value
        return spectrum.integration_time /par
    
