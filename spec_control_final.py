import logging

import avaspec
from time import sleep, time
import numpy as np

from spectral_data_fmts import BaseSpectralData
import default_parameters

logger = logging.Logger(__name__)

class Spectrometer:
    '''Control for Avantes Spectrometers connected via USB.
    Gets default measurement settings from "default_parameters.py". 
    To change settings other than integration time and number of measurements, 
    change defaults.'''
    def __init__(self, start_ctrl:bool = True) -> None:
        self._output_pixels:int = 2048
        self.wavelengths = None

        self.handle = None
        self._spec_id:avaspec.AvsIdentityType = None
        
        self._meas_default_params= default_parameters.MeasurementDefaults()
        self._return_default_measconfig = self._meas_default_params._return_defaults()
        self._last_meas_params:avaspec.MeasConfigType = None

        if start_ctrl:
            self.start_ctrl()

    def start_ctrl(self, USB:bool = True, Ethernet:bool = False)->None:
        if USB and not Ethernet:
            sel_port = 0
        elif Ethernet and not USB:
            sel_port = 256
        elif Ethernet and USB:
            sel_port = -1
        init = avaspec.AVS_Init(sel_port)
        logger.debug(f'AVS_Init -> {init} (devices).')
        
    def activate_spectrometer(self, dlist:tuple = 0, spectrometer:int= 0, load_lambda:bool = True) ->None:
        '''Sets handle and ID for connected spectrometer.'''
        if not dlist:
            dlist = self._get_device_list()
        if len(dlist) == 0:
            logger.error('activate_spectrometer: Length device list is 0.')
        self._spec_id:avaspec.AvsIdentityType = dlist[spectrometer]
        self.handle = avaspec.AVS_Activate(self._spec_id)
        if type(self._spec_id) == avaspec.AvsIdentityType:
            logger.debug(f'activate_spectrometer: {str(self._spec_id.SerialNumber.decode("utf-8"))}, handle = {self.handle}')
        else:
            logger.error(f'type spec_id is {type(self._spec_id)}')
        if load_lambda:
            wavelengths = avaspec.AVS_GetLambda(self.handle)
            self.wavelengths = np.array(wavelengths)[:self._output_pixels]

    def get_wavelengths(self)->np.ndarray:
        '''Calls AVS_GetLambda and stores it in instance.'''
        wlngths = avaspec.AVS_GetLambda(self.handle)
        wlngths = np.array(wlngths)[:self._output_pixels]
        self.wavelengths = wlngths
        return wlngths

    def close_interface(self)->None:
        try:
            done = avaspec.AVS_Done()
            if done == 0:
                logger.info('AVS_Done: 0 (Successful disconnected).')
        except TimeoutError as err:
            raise err(f'AVS_Done: {done}.') from None
        
    def _get_device_list(self, spectrometers:int = 1) ->tuple:
        '''Returns tuple containing avaspec.AvsIdentityType object for each spectrometer found.'''
        dlist = avaspec.AVS_GetList(spectrometers)
        logger.debug(f'AVS_GetList -> {dlist}.')
        return dlist
    
    def _prepare_measure(self, measconfig:avaspec.MeasConfigType) ->int:
        prepare = avaspec.AVS_PrepareMeasure(self.handle, measconf=measconfig)
        if prepare !=0:
            logger.error(f'AVS_PrepareMeasure: {prepare}.')
            raise ValueError('Measurement configuration invalid.')
        return prepare

    def _measure_wrap(self, nmeas:int) ->int:
        measure = avaspec.AVS_Measure(self.handle, windowhandle=0, nummeas=nmeas)
        if measure !=0:
            logger.error(f'AVS_Measure: {measure}')
            raise ValueError('Measurement could not be completed.')
        return measure
    
    def _check_for_data(self) ->bool:
        check = avaspec.AVS_PollScan(self.handle)
        logger.debug(f'AVS_PollScan: {check}')
        return check
    
    def _get_spectral_data(self):
        spectime, spectrum = avaspec.AVS_GetScopeData(self.handle)
        sat_array = avaspec.AVS_GetSaturatedPixels(self.handle)
        spec_fmt = np.array(spectrum)[:self._output_pixels]
        sat_fmt = np.array(sat_array)[:self._output_pixels]
        logger.info(f'AVS_GetScopeData: spectrum {spectime}')
        return spectime, spec_fmt, sat_fmt
    
    def measurement(self, inttime:float, nmeas:int, timeout:float = 20, stepped_inttime:bool = False) ->list[BaseSpectralData]:
        '''Full measurement wrapper. Will block the terminal while measurement is ongoing.
            Changeable parameters are number of measurements and integration time, 
            other parameters are to be changed in the default settings.'''
        output = []
        if stepped_inttime:
            inttime = self._meas_default_params.get_inttime_stepvalue(inttime)
        measconfig = self._meas_default_params._set_integration_time(inttime)
        self._last_meas_params = measconfig
        prepare = self._prepare_measure(measconfig=measconfig)
        if prepare == 0:
            measure = self._measure_wrap(nmeas=nmeas)
            if measure == 0:
                for measurement in range(nmeas):
                    start_time = time()
                    check = self._check_for_data()
                    while check == 0:
                        sleep(0.001)
                        check = self._check_for_data()
                        timeout_clock = (time()-start_time) - nmeas*inttime/1000
                        if timeout_clock > timeout:
                            raise TimeoutError('Requesting measurement data took to long.')
                    spectime, spec_fmt, sat_fmt = self._get_spectral_data()
                    data = BaseSpectralData(utc_time=time(), spec_time=spectime, integration_time=inttime, spec_averages=measconfig.m_NrAverages, spectrum=spec_fmt, saturation_spectrum=sat_fmt)
                    output.append(data)
        return output
    
    def save_wavelengths(self, filepath = 'wavelengths.txt') -> None:
        '''Saves wavelength table from spectrometer to .txt file.'''
        if self.wavelengths:
            with open(filepath, 'w') as w:
                for i in self.wavelengths:
                    w.write(f'{i}\n')
            logger.info(f'Saved wavelengths to {filepath}.')
        else:
            logger.error('Wavelengths not loaded, call AVS_GetLambda first.')


def main():
    'test'
    spec = Spectrometer()
    try:
        spec.activate_spectrometer()
        print(spec._spec_id)
    except Exception as err:
        raise err from None
    finally:
        spec.close_interface()

if __name__ == '__main__':
    from logging.handlers import TimedRotatingFileHandler
    logger.setLevel(logging.DEBUG)
    logfile_handler = TimedRotatingFileHandler('spec_ctrl_test.log',
                                       'midnight',
                                       backupCount=7)
    logfile_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    )
    logfile_handler.setLevel(logging.DEBUG)
    logger.addHandler(logfile_handler)

    main()