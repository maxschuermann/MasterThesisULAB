'''

'''
import logging
import numpy as np
from time import time, sleep

import thorlabs_filterwheel as fw_ctrl
import spec_control_final as spec_ctrl
import lightsource_fetbox as ls_ctrl

from spectral_data_fmts import MetaSpectralData

logger = logging.Logger(__name__)

class OpticsControl:
    '''Control for the optical devices:
        Avantes spectrometer, BDS130A lightsource FETbox controller, Thorlabs FW102C filter wheel.'''
    def __init__(self) -> None:
        self.lightsource = ls_ctrl.LightSource()
        self.spectrometer = spec_ctrl.Spectrometer(start_ctrl=False)
        self.filterwheel = None
        
        self.uv_start_time = 0
        self.vis_start_time = 0

        self.spec_connect = False
        self.fw_connect = False
        self.ls_connect = False

        self._shutter_delay = 0.1
        self.last_inttime = None
        self.shutter_status = None

    def _connect_spectometer(self):
        '''Establishes connection with Avantes spectrometer.'''
        try:
            self.spectrometer.start_ctrl()
            self.spectrometer.activate_spectrometer()
            self.spec_connect = True
            logger.info('Established connection to spectrometer.')
        except Exception as exc:
            logger.error(f'Spectrometer connection attempt failed: {exc}.')

    def _connect_filterwheel(self):
        '''Establishes connection to FW102C filter wheel.'''
        try:
            self.filterwheel = fw_ctrl.FW102C('TR03014244')
            self.filterwheel.connect()
            self.fw_connect = True
            logger.info('Connected to FW102C.')
        except Exception as exc:
            logger.error(f'FW102C connection attempt failed: {exc}.')

    def _connect_lightsource(self, attempts:int = 1, retry_delay:float = 1):
        '''Tries to establish connection to FETbox light source controller.
            attempts: number of attempts,
            retry_delay: time between attempts [s].'''
        self.lightsource.connect_retry(attempts=attempts, retry_delay=retry_delay)
        self.lightsource.set_ttl_control(True)
        self.ls_connect = True
        logger.info('Connected to lighsource controller.')

    def connect(self, spectrometer:bool = False, filterwheel:bool = False, lightsource:bool = False, ls_attempts = 3, ls_retry_delay = 1) ->None:
        '''Establishes communication with optical devices.'''
        if lightsource:
            self._connect_lightsource(attempts=ls_attempts, retry_delay=ls_retry_delay)
        if spectrometer:
            self._connect_spectometer()
        if filterwheel:
            self._connect_filterwheel()

    def disconnect(self, spectrometer:bool = False, filterwheel:bool = False, lightsource:bool = False) ->None:
        '''Disconnects from optical devices.'''
        if spectrometer:
            self.spectrometer.close_interface()
            self.spec_connect = False
            logger.info('Disconnected from spectrometer.')
        if filterwheel:
            self.filterwheel.close()
            self.filterwheel = None
            self.fw_connect = False
            logger.info('Disconnected from FW102C.')
        if lightsource:
            self.lightsource.close()
            self.ls_connect = False
            logger.info('Disconnected from lightsource controller.')

    def lightsource_control(self, uv:bool, vis:bool) -> None:
        '''Turns UV/Vis lamps on/off.'''
        if self.ls_connect:
            #check if already in desired state
            lightsource_status = self.lightsource_status()
            if uv != lightsource_status['uv']:
                self.lightsource.set_uv(uv)
                if uv:
                    self.uv_start_time = time()
                elif not uv:
                    self.uv_start_time = False
            if vis != lightsource_status['vis']:
                self.lightsource.set_vis(vis)
                if vis:
                    self.vis_start_time = time()
                elif not vis:
                    self.vis_start_time = False
            logger.info(f'Set lightsource to: UV = {uv}, Vis = {vis}.')
        else:
            logger.error(f'Lighsource connection is {self.ls_connect}. Connect first.')

    def lightsource_set_shutter(self, shutter:bool)->None:
        '''True: open, False: closed'''
        self.lightsource.set_shutter(shutter)
        self.shutter_status = shutter
        logger.info(f'Set shutter to {shutter}.')

    def lightsource_status(self) -> dict: #mapping/typing not complete
        '''Displays the status of UV/Vis lamps.'''
        if self.ls_connect:
            status = {
                'uv' : self.lightsource.get_uv(),
                'uv_run_time' : self.lightsource.get_uv() * (time()-self.uv_start_time),
                'vis' : self.lightsource.get_vis(),
                'vis_run_time' : self.lightsource.get_vis() * (time()-self.vis_start_time)
            }
            return status
        else:
            logger.error(f'Lightsource connection is {self.ls_connect}. Connect first.')

    def capture_spectrum(self, inttime:float, nmeas:int = 1, filter_pos:int = 0, shutter:bool = True, close_shutter:bool = True, limited_inttimes:bool = False)->list[MetaSpectralData]:
        '''Function to capture spectra. 
            Params:
                inttime: integration time [ms]
                nmeas: number of measurements
                filter_pos: when set to 0 filter wheel is ignored
                    otherwise set filterwheel to position (1-6)
                shutter: open the shutter before measurement
                close_shutter: close the shutter after measurement
                limited_inttimes: uses discrete integration times as implemented in spec_ctrl
            '''
        #get rid of the last feature maybe
        #maybe add routine to check lightsource and not take entry from memory?
        #maybe add function to auto average when nmeas >1?
        if filter_pos:
            if not self.fw_connect:
                logger.error(f'Filterwheel connection is {self.fw_connect}')
            self.filterwheel.set_position(filter_pos)
        # ls_status = self.lightsource_status()
        # dummy ls status because fetbox overloaded with queries:
        uv = False
        vis = False
        if self.uv_start_time !=0:
            uv = time()-self.uv_start_time
        if self.vis_start_time !=0:
            vis = time()-self.vis_start_time
        ls_status = {'uv':uv, 'uv_run_time':0, 'vis':vis, 'vis_run_time':0}
        #when cleaning this code idea: just one entry for uv and vis and it is either the time or false
        #get rid of uv_run_time and vis_run_time
        if self.shutter_status and not shutter:
            self.lightsource_set_shutter(False)
        if shutter and not self.shutter_status:
            self.lightsource_set_shutter(True)
            sleep(self._shutter_delay)
        spectra = self.spectrometer.measurement(inttime=inttime, nmeas=nmeas, stepped_inttime=limited_inttimes)
        if close_shutter and self.shutter_status:
            self.lightsource_set_shutter(False)
            # sleep(self._shutter_delay)
        self.last_inttime = inttime
        output = []
        for i in spectra:
            data = MetaSpectralData(base_data=i, shutter=self.shutter_status, uv_status=ls_status['uv'], uv_run_time=ls_status['uv_run_time'], vis_status=ls_status['vis'], vis_run_time=ls_status['vis_run_time'], filter_pos=filter_pos)
            output.append(data)
        return output
    
def main():
    'Test'
    octrl = OpticsControl()
    try:
        octrl.connect(spectrometer=True, lightsource=True)
        print(octrl.lightsource_status())
        print(octrl.capture_spectrum(1300, 1, 0, 1, False))
        spectra = octrl.capture_spectrum(5, 1, 0, 0, True)
        for i in spectra:
            print(f'{i.id_string}: inttime {i.integration_time}')
            
    except Exception as err:
        logger.exception(err)
        raise err from None
    finally:
        octrl.disconnect(spectrometer=True, lightsource=True)


if __name__ == '__main__':
    from logging.handlers import TimedRotatingFileHandler
    logger.setLevel(logging.DEBUG)
    logfile_handler = TimedRotatingFileHandler('optics_control_final_test.log',
                                       'midnight',
                                       backupCount=7)
    logfile_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    )
    logfile_handler.setLevel(logging.DEBUG)
    logger.addHandler(logfile_handler)

    main()