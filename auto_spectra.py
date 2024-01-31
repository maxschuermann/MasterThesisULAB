'''

'''
import logging
import optics_control_final as opt_ctrl
import default_parameters
from spectral_data_fmts import MetaSpectralData, DarkCorrectedSpectralData, save_json

DEF_desired_average = 10000
DEF_max_saturated = 0
DEF_change_speed_darker = 0.5
DEF_change_speed_brighter = 1

logger = logging.Logger(__name__)

class AutoSpectraAcquisition:
    def __init__(self, desired_average:float = DEF_desired_average, max_saturated:int = DEF_max_saturated, change_speed_darker:float = DEF_change_speed_darker, change_speed_brighter:float = DEF_change_speed_brighter) -> None:
        self.desired_average = desired_average
        self.max_saturated = max_saturated
        self.change_speed_brighter = change_speed_brighter
        self.change_speed_darker = change_speed_darker
        self.optics_ctrl = opt_ctrl.OpticsControl()
        self.default_params = default_parameters.MeasurementDefaults()
        self.inttimes_used = []
        self.init_filter_wheel = False
        self.init_spectrometer = True
        self.init_lightsource = True

    def connect_optical_devices(self) ->None:
        '''Connects optics control to devices.'''
        self.optics_ctrl.connect(spectrometer=self.init_spectrometer, filterwheel=self.init_filter_wheel, lightsource=self.init_lightsource)

    def disconnect_optical_devices(self) ->None:
        '''Disconnects from optical devices.'''
        self.optics_ctrl.disconnect(spectrometer=self.init_spectrometer, filterwheel=self.init_filter_wheel, lightsource=self.init_lightsource)

    def start_auto_measure(self, n_measurements:int, filepath:str = None, averages:int= 0, startparam:float = 0):
        '''Starts auto measurement routine (only bright spectra) and tries to find a feasible
        integration time so the spectra have a acceptable level of brightness.'''
        param = startparam
        output = []
        for n in range(n_measurements):
            inttime = self._find_new_inttime(param)
            logger.debug(f'Trying new inttime {inttime} ms.')
            spectra = self.optics_ctrl.capture_spectrum(inttime, nmeas=averages+1, filter_pos=self._find_new_filter_pos())
            self.inttimes_used.append(inttime)
            spectrum = spectra[0]
            if averages > 0:
                spectrum.average_with(spectra=spectra[1:])
            param = self._calculate_pparameter(spectrum)
            output.append(spectra)
            if filepath:
                for i in spectra:
                    save_json(i, filepath)
            logger.info(f'Recorded {averages+1} spectra, inttime {inttime} ms, saved to {filepath}.')
            logger.debug(f'New param {param}.')
        return output

    def measure_dark_references(self, averages:int = 0, filepath:str = None):
        '''Measures dark reference spectra for every integration time used in "start_auto_measure".'''
        ints = list(set(self.inttimes_used))
        output = []
        for i in ints:
            spectra = self.optics_ctrl.capture_spectrum(inttime=i, nmeas=averages+1, filter_pos=0)
            spectrum = spectra[0]
            if averages >0:
                spectrum.average_with(spectra=spectra[1:])
            output.append(spectra)
            if filepath:
                for i in spectra:
                    save_json(i, filepath)
            logger.info(f'Recorded {averages +1} dark spectra, inttime {i} ms, saved to {filepath}.')
        return output


    def _find_new_filter_pos(self, parameter:float = 0)->int:
        '''Not implemented yet.'''
        ret = 0
        return ret
    
    def _find_new_inttime(self, parameter:float = 0)->float:
        #parameter resembles ratio last spectrum average (e.g.)/desired spectrum average
        #should be 0 for a good spectrum, -1 for a spectrum that is 100% to bright, 1 for a spectrum that is 100% to dark
        #change speed limits the change ratio
        #should this be dependend on the last spectrum (new spectrum = x*last)
        #or should it be like: last spectrum is @x% of desired, new spectrum is 1/x * last
        #does it even make a difference?
        '''Calculates a new integration time from a given parameter.'''
        if parameter <= 0:
            change_speed = self.change_speed_brighter
        else: 
            change_speed = self.change_speed_darker

        if self.optics_ctrl.last_inttime:
            last_inttime = self.optics_ctrl.last_inttime
            last_inttime_steps = (last_inttime - self.default_params.spec_min_inttime)*1000 /self.default_params.stepsize_inttime
            last_inttime_steps = round(last_inttime_steps)
            new_inttime_steps = last_inttime_steps - parameter * change_speed * last_inttime_steps
            new_inttime = round(new_inttime_steps) * self.default_params.stepsize_inttime /1000 + self.default_params.spec_min_inttime
        else:
            #create first inttime: how?
            start_inttime = self.default_params.start_inttime
            #make it a value of n* steps
            new_inttime = self.default_params.get_inttime_stepvalue(start_inttime)
            logger.debug(f'First integration time generated from default values ({new_inttime}) ms.')
        if new_inttime < self.default_params.spec_min_inttime:
            logger.debug(f'Integration time ({new_inttime} ms) out of range. Set to {self.default_params.spec_min_inttime} ms.')
            new_inttime = self.default_params.get_inttime_stepvalue(self.default_params.spec_min_inttime)
        elif new_inttime > self.default_params.spec_max_inttime:
            logger.debug(f'Integration time ({new_inttime} ms) out of range. Set to {self.default_params.spec_max_inttime} ms.')
            new_inttime = self.default_params.get_inttime_stepvalue(self.default_params.spec_max_inttime)
        return new_inttime
    
    def _calculate_pparameter(self, spectrum:MetaSpectralData) ->float:
        '''DONT USE
        Calculates a parameter from a spectrum with regard to desired average/saturated values.'''
        #how to deal with spectra that have a very intense area and are otherwise low intensity?
        #saturation should be avoided thus should be treated first.
        #if saturation > max saturated the spectrum shouldnt be brighter 
        spec_average = spectrum.return_average_level()
        spec_n_saturated = spectrum.return_n_saturated()
        ratio_averages = spec_average/self.desired_average
        pparameter = ratio_averages -1
        # if pparameter < 0 and spec_n_saturated> self.max_saturated:
        #     pparameter = pparameter * -spec_n_saturated/2048 *10 #arbitrary 
        return pparameter
    
    
def main():
    'Test'
    auto = AutoSpectraAcquisition()
    try:
        auto.connect_optical_devices()
        auto.start_auto_measure(10, 'auto_test.json', averages=0)
    except Exception as err:
        logger.exception(err)
        raise err from None
    finally:
        auto.disconnect_optical_devices()

if __name__ == '__main__':
    from logging.handlers import TimedRotatingFileHandler
    logger.setLevel(logging.DEBUG)
    logfile_handler = TimedRotatingFileHandler('auto_spectra_test.log',
                                       'midnight',
                                       backupCount=7)
    logfile_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    )
    logfile_handler.setLevel(logging.DEBUG)
    logger.addHandler(logfile_handler)

    main()