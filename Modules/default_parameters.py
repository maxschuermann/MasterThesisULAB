import avaspec
from spectra_processing.Methods import read_wavelength_txt, _find_closest_value_index

DEF_n_inttime_steps =1000
DEF_start_inttime = 3000 #ms
DEF_spec_n_averages = 1
DEF_spec_min_inttime = 1 #ms
DEF_spec_max_inttime = 30000 #ms

class WavelengthTable:
    '''Class for wavelength array. 
    Not frequently used but could be easier than read txt in every module.
            filename: read txt file with filename to create entry.'''
    def __init__(self, filename:str=None) -> None:
        self.wavelengths = None
        if filename:
            try:
                wlngths = read_wavelength_txt(filename)
                self.wavelengths = wlngths
            except:
                pass
    def return_wavelengths(self):
        return self.wavelengths
    
    def return_wavelength_index(self, wavelength:float):
        idx = _find_closest_value_index(self.wavelengths, wavelength)
        return idx

class MeasurementDefaults:
    def __init__(self, n_inttime_steps:int = DEF_n_inttime_steps, start_inttime:float = DEF_start_inttime, spec_n_averages:int = DEF_spec_n_averages, range_inttime:tuple = (DEF_spec_min_inttime,DEF_spec_max_inttime), calculate_int_stepsize:bool = True) -> None:
        self.measconfig_defaults = avaspec.MeasConfigType()
        #AvaSpec MeasConfigType for spectrometer
        self.measconfig_defaults.m_StartPixel = 0
        self.measconfig_defaults.m_StopPixel = 2067
        self.measconfig_defaults.m_IntegrationTime = 0
        self.measconfig_defaults.m_IntegrationDelay = 0
        self.measconfig_defaults.m_NrAverages = spec_n_averages
        self.measconfig_defaults.m_CorDynDark_m_Enable = 1
        self.measconfig_defaults.m_CorDynDark_m_ForgetPercentage =100
        self.measconfig_defaults.m_Smoothing_m_SmoothPix = 21
        self.measconfig_defaults.m_Smoothing_m_SmoothModel = 0
        self.measconfig_defaults.m_SaturationDetection = 1
        self.measconfig_defaults.m_Trigger_m_Mode = 0
        self.measconfig_defaults.m_Trigger_m_Source = 0
        self.measconfig_defaults.m_Trigger_m_SourceType = 0
        self.measconfig_defaults.m_Control_m_StrobeControl = 0
        self.measconfig_defaults.m_Control_m_LaserDelay = 0
        self.measconfig_defaults.m_Control_m_LaserWidth = 0
        self.measconfig_defaults.m_Control_m_LaserWaveLength = 0
        self.measconfig_defaults.m_Control_m_StoreToRam = 0

        #Other default settings
        self.n_inttime_steps = n_inttime_steps
        self.start_inttime = start_inttime
        self.stepsize_inttime = None
        self.spec_min_inttime = range_inttime[0]
        self.spec_max_inttime = range_inttime[1]
        if calculate_int_stepsize:
            self.stepsize_inttime = self._calculate_stepsize_inttime(n_inttime_steps=n_inttime_steps, range_inttime=range_inttime)


    def _return_defaults(self) -> avaspec.MeasConfigType:
        '''Returns default measurement configuration (including starting int_time).'''
        defaults = self.measconfig_defaults
        defaults.m_IntegrationTime = self.start_inttime
        return defaults
    
    def _set_integration_time(self, integration_time:float) ->avaspec.MeasConfigType:
        '''Sets integration time and returns measurement settings.'''
        measconfig = self.measconfig_defaults
        measconfig.m_IntegrationTime = integration_time
        return measconfig
    
    def _calculate_stepsize_inttime(self, n_inttime_steps:int = None, range_inttime:tuple = None, store_param:bool = True, overwrite_stepsize_inttime:bool = False) ->int:
        '''Calculates the integration time's stepsize ([µs]!!!!) for the given number of steps and range.'''
        if not n_inttime_steps or not range_inttime:
            n_inttime_steps = self.n_inttime_steps
            range_inttime = (self.spec_min_inttime, self.spec_max_inttime)
        max_value = round(range_inttime[1]*1000-range_inttime[0]*1000)
        step_size = round(max_value/n_inttime_steps)
        if store_param and not self.stepsize_inttime:
            self.stepsize_inttime = step_size
        elif overwrite_stepsize_inttime and self.stepsize_inttime:
            self.stepsize_inttime = step_size
        return step_size

    def get_inttime_stepvalue(self, inttime:float) ->float:
        '''Takes a value for integration time and returns the nearest value from the integration time steps.'''
        steps = inttime * 1000 / self.stepsize_inttime
        new_inttime = self. spec_min_inttime + round(steps) * self.stepsize_inttime /1000
        return new_inttime

# DEF_TRANSMISSION_ACQUISITION = {
#     'start_inttime' : 0.002,
#     'stepsize_inttime' : 0.004,
#     'saturation_percentage' : 1,
#     'open_shutter' :True,
#     'close_shutter' :False
# }

# DEF_FLUORESCENCE_ACQUISITION = {
#     'start_inttime' : 1000,
#     'stepsize_inttime' : 500,
#     'saturation_percentage' : 1,
#     'open_shutter' :True,
#     'close_shutter' :True

# class ExperimentDefaults:
#     '''Default acquisition parameters for transmission/fluorescence experiments.'''
#     def __init__(self, acquisition_type:str) -> None:
#         self._defaults = None
#         if acquisition_type == 'transmission':
#             self._defaults = DEF_TRANSMISSION_ACQUISITION
#         elif acquisition_type == 'fluorescence':
#             self._defaults == DEF_FLUORESCENCE_ACQUISITION
#         self.start_inttime = self._defaults['start_inttime']
#         self.stepsize_inttime = self._defaults['stepsize_inttime']
#         self.saturation_percentage = self._defaults['saturation_percentage']
#         self.open_shutter = self._defaults['open_shutter']
#         self.close_shutter = self._defaults['close_shutter']

# }