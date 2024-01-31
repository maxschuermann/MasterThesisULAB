'''Cleaned version of experiment.py.
    Contains functions dedicated to analyze the reactors output (rather dynamic measurements, analysis of series of spectra).
    Additional: Utility functions for data manipulation and visualization.
    Spectra analysis can be performed (depending on the application) with suiting scipy functions such as peak finding. 
    Basic methods (maximum at a given wavelength in a time series, ...) are included.'''

import logging
import numpy as np
import spectral_data_fmts as fmts
from spectra_processing import Methods
from dataclasses import dataclass, field
from time import time, sleep
import matplotlib.pyplot as plt
import optics_control_final as opt_ctrl

import os
import json

logger = logging.Logger(__name__)

class ExperimentMeasurements:
    '''Class containing functions to create series of spectra,
    wrappers to access equipment, wrappers for single measurements.'''
    def __init__(self) -> None:
        self.opt_ctrl = opt_ctrl.OpticsControl()
        self._acquisitions = []
        self.wavelength_table = None
        self.experiment_start_time = time()

    def connect_devices(self,spectrometer:bool = True, filterwheel:bool = False, lightsource:bool = True, ls_attempts:int = 3, ls_retry_delay:float = 0.5)->None:
        '''Select and start control over devices.'''
        self.opt_ctrl.connect(spectrometer=spectrometer, filterwheel=filterwheel, lightsource=lightsource, ls_attempts=ls_attempts, ls_retry_delay=ls_retry_delay)
        if spectrometer:
            self.wavelengths = self._auto_spectra.opt_ctrl.spectrometer.wavelengths
        logger.info(f'Connected to optical devices: spec = {spectrometer}, FW102C = {filterwheel}, lamp = {lightsource}.')

    def disconnect_devices(self, spectrometer:bool = True, filterwheel:bool = False, lightsource:bool = True)->None:
        '''Select and stop control over devices.'''
        self.opt_ctrl.disconnect(spectrometer=spectrometer, filterwheel=filterwheel, lightsource=lightsource)
        logger.info(f'Disconnected from optical devices: spec = {spectrometer}, FW102C = {filterwheel}, lamp = {lightsource}.')

    def import_wavelengths_from_txt(self, filepath:str = 'wavelengths.txt')->None:
        '''Reads txt file and stores wavelength table to instance.'''
        self.wavelengths = Methods.read_wavelength_txt(filepath=filepath)
        logger.info(f'Read wavelengths from {filepath}.')

    def import_wavelengths_from_spectrometer(self)->None:
        '''Tries to access wavelength table from Avantes spectrometer.'''
        self.wavelengths = self.opt_ctrl.spectrometer.wavelengths
        if self.wavelengths is not type(np.ndarray):
            try:
                self.wavelengths = self.opt_ctrl.spectrometer.get_wavelengths()
            except Exception as exc:
                logger.error(exc)

    def _find_wavelength_index(self, wavelength:float) ->int:
        '''Returns index of nearest wavelength contained in instances wavelength item.'''
        idx = Methods._find_closest_value_index(self.wavelengths, wavelength)
        return idx
    
    def measure_until_saturation(self, start_inttime:float, inttime_stepsize:float, saturation_limit:float, max_n_meas:int, measure_refs:bool=True,**kwargs):
        '''Measures spectra until a given number of pixels (percent of 2048 total pixels) is saturated.
        Was useful for recoding fluorescence spectra.'''
        #this is more a "static" measurement
        max_sat_pixels = 2048/100 * saturation_limit
        meas_count = 0
        last_saturation = 0
        inttimes = []
        spectra_out = []
        while last_saturation < max_sat_pixels and meas_count < max_n_meas and start_inttime <= 30000:
            spectrum = self.opt_ctrl.capture_spectrum(inttime=start_inttime, shutter=True, **kwargs)
            inttimes.append(start_inttime)
            start_inttime += inttime_stepsize
            meas_count += 1
            last_saturation = spectrum[0].return_n_saturated()
            spectra_out.append(spectrum[0])
        if measure_refs:
            refs = self._record_dark_ref_from_list(inttimes, **kwargs)
            refd = []
            for i in range(len(spectra_out)):
                refd.append(fmts.DarkCorrectedSpectralData(spectra_out[i], refs[i]))
            return refd
        else:
            return inttimes, spectra_out
        
    def _record_dark_ref_from_list(self, inttime_list, **kwargs):
        '''Measures stray light reference spectra from a list of integration times.'''
        spectra_out= []
        for inttime in inttime_list:
            spectrum = self.opt_ctrl.capture_spectrum(inttime=inttime, shutter=False, **kwargs)
            spectra_out.append(spectrum[0])
        return spectra_out
    
    def continuous_measurement(self, inttime:float, duration:float, interval:float, **kwargs):
        '''Records a row of spectra.
            duration: duration of cont measurement [s]
            inttime: integration time [ms]
            interval: time between measurements'''
        #measuring as fast as possible creates way to much data to handle
        start_time = time()
        spectra_out=[]
        while (time()-start_time) < duration:
            spectrum = self.opt_ctrl.capture_spectrum(inttime=inttime, **kwargs)
            spectra_out.append(spectrum[0])
            sleep(interval)
        acq_item = ExperimentAcquisition(start_time=start_time, acq_type='continuous_measurement', spectra=spectra_out)
        self._acquisitions.append(acq_item)
        return acq_item
    
    def continuous_measurement_dark_refs(self, inttime:float, duration:float, interval:float, dref_interval:int, **kwargs)->None:
        '''Starts continuous measurement of spectra. 
            inttime: integration time [ms]
            duration: duration of cont measurement [s]
            interval: time between measurements [s]
            dref_interval: measures dref every dref interval spectra
            '''
        start_time = time()
        spectra = []
        drefs = []
        count = 0
        while (time()-start_time) < duration:
            count +=1
            spectrum = self.opt_ctrl.capture_spectrum(inttime=inttime, **kwargs)
            spectra.append(spectrum[0])
            if count == dref_interval:
                sleep(interval)
                dref = self.opt_ctrl.capture_spectrum(inttime=inttime, shutter=False, **kwargs)
                drefs.append(dref[0])
                count=0
            sleep(interval)
        self._acquisitions.append(ExperimentAcquisition(start_time=start_time, acq_type='continuous_measurement', spectra=spectra))
        self._acquisitions.append(ExperimentAcquisition(start_time=start_time, acq_type='dark_refs_cont', spectra=drefs))
        
    def save_acquisitions_to_json(self, directory:str)->None:
        '''Saves all acquisitions in the experiment to "directory".'''
        os.makedirs(directory, exist_ok=True)
        for acq in self._acquisitions:
            filepath = os.path.join(directory, f'{acq.start_time}_{acq.acq_type}.json')
            spectral_data = {}
            for spectrum in acq.spectra:
                dict_rep = spectrum.__dict__
                for key in dict_rep:
                    if type(dict_rep[key]) == np.ndarray:
                        dict_rep[key] = np.array2string(dict_rep[key], separator=',', threshold=2048)
                spec_dict_out = {f'{dict_rep["utc_time"]}':dict_rep}
                spectral_data.update(spec_dict_out)
            acq.spectra = spectral_data
            with open(filepath, 'w') as file:
                json.dump(acq.__dict__, file)


@dataclass
class ExperimentAcquisition:
    '''Class to contain a row of measurements. Includes functions for referencing, ...'''
    start_time:float
    acq_type:str
    spectra:list[fmts.MetaSpectralData] = field(repr=False)
    refs:list = field(init=False)

    def __repr__(self) -> str:
        string = f'{self.acq_type}_{Methods.convert_utc_to_human(self.start_time)}: {len(self.spectra)} spectra'
        return string
    
    def dark_reference_spectra(self, dref)->None:
        '''Dark reference all spectra'''
        #map utc times to find nearest value
        for spectrum in self.spectra:
            nearest = min(dref.spectra, key=lambda x: abs(x.utc_time - spectrum.utc_time))
            spectrum = fmts.DarkCorrectedSpectralData(spectrum, nearest, ignore_shutter=True)
    
    def subtraction_with_spectra(self, ref_start, ref_end)->None:
        '''Subtract average from ref_start, ref_end from all other spectra.
        refs are stored as self.refs.'''
        reltimes = self.return_relative_times()
        spectra = self.spectra
        refs = spectra[ref_start:ref_end]
        del spectra[ref_start:ref_end]
        refs[0].average_with(refs[1:])
        for spectrum in spectra:
            spectrum = spectrum._subtract_spectrum(refs[0], write_id=True)
        acq_item = ExperimentAcquisition(start_time=self.start_time, acq_type=f'subtraction_rel_to{reltimes[ref_start]}:{reltimes[ref_end]}', spectra=spectra)
        return acq_item
    
    def return_transmittance_rel_to_spectra(self, ref_start, ref_end)->None:
        '''Calculates transmittance relative to average of spectra from ref_start(index) to ref_end.'''
        reltimes = self.return_relative_times()
        spectra = self.spectra
        refs = spectra[ref_start:ref_end]
        del spectra[ref_start:ref_end]
        refs[0].average_with(refs[1:])
        for spectrum in spectra:
            spectrum = spectrum.return_transmittance(refs[0])
        acq_item = ExperimentAcquisition(start_time=self.start_time, acq_type=f'transmittance_rel_to{reltimes[ref_start]}:{reltimes[ref_end]}', spectra=spectra)
        return acq_item

    def return_absorbance_rel_to_spectra(self, ref_start, ref_end):
        '''Calculates absorbance relative to refs average.'''
        reltimes = self.return_relative_times()
        spectra = self.spectra
        refs = spectra[ref_start:ref_end]
        del spectra[ref_start:ref_end]
        refs[0].average_with(refs[1:])
        for spectrum in spectra:
            spectrum = spectrum.return_absorbance(refs[0])
        acq_item = ExperimentAcquisition(start_time=self.start_time, acq_type=f'absorbance_rel_to{reltimes[ref_start]}:{reltimes[ref_end]}', spectra=spectra)
        return acq_item
    
    def return_relative_times(self)->list:
        '''Returns acquisition times relative to the start point of the measurement.'''
        rel_times = []
        spectra = self.spectra
        for spectrum in spectra:
            rel_time = spectrum.utc_time - self.start_time
            rel_times.append(rel_time)
        return rel_times
    
    def return_areas_around_wavelength(self, wavelength:float, area_around:int, wavelenght_table:np.ndarray)->list:
        '''Returns area +- area_around around given wavelength.'''
        slices = []
        spectra = self.spectra
        for spectrum in spectra:
            area = spectrum.return_values_at_wavelength(wavelength, area_around, wavelenght_table)
            slices.append(area)
        return slices
    
    def return_spectra_in_timeperiod(self, time_start:float, time_stop:float):
        '''Returns spectra in time period [s] between start and end relative to start of the acquisition.'''
        reltimes = self.return_relative_times()
        startindex = Methods._find_closest_value_index(reltimes, time_start)
        stopindex = Methods._find_closest_value_index(reltimes, time_stop)
        spectra = self.spectra[startindex:stopindex]
        acq_item = ExperimentAcquisition(self.start_time, acq_type=f'time_period_{time_start}:{time_stop}', spectra=spectra)
        return acq_item
    
    def return_minimum_value_in_wavelength_area_over_time(self, wavelength:float, area_around:int, wavelength_table:np.ndarray)->list:
        '''Returns minimum values from wavelength area for the whole acquisition.'''
        wl_values = self.return_areas_around_wavelength(wavelength=wavelength, area_around=area_around, wavelength_table=wavelength_table)
        wl_mins = []
        for wl_array in wl_values:
            wl_mins.append(np.min(wl_array))
        return wl_mins
    
    def return_absolute_minimum_value_from_wl_area(self, wavelength:float, area_around:int, wavelength_table:np.ndarray)->float:
        '''Returns minimum value at a certain wavelength (or minimum in an area of wavelengths) of the acquisition over time.'''
        wl_mins = self.return_minimum_value_in_wavelength_area_over_time(wavelength=wavelength, area_around=area_around, wavelength_table=wavelength_table)
        return min(wl_mins)
    
    def return_gradient(self)->list:
        '''Returns difference from spectrum to spectrum'''
        diffs = []
        for i in range(len(self.spectra)-1):
            diff = np.copy(self.spectra[i+1].spectrum) -np.copy(self.spectra[i].spectrum)
            diffs.append(diff)
        return diffs
    
class ExperimentResults:
    '''Class to create a result from the experiments acquisitions.'''
    #This section is not that clean, there are mostly plotting functions in here
    #Most results are generated by class above (Experiment Acquisition)
    #Interesting is maybe the "find_timeslices" function
    #   idea was to create segments from my time series based on deviation
    #   looks really cool when plotted
    def __init__(self, acq_list:list[ExperimentAcquisition], wlengths_from_txt:str = 'wavelengths.txt') -> None:
        self.wavelengths = None
        self.acq_list= acq_list
        self.acq_data:ExperimentAcquisition = None
        self.acq_dref:ExperimentAcquisition = None
        self.timeslice_list = []
        self._sort_acqs()
        self._dref_results()
        # self.reltimes = self.acq_data.return_relative_times()
        if '.txt' in wlengths_from_txt:
            self.wavelengths = Methods.read_wavelength_txt(wlengths_from_txt)
    
    def _sort_acqs(self)->None:
        for acq in self.acq_list:
            if 'continuous' in acq.acq_type and self.acq_data is None:
                self.acq_data = acq
            elif 'dark' in acq.acq_type and self.acq_data is None:
                self.acq_dref = acq

    def _dref_results(self)->None:
        if self.acq_data and self.acq_dref is not None:
            self.acq_data.dark_reference_spectra(self.acq_dref)

    def smooth_spectral_data(self, window_size:int):
        for spectrum in self.acq_data.spectra:
            spectrum.spectrum = Methods.moving_average(spectrum.spectrum, window=window_size)

    def smooth_time_domain(self, window_size:int):
        self.acq_data.spectra
        for i in range(len(self.acq_data.spectra)-(window_size-1)):
            self.acq_data.spectra[i].average_with(self.acq_data.spectra[i+1:i+window_size])

    def find_timeslices(self, slope_threshold:float, size_threshold:float,  compare_next_n:int, wl_area:tuple = (485, 3)):
        self.timeslice_list = []
        indices = [0]
        for i in range(len(self.acq_data.spectra)-compare_next_n):
            if wl_area is not None:
                diffs = self.acq_data.spectra[i].compare_area_to_list(self.acq_data.spectra[i+1:i+compare_next_n],wavelength= wl_area[0], area_around= wl_area[1], wavelength_table=self.wavelengths)
            else:
                diffs = self.acq_data.spectra[i].compare_spectrum_to_list(self.acq_data.spectra[i+1:i+compare_next_n])
            totaldiff = 0
            maxdiff = np.max(self.acq_data.spectra[i]._subtract_spectrum(self.acq_data.spectra[i+compare_next_n], write_id=False))
            for diff in diffs:
                totaldiff += np.mean(np.abs(diff))
                totaldiff /= compare_next_n
            if totaldiff >= slope_threshold or maxdiff >= size_threshold:
                indices.append(i+int(compare_next_n/2))
        indices.append(len(self.acq_data.spectra))
        idcs = []
        for k in range(len(indices)-1):
            thicc = indices[k]- indices[k+1]
            if abs(thicc) <=5:
                idcs.append(k)
        findcs = [indices[p] for p in range(len(indices)) if p not in idcs]
        relt = self.acq_data.return_relative_times()
        for j in range(len(findcs)-1):
            timeslice = ExperimentAcquisition(self.acq_data.start_time, acq_type=f'timeslice:{relt[j]}_{relt[j+1]}', spectra=self.acq_data.spectra[findcs[j]:findcs[j+1]])
            self.timeslice_list.append(timeslice)

    def show_acquisition3D(self)->None:
        x_wl = self.wavelengths
        y_time = self.acq_data.return_relative_times()
        x_wl, y_time = np.meshgrid(x_wl, y_time)
        z_spectra = []
        for spectrum in self.acq_data.spectra:
            z_spectra.append(spectrum.spectrum)
        z_spectra = np.array(z_spectra)
        fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize = (8,8))
        ax.plot_surface(x_wl, y_time, z_spectra)
        ax.set_zlim(np.min(z_spectra), np.max(z_spectra))
        ax.set_xlabel('$\lambda$ [nm]')
        ax.set_ylabel('Time since start [s]')
        plt.show()

    def show_acq_contour(self, levels:int = 10, timeslot:tuple = None,filepath=None, size=(8,8), scale_spectra=1, zlim=(0,1))->None:
        if timeslot:
            ts_data = self.acq_data.return_spectra_in_timeperiod(timeslot[0], timeslot[1])
        else:
            ts_data = self.acq_data
        x_wl=self.wavelengths
        y_time = ts_data.return_relative_times()
        x_wl, y_time = np.meshgrid(x_wl, y_time)
        z_spectra = []
        for spectrum in ts_data.spectra:
            z_spectra.append(spectrum.spectrum * scale_spectra)
        z_spectra = np.array(z_spectra)
        plt.figure(figsize=size)
        plt.xlabel('$\lambda$ [nm]')
        plt.ylabel('Time since start [s]')
        plt.xlim(np.min(x_wl), np.max(x_wl))
        plt.ylim(np.min(y_time), np.max(y_time))
        c_levels= np.linspace(zlim[0],zlim[1]*scale_spectra, levels)
        cplt = plt.contourf(x_wl,y_time,z_spectra,levels=c_levels)
        plt.colorbar(cplt)
        if filepath:
            plt.savefig(filepath, format='svg')
        plt.show()

    def show_single_wl_time(self, wavelength:float, timeslot:tuple=None,filepath=None, ylabel = 'Extinction [%]',ylim=(0,100), size=(8,8), yscale=100, lgd=True)->None:
        if timeslot:
            ts_data = self.acq_data.return_spectra_in_timeperiod(timeslot[0], timeslot[1])
        else:
            ts_data = self.acq_data
        reltimes = ts_data.return_relative_times()
        y_spectra = []
        for spectrum in ts_data.spectra:
            wl_value = spectrum.return_values_at_wavelength(wavelength=wavelength, area_around=0, wavelength_table=self.wavelengths) *yscale
            y_spectra.append(wl_value)
        y_spectra = np.array(y_spectra)
        plt.figure(figsize=size)
        plt.xlim(np.min(reltimes), np.max(reltimes))
        plt.ylim(ylim[0], ylim[1])
        plt.xlabel('Time since start [s]')
        plt.ylabel(ylabel)
        plt.plot(reltimes, y_spectra, label=f'{wavelength} nm')
        if lgd:
            plt.legend()
        if filepath:
            plt.savefig(filepath, format='svg')
        plt.show()


    def show_timeslices_single_plot(self)->None:
        cmap1 = plt.cm.get_cmap('Blues')
        cmap2 = plt.cm.get_cmap('Reds')
        cmaps = [cmap1, cmap2]
        tlist = self.timeslice_list
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection= '3d')
        for i in range(len(tlist)):
            if i % 2 ==0:
                color = cmaps[1]
            else:
                color = cmaps[0]
            x_wls = self.wavelengths
            y_times = tlist[i].return_relative_times()
            x_wls, y_times = np.meshgrid(x_wls,y_times)
            zspectra = [tlist[i].spectra[j].spectrum for j in range(len(tlist[i].spectra))]
            zspectra = np.array(zspectra)
            ax.plot_surface(x_wls, y_times, zspectra, cmap = color)
        plt.show()
        
def load_acquisitions_from_directory(directory:str) ->list:
    '''Loads all acquisitions (jsons) of an experiment from a directory.'''
    files = os.listdir(directory)
    jsons = []
    for file in files: 
        if '.json' in file:
            jsons.append(file)
    readacqs = []
    for jsonfile in jsons:
        filepath = os.path.join(directory, jsonfile)
        with open(filepath, 'r') as readfile:
            rawdata = json.load(readfile)
            readacqs.append(rawdata)
    acqs = []
    for readacq in readacqs:
        ###in line below is "type" not "acq_type". change or implement loop to remain compatible
        ea_acq = ExperimentAcquisition(readacq['start_time'], readacq['acq_type'], readacq['spectra'])
        acqs.append(ea_acq)
    for acq in acqs:
        spectra =[]
        rwspectra = acq.spectra
        for tme in rwspectra:
            spectrum = rwspectra[tme]
            basedata = fmts.BaseSpectralData(spectrum['utc_time'], spectrum['spec_time'], spectrum['integration_time'], spectrum['spec_averages'], spectrum['spectrum'], spectrum['saturation_spectrum'])
            if 'shutter' in spectrum.keys():
                metadata = fmts.MetaSpectralData(basedata, spectrum['shutter'], spectrum['uv_status'], spectrum['uv_run_time'], spectrum['vis_status'], spectrum['vis_run_time'], spectrum['filter_pos'], id_string=spectrum['id_string'])
                if 'backup_dark_utc_time' in spectrum.keys():
                    corrdata = fmts.DarkCorrectedSpectralData(metadata, inst_from_json=True)
                    corrdata.backup_basespectrum = spectrum['backup_basespectrum']
                    corrdata.backup_dark_reference = spectrum['backup_dark_reference']
                    corrdata.backup_dark_utc_time = spectrum['backup_dark_utc_time']
                    spectra.append(corrdata)
                else:
                    spectra.append(metadata)
            else: 
                spectra.append(basedata)
        for spec in spectra:
            spec.spectrum = np.array(eval(spec.spectrum))
            spec.saturation_spectrum = np.array(eval(spec.saturation_spectrum))
            try:
                spec.backup_dark_reference = np.array(eval(spec.backup_dark_reference))
                spec.backup_basespectrum = np.array(eval(spec.backup_basespectrum))
            except Exception as e:
                # logger.debug(e)
                pass
        acq.spectra = spectra
    return acqs