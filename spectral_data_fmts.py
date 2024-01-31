'''

'''
import logging
import numpy as np
from dataclasses import dataclass, field
import json
import os
from spectra_processing import Methods

logger = logging.Logger(__name__)

_string_abbr = {
        #types
        'recorded_spectrum':'utc_id(%f):',
        'shutter_open':'(BRIGHT)',
        'shutter_closed':'(DARK)',
        'operation':'[%s:%s]',
        'subtraction':'minus',
        'addition':'plus',
        'averaged':'aver'
    }

@dataclass
class BaseSpectralData:
    '''Basic data format to catch the spectrometers output.'''
    utc_time:float
    spec_time:int
    integration_time:float
    spec_averages:int
    spectrum:np.ndarray = field(repr=False)
    saturation_spectrum:np.ndarray = field(repr=False)
    id_string:str =field(default='', init=False)

    def _to_id_string(self, string:str, separator:str = '') ->None:
        '''Appends characters to the ID string, separated by specified separator (default = empty string).'''
        if len(self.id_string) != 0:
            self.id_string += separator
        self.id_string += string



@dataclass(init=False, repr=False)
class MetaSpectralData(BaseSpectralData):
    '''Adds metadata from (other) optical instruments to BaseSpectralData.'''
    def __init__(self, base_data:BaseSpectralData, shutter:bool, uv_status:bool, uv_run_time:float, vis_status:bool, vis_run_time:float, filter_pos:int, id_string:str = None) -> None:
        super().__init__(base_data.utc_time, base_data.spec_time, base_data.integration_time, base_data.spec_averages, base_data.spectrum, base_data.saturation_spectrum)
        #self.base_data = base_data
        self.shutter = shutter 
        self.uv_status = uv_status
        self.uv_run_time = uv_run_time
        self.vis_status = vis_status
        self.vis_run_time = vis_run_time
        self.filter_pos = filter_pos
        if not id_string:
            self.id_string = f'ID{self._dark_or_not()}:{self.utc_time}'
        else:
            self.id_string = id_string
    
    def __repr__(self) -> str:
        string = f'[{self.id_string}]:[utc_time:{self.utc_time}, integration_time:{self.integration_time}, shutter:{self.shutter}, filter_pos:{self.filter_pos}]'
        return string
    
    def _base_data(self) ->BaseSpectralData:
        '''Returns BaseSpectralData object the instance was created on.'''
        base_data = BaseSpectralData(self.utc_time, self.spec_time, self.integration_time, self.spec_averages, self.spectrum, self.saturation_spectrum)
        return base_data
    
    def _dark_or_not(self) ->str:
        if self.shutter:
            ret = _string_abbr['shutter_open']
        if not self.shutter:
            ret = _string_abbr['shutter_closed']
        return ret
    
    def _compatible(self, pot_ref, inttime:bool = True, spec_averages:bool = False, inttime_threshold:float=0) ->bool:
        '''Returns True if inttime and/or spec averages of the two spectra are equal.
                inttime_threshold: allows deviation of inttimes with size inttime_threshold.'''
        ret = False
        if inttime:
            if abs(self.integration_time - pot_ref.integration_time) <= inttime_threshold:
                ret = True
        if spec_averages:
            if self.spec_averages == pot_ref.spec_averages:
                ret = True
            else:
                ret = False
        return ret
    
    def return_average_level(self)->float:
        '''Returns average of spectrum.'''
        aver = np.mean(self.spectrum)
        return aver
    
    def return_n_saturated(self)->int:
        '''Returns the number of saturated pixels.'''
        nsat = np.sum(self.saturation_spectrum)
        return nsat
    
    def return_values_at_wavelength(self, wavelength:float, area_around:int, wavelength_table:np.ndarray) ->np.ndarray:
        '''Returns slice of a spectrum with wavelength in the center of the slice and values +- area_around around the center.'''
        index = Methods._find_closest_value_index(wavelength_table, wavelength)
        if area_around == 0:
            return self.spectrum[index]
        else:
            return self.spectrum[index-area_around:index+area_around]

    def average_with(self, spectra)->None:
        '''Averages this spectrum with spectra from list or single spectrum..'''
        #add standard deviation to averaged spectrum!
        #could be really useful to detect changes, stability,.... 
        sum_spectra = self.spectrum
        if type(spectra) == list:
            self._to_id_string(f'[{_string_abbr["averaged"]}:', '_')
            n_spectra = len(spectra) +1
            for i in spectra:
                if self._compatible(i):
                    sum_spectra += i.spectrum
                    self._to_id_string(f'({i.id_string})')
                else:
                    raise ValueError
            self._to_id_string(']')
        elif type(spectra) == DarkCorrectedSpectralData or type(spectra) == MetaSpectralData or type(spectra) == ManipulatedSpectralData:
            n_spectra = 2
            sum_spectra += spectra.spectrum
            self._to_id_string(f'[{_string_abbr["averaged"]}:({spectra.id_string})]')
        avgd = sum_spectra/n_spectra
        self.spectrum = avgd

    def _subtract_spectrum(self, spectral_data, write_id:bool = True)->np.ndarray:
        '''Subtracts spectrum "spectrum" from a copy of the instances spectrum and notes it in the ID_string.'''
        subtspec = np.copy(self.spectrum) - np.copy(spectral_data.spectrum)
        if write_id:
            self._to_id_string(f'[{_string_abbr["subtraction"]}:({spectral_data.id_string})]', '_')
        return subtspec
    
    def return_transmittance(self, spectral_ref)->np.ndarray:
        '''Returns self/ref.'''
        Trm = np.copy(self.spectrum)/np.copy(spectral_ref.spectrum)
        return Trm
    
    def return_absorbance(self, spectral_ref)->np.ndarray:
        '''Returns log(ref/self).'''
        Abs = np.log(np.copy(spectral_ref.spectrum) / np.copy(self.spectrum))
        return Abs

    def compare_spectrum_to_list(self, spectral_data:list)->list:
        '''Returns differences of the spectrum compared to spectra in a list.
            (Used for time resolving).'''
        diffs = []
        for spectrum in spectral_data:
            diff = self._subtract_spectrum(spectrum, write_id=False)
            diffs.append(diff)
        return diffs
    
    def compare_area_to_list(self, spectral_data:list, wavelength:float, area_around:int, wavelength_table:np.ndarray)->list:
        '''Compares area of a spectrum with spectra from a list and returns differences in a list.'''
        diffs = []
        area = self.return_values_at_wavelength(wavelength=wavelength, area_around=area_around, wavelength_table=wavelength_table)
        for spectrum in spectral_data:
            c_area = spectrum.return_values_at_wavelength(wavelength=wavelength, area_around=area_around, wavelength_table=wavelength_table)
            diff = np.copy(area)-np.copy(c_area)
            diffs.append(diff)
        return diffs

    def get_abs_difference(self, spectral_data, compatibility_check:bool=True)->float:
        '''Returns the absolute difference of two spectra.'''
        check = True
        if compatibility_check:
            check = self._compatible(spectral_data)
        if check:
            diff = self._subtract_spectrum(spectral_data, write_id=False)
            absdiff = np.sum(abs(diff))
        else:
            raise ValueError
        return absdiff
    
@dataclass(init=False, repr=False)
class DarkCorrectedSpectralData(MetaSpectralData):
    '''Adds a dark correction to MetaSpectralData. Needs a spectrum and matching dark reference.'''
    def __init__(self, meta_data: MetaSpectralData, dark_reference:MetaSpectralData = None, inst_from_json:bool =False, ignore_shutter = False) -> None:
        super().__init__(meta_data._base_data(), meta_data.shutter, meta_data.uv_status, meta_data.uv_run_time, meta_data.vis_status, meta_data.vis_run_time, meta_data.filter_pos)
        if not ignore_shutter and meta_data.shutter != 1:
            raise ValueError(f"Shutter of given spectrum is '{meta_data.shutter}.")
        if not inst_from_json:
            if not ignore_shutter and dark_reference.shutter != 0:
                raise ValueError(f"Shutter in dark reference has value '{dark_reference.shutter}.'")
            if self.integration_time != dark_reference.integration_time:
                raise ValueError(f"Integration times not matching ({self.integration_time} and {dark_reference.integration_time}).")
            self.backup_dark_utc_time = dark_reference.utc_time
            # self.backup_dark_reference = dark_reference.spectrum #backup of dark reference. necessary?
            # self.backup_basespectrum = self.spectrum #backup. necessary?

            self._dark_reference(dark_reference)
        if inst_from_json:
            backup_basespectrum = None

    def _meta_data(self) ->MetaSpectralData:
        metadata = MetaSpectralData(self._base_data, self.shutter, self.uv_status, self.uv_run_time, self.vis_run_time, self.vis_run_time, self.filter_pos)
        return metadata
    
    def _dark_reference(self, dark_ref:MetaSpectralData)->None:
        '''Applies dark reference to instance and sets spectrum to dark corrected spectrum.'''
        if self._compatible(dark_ref, inttime=True, spec_averages=False):
            self.spectrum = self._subtract_spectrum(dark_ref)
        else:
            logger.error(f'Dark reference {dark_ref.utc_time} not compatible with spectrum {self.utc_time}.')

@dataclass(init=False, repr=False)
class ManipulatedSpectralData(DarkCorrectedSpectralData):
    '''NEVER USED; UNNECESSARY!
    Adds several operations for spectra manipulation.'''
    def __init__(self, corrected_data:DarkCorrectedSpectralData)-> None:
        self.backup_corrected_data = corrected_data #backup
        self.id_string = corrected_data.id_string

    def _add_spectrum(self, spectrum:DarkCorrectedSpectralData)->np.ndarray:
        '''Adds spectrum and notes it in the ID_string.'''
        addspec = np.copy(self.spectrum) -np.copy(spectrum.spectrum)
        self._to_id_string(f'[{_string_abbr["addition"]}:({spectrum.id_string})]', '_')
        return addspec

def save_json(spectrum:MetaSpectralData, filepath:str) ->None:
    '''Saves a spectrum in a json file.'''
    rdict = spectrum.__dict__
    for key in rdict:
        if type(rdict[key]) == np.ndarray:
            rdict[key] = np.array2string(rdict[key], separator=',', threshold=2048)
    dict_rep = {f'{rdict["utc_time"]}':rdict}
    alldata={}
    if os.path.isfile(filepath):
        with open(filepath, 'r') as inputfile:
            try:
                existingdata = json.load(inputfile)
                if existingdata:
                    alldata = existingdata
            except json.JSONDecodeError:
                logger.error(f'Filepath {filepath} not empty and not readable.')
    alldata.update(dict_rep)
    with open(filepath, 'w') as jsonfile:
        json.dump(alldata, jsonfile)
    

def read_json(filepath:str, ignore_shutter:bool=False)->list:
    '''Reads a spectrum dumped to a json file. Returns MetaSpectralData or DarkCorrectedSpectralData.'''
    output = []
    with open(filepath, 'r') as jsfile:
        for line in jsfile:
            data = json.loads(line)
            for key in data:
                spectrum = data[key]
                basedata = BaseSpectralData(spectrum['utc_time'], spectrum['spec_time'], spectrum['integration_time'], spectrum['spec_averages'], spectrum['spectrum'], spectrum['saturation_spectrum'])
                if 'shutter' in spectrum.keys():
                    metadata = MetaSpectralData(basedata, spectrum['shutter'], spectrum['uv_status'], spectrum['uv_run_time'], spectrum['vis_status'], spectrum['vis_run_time'], spectrum['filter_pos'], id_string=spectrum['id_string'])
                    if 'backup_dark_utc_time' in spectrum.keys():
                        corrdata = DarkCorrectedSpectralData(metadata, inst_from_json=True, ignore_shutter=ignore_shutter)
                        try:
                            corrdata.backup_basespectrum = spectrum['backup_basespectrum']
                            corrdata.backup_dark_reference = spectrum['backup_dark_reference']
                        except:
                            pass
                        corrdata.backup_dark_utc_time = spectrum['backup_dark_utc_time']
                        output.append(corrdata)
                    else:
                        output.append(metadata)
                else: 
                    output.append(basedata)
    for i in output:
        i.spectrum = np.array(eval(i.spectrum))
        i.saturation_spectrum = np.array(eval(i.saturation_spectrum))
        try:
            i.backup_basespectrum = np.array(eval(i.backup_basespectrum))
            i.backup_dark_reference = np.array(eval(i.backup_dark_reference))
        except Exception as e:
            # logger.info(e)
            pass
    return output


def dark_reference_files(spectra_filepath:str, reference_filepath:str, output_filepath:str = None, ignore_shutter:bool = False) ->list[DarkCorrectedSpectralData]:
    '''Reads spectra and references from a json file, references the spectra 
    and creates a new json file with referenced spectra when given filepath.'''
    output = []
    spectra = read_json(spectra_filepath)
    references = read_json(reference_filepath)
    for spectrum in spectra:
        pot_refs = []
        for ref in references:
            if spectrum._compatible(ref):
                pot_refs.append(ref)
        if len(pot_refs) ==0:
            logger.error(f'No matching reference found for {spectrum.id_string} in {reference_filepath}.')
        newest = pot_refs[0]
        for com_ref in pot_refs:
            if com_ref.utc_time > newest.utc_time:
                newest = com_ref
        refd_spectrum = DarkCorrectedSpectralData(spectrum, newest, ignore_shutter=ignore_shutter)
        output.append(refd_spectrum)
        if output_filepath:
            save_json(refd_spectrum, output_filepath) 
        logger.debug(f'Referenced spectrum {spectrum.id_string} with {newest.id_string}, saved to {output_filepath}.')
    return output

def load_json(filepath:str)->dict:
    '''Reads a json file and returns dictionary {filepath : filepath contents}'''
    output = []
    with open(filepath, 'r') as jsfile:
        for line in jsfile:
            data = json.loads(line)
            for key in data:
                spectrum = data[key]
                basedata = BaseSpectralData(spectrum['utc_time'], spectrum['spec_time'], spectrum['integration_time'], spectrum['spec_averages'], spectrum['spectrum'], spectrum['saturation_spectrum'])
                if 'shutter' in spectrum.keys():
                    metadata = MetaSpectralData(basedata, spectrum['shutter'], spectrum['uv_status'], spectrum['uv_run_time'], spectrum['vis_status'], spectrum['vis_run_time'], spectrum['filter_pos'], id_string=spectrum['id_string'])
                    if 'backup_dark_utc_time' in spectrum.keys():
                        corrdata = DarkCorrectedSpectralData(metadata, inst_from_json=True)
                        corrdata.backup_basespectrum = spectrum['backup_basespectrum']
                        corrdata.backup_dark_reference = spectrum['backup_dark_reference']
                        corrdata.backup_dark_utc_time = spectrum['backup_dark_utc_time']
                        output.append(corrdata)
                    else:
                        output.append(metadata)
                else: 
                    output.append(basedata)
    for i in output:
        i.spectrum = np.array(eval(i.spectrum))
        i.saturation_spectrum = np.array(eval(i.saturation_spectrum))
        try:
            i.backup_dark_reference = np.array(eval(i.backup_dark_reference))
            i.backup_basespectrum = np.array(eval(i.backup_basespectrum))
        except Exception as e:
            logger.info(e)
    return {filepath:output}