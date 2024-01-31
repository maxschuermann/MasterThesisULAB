'''


'''
import logging

from serial.tools import list_ports
from plateflo import fetbox
from time import sleep

ls_logger = logging.Logger(__name__)
fetbox.fetbox_logger.setLevel(logging.ERROR)

class LightSource:
    '''Serial control of Edmund Optics BDS130A with an Arduino-based controller.'''   
    def __init__(self) -> None:
        self.pins={
        'PIN_TTL_ENABLE' : 7,
        'PIN_UV_CTRL' : 4,
        'PIN_SHUTTER_CTRL' : 5,
        'PIN_VIS_CTRL' : 6,
        'PIN_UV_MON' : 8,
        'PIN_SHUTTER_MON' : 9,
        'PIN_VIS_MON' : 10
        }
        self.controller = None
        self.port = None

    def connect(self):
        '''Establishes serial connection to the Arduino light source controller.'''
        ports = list_ports.comports()
        for port in ports:
            if 'CH340' in port.description:
                self.port= port
        if not self.port:
            ls_logger.error('No controllers detected.')
            raise ConnectionError("No FETbox controllers detected.")
        try:
            self.controller= fetbox.FETbox(self.port.usb_description())
        except Exception as exc:
            ls_logger.error('Could not connect to controller')
            raise ConnectionError("Could not connect to controller.") from exc
        ls_logger.info('Connected to FETbox light source controller')
        

    def connect_retry(self, attempts:int=3, retry_delay:float=1)->bool:
        '''Tries to establish connection to FETBox `attempts` times.
            retry_delay: time between attempts [s].
            Returns `True` if connection attempt successful.'''
        for att in range(attempts):
            try:
                self.connect()
                if self.controller is not None:
                    ls_logger.info(f'Connected to FETBox, after {att+1} attempts.')
                    return True
            except Exception as e:
                ls_logger.info(f'{e}, attempt {att+1}.')
                sleep(retry_delay)
        ls_logger.error(f'Could not connect to FETBox after {attempts} attempts.')

    def close(self):
        '''Disable TTL control and close serial connection.'''
        self.controller.digital_write(self.pins['PIN_TTL_ENABLE'], 0)
        self.controller.kill()
        self.controller = None
        ls_logger.info('Closed connection to FETBox controlling lightsource.')

    def digitalwrite(self, pin:str, value:bool):
        '''Sets a pin to a certain value (bool).'''
        self.controller.digital_write(self.pins[pin], value)

    def digitalread(self, pin:str)->bool:
        '''Reads the digital value of a pin (bool).'''
        value = self.controller.digital_read(self.pins[pin])
        return value
    
    def set_ttl_control(self, enable:bool):
        '''Enables/disables the TTL control of BDS130A.'''
        self.digitalwrite('PIN_TTL_ENABLE', enable)
        ls_logger.info(f'TTL control set to {enable}')

    def get_uv(self) -> bool:
        '''Displays the status of the UV (Deuterium) lamp.'''
        uv = self.digitalread('PIN_UV_MON')
        return uv

    def set_uv(self, uv:bool) -> None:
        '''Turns the UV (Deuterium) lamp on/off.'''
        self.digitalwrite('PIN_UV_CTRL', uv)
        ls_logger.info(f'UV lamp set to {uv}.')

    def get_vis(self) -> bool:
        '''Displays the status of the Visible spectrum (Tungsten/Halogen) lamp.'''
        vis = self.digitalread('PIN_VIS_MON')
        return vis
    
    def set_vis(self, vis:bool) -> None:
        '''Turns the Visible spectrum (Tungsten/Halogen) lamp on/off.'''
        self.digitalwrite('PIN_VIS_CTRL', vis)
        ls_logger.info(f'Vis lamp set to {vis}.')

    def get_shutter(self) ->bool:
        '''Displays the status of the shutter.

            0: closed,
            1: open.'''
        shutter = self.digitalread('PIN_SHUTTER_MON')
        return shutter
    
    def set_shutter(self, shutter:bool) -> None:
        '''Sets the shutter open/close.
            
            0: closed,
            1: open.'''
        self.digitalwrite('PIN_SHUTTER_CTRL', shutter)

def main():
    'Test'
    light = LightSource()
    light.connect()
    light.set_shutter(1)
    light.set_uv(1)
    light.get_uv()
    light.set_uv(0)
    light.set_shutter(0)
    light.close()

if __name__ == "__main__":
    from logging.handlers import TimedRotatingFileHandler        
    log_filename = "lightsource_log.log"
    handler = TimedRotatingFileHandler(
        log_filename,
        when="midnight",
        interval=1,
        backupCount=7
    )
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    handler.setLevel(logging.DEBUG)
    ls_logger.setLevel(logging.DEBUG)
    ls_logger.addHandler(handler)

    main() 
