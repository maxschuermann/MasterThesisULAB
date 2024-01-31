'''
ThorLabs filter wheel serial control.

Copyright Robert Pazdzior & Max Schuermann (2023)
'''
import logging
from time import time

import serial
from serial.tools import list_ports

logger = logging.Logger(__name__)


class FW102C():
    '''Thorlabs motorized filter wheel (FW102C) serial control.'''
    CMDS = {
        'get_id': '*idn?',
        'set_pos': 'pos=%i',
        'get_pos': 'pos?',
        'set_pcount': 'pcount=%i',
        'get_pcount': 'pcount?',
        'set_speed': 'speed=%i',
        'get_speed': 'speed?',
        'set_sens': 'sensors=%i',
        'get_sens': 'sensors?',
        'set_baud': 'baud=%i',
        'get_baud': 'baud?',
        'set_trigger': 'trig=%i',
        'get_trigger': 'trig?',
        'save': 'save'
    }
    def __init__(self, hwid: str, pcount:int = 6, baud:int = 115200) -> None:
        self.port = None
        ports = list_ports.comports()
        if not ports:
            logger.error('No available serial ports detected.')
            raise ConnectionError('No available serial ports detected.')
        logger.debug('Scanning for filter wheel on: %s', ports)
        for port in ports:
            if hwid in port.hwid:
                logger.debug('Found filter wheel (%s) on %s', hwid, port.usb_description())
                self.port = port.device
        if not self.port:    
            logging.error('No filter wheel (HWID=%s) detected', hwid)
            raise ConnectionError(f'No filter wheel (HWID={hwid}) detected')

        self._ser = serial.Serial()
        self._ser.baudrate = baud
        self._ser.port = self.port
        self._ser.timeout = 1.5
        self.timeout_moving = 6
        self.pcount = pcount

    def connect(self):
        'Open FW102C serial port.'
        try:
            self._ser.open()
        except serial.SerialException as err:
            logger.exception(err)
            raise err from None

    def close(self):
        'Close FW102C serial port.'
        try:
            self._ser.close()
        except serial.SerialException as err:
            logger.exception(err)
            raise err from None

    def _send_command(self, cmd:str, timeout:float = 6) -> None:
        '''Encode and write command to filter wheel serial.
        Waits for ">" as response, times out after timeout s.'''
        cmd_b = (cmd + '\r').encode()
        logger.debug('Sending command: %s', cmd)
        self._ser.flush()
        self._ser.write(cmd_b)

        # Read query echo from device
        try:
            echo = self._ser.read_until(cmd_b).decode()
            logger.debug('Received echo: "%s".', echo.strip('\r'))
        except serial.SerialTimeoutException as err:
            logger.exception(err)
            raise err from None

        # Wait for command execution
        try:
            rsp = ''
            start_time = time()
            while '>' not in rsp:
                rsp += self._ser.read(1).decode()
                if time()-start_time > timeout:
                    raise serial.SerialTimeoutException("Response took to long.")
            logger.debug('Command "%s" executed.', cmd)
        except serial.SerialTimeoutException as err:
            logger.exception(err)
            raise err from None

    def _send_query(self, query:str) -> str:
        '''Encode and write query to filter wheel serial.'''
        query_b = (query + '\r').encode()
        logger.debug('Sending query "%s".', query)
        self._ser.flush()
        self._ser.write(query_b)

        # Read query echo from device
        try:
            echo = self._ser.read_until(query_b).decode()
            logger.debug('Received echo: "%s".', echo.strip('\r'))
        except serial.SerialTimeoutException as err:
            logger.exception(err)
            raise err from None

        # Get query response
        try:
            rsp = self._ser.read_until(b'\r').decode().strip('\r')
            logger.debug('Querried: "%s"; Response: "%s"', query, rsp)
        except serial.SerialTimeoutException as err:
            logger.exception(err)
            raise err from None
        return rsp

    def get_position(self) ->int:
        '''Returns current position as an integer.'''
        pos = self._send_query(self.CMDS['get_pos'])
        return int(pos)

    def set_position(self, newpos:int, respect_bound:bool = False) ->None:
        '''Move the filter wheel to position `newpos`.
        `respect_bound` hinders the wheel to move directly from first to last position.'''
        if newpos not in range(1, self.pcount+1):
            logger.error('Position "%s" out of range.', newpos)
        if respect_bound:
            cur_pos = self.get_position()
            if abs(newpos-cur_pos) >= self.pcount//2:
                w_pos1= (2*cur_pos+newpos)//3
                w_pos2= (cur_pos+2*newpos)//3
                self._send_command(self.CMDS['set_pos'] % w_pos1)
                self._send_command(self.CMDS['set_pos'] % w_pos2)
                self._send_command(self.CMDS['set_pos'] % newpos)
            else:
                self._send_command(self.CMDS['set_pos'] % newpos)
        else:
            self._send_command(self.CMDS['set_pos'] % newpos)

    def get_id(self) ->str: #a bit useless since contained in get_settings()
        '''Displays the Thorlabs device ID and firmware version.'''
        rsp = self._send_query(self.CMDS['get_id'])
        return rsp

    def set_speed(self, speed_mode:bool) ->None:
        '''Sets the speed mode:
            0: low speed,
            1: high speed.'''
        logger.info(f'Set speed mode to {speed_mode}.')
        self._send_command(self.CMDS['set_speed'] % speed_mode)

    def set_sensor_mode(self, sensor_mode:bool) ->None:
        '''Sets the sensor mode:
            0: Sensor active when device not idle,
            1: Sensor always active (More stray light!).'''
        logger.info(f'Set sensor mode to {sensor_mode}.')
        self._send_command(self.CMDS['set_sens'] % sensor_mode)

    def set_baud_rate(self, baud_mode:bool) ->None:
        '''Sets the baud rate of the serial port:
            0: 9600,
            1: 115200.'''
        logger.info(f'Set baud rate mode to {baud_mode}.')
        self._send_command(self.CMDS['set_baud'] % baud_mode)

    def set_trigger_mode(self, trigger_mode:bool) ->None:
        '''Sets the trigger mode of the device:
            0: external trigger in input mode,
            1: external trigger in output mode.'''
        logger.info(f'Set trigger mode to {trigger_mode}.')
        self._send_command(self.CMDS['set_trigger'] % trigger_mode)

    def set_pcount(self, new_pcount:int) ->None:
        '''Sets the wheel to 6 or 12 available positions.
        Are you sure you want to change this?'''
        logger.info(f'Set position count to {new_pcount}.')
        self._send_command(self.CMDS['set_pcount'] % new_pcount)

    def save_defaults(self) ->None:
        '''Sets the current settings as default.'''
        logger.info('Saved current settings as default.')
        self._send_command(self.CMDS['save'])

    def get_settings(self): #typing for output missing
        '''Displays the current settings of the filter wheel:
        
            ID: Thorlabs ID & firmware version,
            PositionCount: Number of filter slots,
            SpeedMode:
                0: low speed,
                1: high speed,
            SensorMode:
                0: sensor deactivates when device idle,
                1: sensor always active,
            BaudMode:
                0: 9600,
                1: 115200,
            TriggerMode:
                0: input mode,
                1: output mode.
        '''
        settings = {
            'ID': self._send_query(self.CMDS['get_id']),
            'PositionCount': int(self._send_query(self.CMDS['get_pcount'])),
            'SpeedMode': int(self._send_query(self.CMDS['get_speed'])),
            'SensorMode': int(self._send_query(self.CMDS['get_sens'])),
            'BaudMode': int(self._send_query(self.CMDS['get_baud'])),
            'TriggerMode': int(self._send_query(self.CMDS['get_trigger']))
        }
        return settings


def main():
    'Test'
    filterwheel = FW102C('TP03014244')
    try:
        filterwheel.connect()
        logger.debug("Settings: %s", filterwheel.get_settings())
        logger.debug("Position: %s", filterwheel.get_position())
        logger.debug('Setting position: 1')
        filterwheel.set_position(1)
        logger.debug("Position: %s", filterwheel.get_position())
        logger.debug('Setting position: 6. Respect bounds.')
        filterwheel.set_position(6, respect_bound=True)
        logger.debug("Position: %s", filterwheel.get_position())
    except Exception as err:
        logger.exception(err)
        raise err from None
    finally:
        filterwheel.close()

if __name__ == '__main__':
    from logging.handlers import TimedRotatingFileHandler
    logger.setLevel(logging.DEBUG)
    logfile_handler = TimedRotatingFileHandler('filterwheel_test.log',
                                       'midnight',
                                       backupCount=7)
    logfile_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    )
    logfile_handler.setLevel(logging.DEBUG)
    logger.addHandler(logfile_handler)

    main()
