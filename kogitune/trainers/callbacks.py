from typing import Union
import time
import transformers
from ..adhoc_args import AdhocArguments, format_unit, verbose_print

def parse_time_as_second(time:str):
    if isinstance(time, int):
        return time
    hms = map(int, time.split(':'))
    if len(hms) == 3:
        return hms[0] * 3600 + hms[1] * 60 + hms[2]
    return hms[0] * 3600

class TimeoutStoppingCallback(transformers.TrainerCallback):

    def __init__(self, max_time: Union[int,str], **kwargs):
        self.start_time = time.time()
        self.save_count = 0
        self.step_count = 0
        with AdhocArguments.from_main(max_time=max_time, **kwargs) as aargs:
            self.max_time = aargs['max_time']
            self.safe_save = aargs['safe_save|=false']
            self.safety_time = aargs['safety_time|=300']
            if self.max_time is not None:
                verbose_print(f'タイムアウト時間 {format_unit(self.max_time, scale=60)} max_time={self.max_time} 設定したよ！')
                self.estimated_end_time = self.start_time + parse_time_as_second(self.max_time) -self.safety_time

    def on_save(self, args, state, control, **kwargs):
        current_time = time.time()
        remaining = self.estimated_end_time - current_time        
        self.save_count += 1
        if self.safe_save:
            interval = (current_time - self.start_time) / self.save_count
            if interval *  1.1 > remaining:
                verbose_print(f'残り時間 {format_unit(remaining, scale=60)} 必要な時間 {format_unit(interval, scale=60)} そろそろ時間だから終了するよ！')
                control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        remaining = self.estimated_end_time - current_time
        self.step_count += 1
        if self.max_time is not None:
            interval = (current_time - self.start_time) / self.save_count
            if remaining < (interval * 2):
                verbose_print(f'残り時間 {format_unit(remaining, scale=60)} が少ないから緊急停止するよ')
                control.should_save = True
                control.should_training_stop = True

