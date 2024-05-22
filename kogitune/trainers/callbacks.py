from typing import Union
import time
import transformers
import kogitune.adhocs as adhoc

DAY_SEC = 24 * 3600
HOUR_SEC = 3600

def parse_time_as_second(time:str):
    if isinstance(time, int):
        return time
    if time.endswith('d'):
        return int(float(time[:-1]) * DAY_SEC)
    if time.endswith('h') or time.endswith('H'):
        return int(float(time[:-1]) * HOUR_SEC)
    if time.endswith('m') or time.endswith('M'):
        return int(float(time[:-1]) * 60)
    hms = list(map(int, time.split(':')))
    if len(hms) == 4:  # 1:23:00:15
        return hms[0] * DAY_SEC + hms[1] * HOUR_SEC + hms[2] * 60 + hms[3]
    if len(hms) == 3:  # 1:38:00
        return hms[0] * DAY_SEC + hms[1] * HOUR_SEC + hms[2] * 60
    elif len(hms) == 2: # 12:00
        return hms[0] * HOUR_SEC + hms[1] * 60
    return hms[0] * HOUR_SEC

class TimeoutStoppingCallback(transformers.TrainerCallback):

    def __init__(self, max_time: Union[int,str], safe_time=300, safe_save=False):
        self.start_time = time.time()
        self.save_count = 0
        self.step_count = 0
        self.max_time = parse_time_as_second(max_time)
        self.safe_time = parse_time_as_second(safe_time)
        self.safe_save = safe_save
        adhoc.notice(f'タイムアウト時間 {adhoc.format_unit(self.max_time, scale=60)} 設定したよ！', max_time=max_time)
        self.estimated_end_time = self.start_time + self.max_time -self.safe_time

    def on_save(self, args, state, control, **kwargs):
        current_time = time.time()
        remaining = self.estimated_end_time - current_time        
        self.save_count += 1
        if self.safe_save:
            interval = (current_time - self.start_time) / self.save_count
            if interval *  1.1 > remaining:
                adhoc.notice(f'残り時間 {adhoc.format_unit(remaining, scale=60)} 必要な時間 {adhoc.format_unit(interval, scale=60)} そろそろ時間だから終了するよ！')
                control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        remaining = self.estimated_end_time - current_time
        self.step_count += 1
        interval = (current_time - self.start_time) / self.step_count
        if remaining < (interval * 2):
            adhoc.notice(f'残り時間 {adhoc.format_unit(remaining, scale=60)} が少ないから緊急停止するよ', color='red')
            control.should_save = True
            control.should_training_stop = True

def load_callbacks(**kwargs):
    with adhoc.from_kwargs(**kwargs) as aargs:
        callbacks = []
        if 'max_time' in aargs or 'sge_walltime_sec' in aargs:
            max_time = aargs['max_time|sge_walltime_sec']
            safe_time = aargs['safe_time|=300']
            callbacks.append(TimeoutStoppingCallback(max_time=max_time, safe_time=safe_time))
        return callbacks
