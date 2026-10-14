import json
import math
import shutil
import subprocess
from pathlib import Path

def run(args, cancel=None):
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:
        while True:
            try:
                out, err = p.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if cancel and cancel.is_set():
                    p.kill()
                    p.communicate()
                    raise InterruptedError('Cancelled during media processing.')
        if p.returncode:
            raise RuntimeError(err[-4000:] or f'Command failed: {args[0]}')
    return out

def probe(path, count=False):
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        raise RuntimeError('FFprobe is missing; run Pinokio Install.')
    args = [ffprobe,'-v','error'] + (['-count_frames'] if count else [])
    data = json.loads(run(args+['-show_streams','-show_format','-of','json',str(path)]))
    video = next((s for s in data['streams'] if s['codec_type']=='video'), None)
    if not video:
        raise ValueError('File has no video stream.')
    rate = video.get('avg_frame_rate','0/1').split('/')
    fps = float(rate[0])/float(rate[1]) if float(rate[1]) else 0
    return {'width':int(video['width']),'height':int(video['height']),'fps':fps,'duration':float(video.get('duration') or data['format'].get('duration',0)), 'frames':int(video.get('nb_read_frames') or video.get('nb_frames') or 0),'audio':any(s['codec_type']=='audio' for s in data['streams'])}

def frame_count(seconds, fps, stride):
    # Never round upward into unavailable source frames.
    return max(stride+1, int((seconds*fps-1)//stride)*stride+1)

def dimensions(width, height, longest, multiple):
    scale = min(longest/max(width,height),1)
    return (max(multiple,round(width*scale/multiple)*multiple), max(multiple,round(height*scale/multiple)*multiple))

def boundaries(path, duration, cancel=None):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    cuts = [0.0]
    previous = None
    try:
        for i in range(int(duration*2)):
            if cancel and cancel.is_set():
                raise InterruptedError('Cancelled during scene analysis.')
            time = i/2
            cap.set(cv2.CAP_PROP_POS_MSEC,time*1000)
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(64,36)).astype('float32')/255
            if previous is not None:
                # A conservative hard-cut heuristic. Not a semantic AI claim.
                distance = float(np.mean(np.abs(small-previous)))
                if distance > .32 and time-cuts[-1] > 1:
                    cuts.append(time)
            previous = small
    finally:
        cap.release()
    return cuts + [duration]

def windows(cuts, duration):
    result = []
    for start, end in zip(cuts,cuts[1:]):
        # No overlapping clips, duplicated tail padding, or crossing detected cuts.
        for i in range(int((end-start+1e-6)//duration)):
            result.append(round(start+i*duration,6))
    return result

def split(source, target, start, frames, fps, width, height, cancel=None):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpeg missing; run Pinokio Install.')
    # Fractional seeks can leave the first timestamp between output ticks.
    # Anchor CFR at zero and allow one output tick of muxer margin, while
    # trimming audio separately to the requested video duration.
    filters = f'setpts=PTS-STARTPTS,fps={fps}:start_time=0,scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1'
    run([ffmpeg,'-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(source),'-map','0:v:0','-map','0:a:0?', '-vf',filters,'-af',f'atrim=duration={frames/fps},asetpts=PTS-STARTPTS','-frames:v',str(frames),'-t',str((frames+1)/fps),'-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-ar','48000','-movflags','+faststart',str(target)],cancel)
    meta = probe(target,True)
    if meta['frames'] != frames or meta['width'] != width or meta['height'] != height or abs(meta['fps']-fps)>.01:
        raise RuntimeError(f'Clip failed tensor-shape audit: {target.name}: {meta}')
    return meta

def storyboard(path, count=8):
    import cv2
    from PIL import Image
    meta = probe(path)
    cap = cv2.VideoCapture(str(path))
    images = []
    try:
        for i in range(count):
            t = (meta['duration'] - 1/meta['fps']) * i/max(count-1,1)
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
            ok, bgr = cap.read()
            if ok:
                im = Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                im.thumbnail((512,512))
                images.append((round(t,3),im))
    finally:
        cap.release()
    return images

def extract_audio(video, dest, cancel=None):
    run([shutil.which('ffmpeg'),'-v','error','-y','-i',str(video),'-vn','-ac','1','-ar','16000',str(dest)],cancel)

def audio_events(path):
    import soundfile as sf
    import numpy as np
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    width = int(sr*.02)
    chunks = len(y)//width
    if chunks == 0:
        return {'rms':0.0,'transients_seconds':[],'method':'RMS onset heuristic, not semantic recognition'}
    rms = np.sqrt((y[:chunks*width].reshape(chunks,width)**2).mean(axis=1))
    delta = np.diff(rms,prepend=rms[0])
    threshold = max(float(np.percentile(delta,95)),.03)
    events = []
    for i in np.where(delta>threshold)[0]:
        time = round(i*.02,3)
        if not events or time-events[-1]>.15:
            events.append(time)
    return {'rms':round(float(np.mean(rms)),5),'transients_seconds':events,'method':'RMS onset heuristic, not semantic recognition'}
