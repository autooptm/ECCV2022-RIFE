import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--compat', dest='compat', action='store_true',
                    help='run the original path exactly')
parser.add_argument('--bit-exact', dest='bit_exact', action='store_true',
                    help='keep only the changes that cannot change a pixel')
parser.add_argument('--no-opt-refine', dest='no_opt_refine', action='store_true',
                    help='disable the refine-tail optimization; '
                         '--bit-exact implies this')
args = parser.parse_args()
if args.compat:
    os.environ['RIFE_COMPAT'] = '1'
_FAST = not args.compat
_BITEXACT = args.bit_exact and _FAST
if _BITEXACT:
    os.environ['RIFE_NO_OPT_2'] = '1'
_OPT_REFINE = _FAST and not _BITEXACT and not args.no_opt_refine
if _FAST and not _BITEXACT:
    os.environ.setdefault('OPENCV_FFMPEG_WRITER_OPTIONS',
                          'threads;%s' % os.environ.get('RIFE_OPT_1', '8'))
assert (not args.video is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

_src_bgr = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if(args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded ArXiv-RIFE model")
model.eval()
model.device()

if _OPT_REFINE:
    import model.refine as _refine

    def _opt_wrap(orig):
        def f(self, *a, **k):
            with torch.autocast('cuda', dtype=torch.float16):
                out = orig(self, *a, **k)
            if torch.is_tensor(out):
                return out.float()
            if isinstance(out, (list, tuple)):
                return type(out)(o.float() if torch.is_tensor(o) else o for o in out)
            return out
        return f
    _refine.Contextnet.forward = _opt_wrap(_refine.Contextnet.forward)
    _refine.Unet.forward = _opt_wrap(_refine.Unet.forward)

if _FAST and torch.cuda.is_available():
    class _StepFlownet(torch.nn.Module):
        def __init__(self, m, warmup=3):
            super().__init__()
            self.m, self.warmup, self.cache = m, warmup, {}

        def forward(self, i0, i1, sc):
            key = (i0.dtype, i0.shape[2], i0.shape[3])
            if key not in self.cache:
                si0, si1 = i0.clone(), i1.clone()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(self.warmup):
                        self.m.inference(si0, si1, sc)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    so = self.m.inference(si0, si1, sc)
                self.cache[key] = (si0, si1, so, g)
            si0, si1, so, g = self.cache[key]
            si0.copy_(i0)
            si1.copy_(i1)
            g.replay()
            return so.clone()
    _infer = _StepFlownet(model)
else:
    _infer = lambda i0, i1, sc: model.inference(i0, i1, sc)

if not args.video is None:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * (2 ** args.exp)
    else:
        fpsNotAssigned = False
    videogen = None
    if _FAST and not _BITEXACT:
        _cap = cv2.VideoCapture(args.video)
        if _cap.isOpened():
            _src_bgr = True

            def _cv2_frames(cap):
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    yield fr
            videogen = _cv2_frames(_cap)
        else:
            _cap.release()
    if videogen is None:
        videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]
h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** args.exp), int(np.round(args.fps)), args.ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt),
                        item if _FAST else item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item if _FAST else item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
             if not user_args.img is None:
                  frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
             if user_args.montage:
                  frame = frame[:, left: left + w]
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):
    global model
    middle = _infer(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n//2)
    second_half = make_inference(middle, I1, n=n//2)
    if n%2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

def _load_frame(frame):
    if _FAST:
        t = torch.from_numpy(frame).to(device, non_blocking=True)
        if _src_bgr:
            t = t[:, :, [2, 1, 0]]
        t = (t.permute(2, 0, 1).unsqueeze(0).float() / 255.).contiguous()
    else:
        t = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(
            device, non_blocking=True).unsqueeze(0).float() / 255.
    return pad_image(t)

def _store_frame(t):
    if _FAST:
        return t.byte().flip(0).permute(1, 2, 0)[:h, :w].contiguous().cpu().numpy()
    return t.byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

def to_writer(src_frame):
    if _src_bgr == _FAST:
        return src_frame
    return np.ascontiguousarray(src_frame[:, :, ::-1])

if args.montage:
    left = w // 4
    w = w // 2
tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = _load_frame(lastframe)
I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
temp = None # save lastframe when processing static frame

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I0_small = I1_small if _FAST else F.interpolate(
        I0, (32, 32), mode='bilinear', align_corners=False)
    I1 = _load_frame(frame)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if ssim > 0.996:        
        frame = read_buffer.get() # read a new frame
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = _load_frame(frame)
        I1 = _infer(I0, I1, args.scale)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        if _src_bgr:
            frame = np.ascontiguousarray(frame[:, :, ::-1])
    
    if ssim < 0.2:
        output = []
        for i in range((2 ** args.exp) - 1):
            output.append(I0)
    else:
        output = make_inference(I0, I1, 2**args.exp-1) if args.exp else []

    if args.montage:
        lf = to_writer(lastframe)
        write_buffer.put(np.concatenate((lf, lf), 1))
        for mid in output:
            write_buffer.put(np.concatenate((lf, _store_frame(mid[0] * 255.)), 1))
    else:
        write_buffer.put(to_writer(lastframe))
        for mid in output:
            write_buffer.put(_store_frame(mid[0] * 255.))
    pbar.update(1)
    lastframe = frame
    if break_flag:
        break

if args.montage:
    _lf = to_writer(lastframe)
    write_buffer.put(np.concatenate((_lf, _lf), 1))
else:
    write_buffer.put(to_writer(lastframe))

write_buffer.put(None)

import time
while(not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.release()

# move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.video is None:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)
