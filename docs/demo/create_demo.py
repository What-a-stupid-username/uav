"""Generate the explicitly authorized public demo; no manual media inspection."""
import argparse,subprocess,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');args=p.parse_args()
w,h,fps,seconds=640,360,24,12
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',19)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',16)
decode=subprocess.Popen(['ffmpeg','-v','error','-ss','60','-i',args.source,'-t',str(seconds),'-vf',f'fps={fps},scale={w}:{h}','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
encode=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',args.output],stdin=subprocess.PIPE)
try:
 for i in range(fps*seconds):
  data=decode.stdout.read(w*h*3)
  if len(data)!=w*h*3:raise RuntimeError('incomplete background frame')
  image=Image.frombytes('RGB',(w,h),data).convert('RGBA');layer=Image.new('RGBA',(w,h));draw=ImageDraw.Draw(layer);t=i/fps
  draw.text((w-142,18),'UAV STUDIO',font=small,fill=(255,255,255,185),stroke_width=1,stroke_fill=(0,0,0,100))
  x=210+120*math.sin(t*.55);y=235+35*math.sin(t*.8)
  draw.text((x,y),'watermark',font=font,fill=(255,255,255,175),stroke_width=1,stroke_fill=(0,0,0,80))
  phase=int(t//2)
  if t%2<1.25:
   pos=((35,48),(350,88),(200,145))[phase%3]
   draw.text(pos,'PREVIEW',font=font,fill=(255,255,255,180),stroke_width=1,stroke_fill=(0,0,0,80))
  logo=Image.new('RGBA',(112,84));ld=ImageDraw.Draw(logo)
  ld.polygon([(17,39),(29,17),(43,27),(37,44),(24,51)],outline=(255,255,255,235),width=3)
  ld.line([(23,39),(30,27),(36,32),(31,43)],fill=(255,255,255,235),width=3)
  ld.text((46,23),'UAV',font=font,fill=(255,255,255,235),stroke_width=1,stroke_fill=(0,0,0,130))
  ld.text((24,52),'STUDIO',font=small,fill=(255,255,255,235),stroke_width=1,stroke_fill=(0,0,0,130))
  logo=logo.rotate(-t*12,resample=Image.Resampling.BICUBIC,expand=True)
  layer.alpha_composite(logo,(80-logo.width//2,292-logo.height//2))
  encode.stdin.write(Image.alpha_composite(image,layer).convert('RGB').tobytes())
finally:
 decode.stdout.close();encode.stdin.close()
 dc=decode.wait();ec=encode.wait()
 if dc or ec:raise RuntimeError(f'encoder/decoder failed: {dc}/{ec}')
print('Generated demo: 288 frames, 640x360, 24fps; four watermark behaviors.')
