# Gix-Detailer

[AUTOMATIC1111님의 WEBUI 확장입니다.](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

이미지를 그릴 때, 각 단계별로 좁은 부분마다 세부표현을 수정해
최상의 퀄리티로 완성하기 위한 확장입니다.

- t2i, i2i 모두 지원.
- i2i 에서 Hires.Fix 지원
- 이미지를 각 타일로 나누어 상세표현
- 각 타일을 상세표현을 할 때, 자동으로 부분 프롬프트를 생성
- 상세표현 후, 얼굴을 추가로 상세표현

-------

[Extension for SD WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

When generating an image, refine the detail in each narrow part at each stage.
This is an extension to complete with the highest quality.

- Support Both t2i, i2i
- Hires. Fix on i2i
- Detailed representation of images divided into tiles
- Automatically generate partial prompts when detailing each tile
- After detail processing, face is further processed in detail.

## Usage
1. WEBUI 확장에서 설치 ( Extensions > Install from URL )
2. t2i 또는 i2i 의 Script 항목에서 Gix Detailer 선택
3. Load 버튼 클릭
4. 이미지 생성

-------

1. Install at WEBUI ( Extensions > Install from URL )
2. Select 'Gix Detailer'  from Script Items of t2i or i2i screen
3. Click 'Load' Button
4. Generate image

## inspired
[MultiDiffusion](https://multidiffusion.github.io/)

## implements
[WD Tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)

[Ultimate-upscale](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111)

[Detection Detailer](https://github.com/dustysys/ddetailer)
