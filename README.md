# IoT Servo Tracker

서보모터 기반 팬-틸트 객체 추적 프로젝트입니다. PDF 명세의 핵심 요구사항을 기준으로 라즈베리파이 엣지 제어, RTX 서버 비전 추론, Streamlit 웹 명령 화면, MQTT/ZMQ 통신, 센서 검증, 능동탐색, 안전대기 복구 흐름을 실행 가능한 Python 런타임으로 구성했습니다.

## 핵심 기능

- Streamlit 웹 화면에서 자연어 추적 대상 입력, 추적 시작, 중지, 재검출, 중립각 복귀 명령 생성
- MQTT 명령/상태 패킷 구조와 중복 명령 방지용 `cmd_id` 지원
- ZMQ 영상/추론 패킷 송수신 런타임
- 라즈베리파이 엣지 상태머신: `IDLE`, `SCAN`, `DELAY_COMPENSATION`, `TRACKING`, `SAFE_HOLD`, `LIMITED_RESCAN`, `CENTERING`, `ERROR`
- `Kinit` 연속 검출 기반 능동탐색 확정, 탐색 실패 시 중립각 복귀
- 명령 고유번호 재실행 방지, stale 추론 결과 폐기, thread-safe 런타임 상태 보호
- 픽셀 오차 기반 팬/틸트 PD 제어, deadband, 최대 각속도, 최대 각가속도, PWM 펄스폭 매핑
- ToF/초음파/리밋스위치 기반 단순 임계값 검증과 안전대기 전환
- 안전대기 soft-stop, 동일 대상 재검출, 제한재탐색, timeout 중립각 복귀
- 순환버퍼 기반 지연 보정용 프레임/검출 이력 관리
- Hugging Face에서 내려받는 WeDetect-Ref + WeDetect-Uni 로컬 추론 경계와 Ultralytics YOLO/ByteTrack 계열 production pipeline 경계
- YOLO 추적 중 대상이 사라지면 빈 결과를 안전대기 조건으로 전달하고, 연속 상실 시 WeDetect 재검출로 재잠금
- YOLO가 높은 confidence로 유사 물체를 잡거나 큰 배경/사물 bbox로 흡수되는 경우를 중심 이동량, 면적 증가율, 화면 점유율, 종횡비, ID 변경 시 IoU로 차단하되, 짧은 ID 변경은 위치/면적이 안정적이면 같은 대상으로 유지
- 실제 하드웨어 없이 동작 확인 가능한 시뮬레이션 드라이버와 단위 테스트

## 폴더 구조

```text
.
├── config/                     # 설정 예시
├── docs/                       # 아키텍처와 하드웨어 배선 문서
├── scripts/                    # 로컬 시뮬레이션 스크립트
├── src/iot_servo_tracker/
│   ├── common/                 # 설정, 패킷, 시간 유틸
│   ├── comms/                  # MQTT/ZMQ 통신 어댑터
│   ├── control/                # 상태머신, PD 제어, 센서 검증, 서보 드라이버
│   ├── edge/                   # 라즈베리파이 엣지 런타임
│   ├── server/                 # 추론 서버 런타임과 비전 파이프라인 경계
│   ├── sim/                    # 라즈베리파이 없는 오프라인 시뮬레이션
│   └── web/                    # Streamlit 웹 제어 화면
└── tests/                      # 표준 unittest 기반 테스트
```

## 빠른 실행

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
python3 scripts/simulate_control_loop.py --scenario normal --steps 80
python3 scripts/simulate_control_loop.py --scenario lost --steps 70
python3 scripts/simulate_control_loop.py --scenario retarget --steps 80
python3 scripts/simulate_control_loop.py --scenario sensor --steps 70
python3 scripts/simulate_full_stack.py --scenario retarget --steps 80
python3 scripts/simulate_full_stack.py --production --webcam --camera-index 0 --query "person"
python3 scripts/validate_live_webcam_stack.py --camera-index 0 --frames 60 --query "person" --save-last-frame /tmp/iot-live-stack.jpg
```

`scripts/simulate_control_loop.py`는 Raspberry Pi, PCA9685, 거리센서, 카메라, MQTT/ZMQ 없이 `EdgeRuntime` 상태머신과 서보 제어 흐름을 로컬에서 재현합니다. `lost`는 bbox 상실 후 안전대기 진입, `retarget`은 상실 상태에서 새 TRACK 명령으로 다른 물체를 다시 스캔하는 흐름, `sensor`는 초음파 거리 급락으로 안전대기 진입을 확인합니다. 자동 분석이 필요하면 `--jsonl`을 붙여 프레임별 이벤트를 JSON Lines로 출력할 수 있습니다.

`scripts/simulate_full_stack.py`는 Streamlit 버튼 클릭에 해당하는 웹 명령부터 in-memory MQTT, 엣지 런타임, in-memory ZMQ multipart 프레임 전달, 비전 서버 처리, MQTT 상태 수신, 웹 표시 상태까지 한 프로세스에서 재현합니다. 출력의 `WEB=DETECTING`/`WEB=TRACKING`은 웹 화면이 마지막 상태 패킷을 받았을 때 보여줄 상태입니다. 기본은 가상 웹캠 프레임을 사용하고, 로컬 PC 웹캠 프레임을 통신 경로에 태우려면 `--webcam --camera-index 0`을 붙입니다.

실제 WeDetect + YOLO 추론까지 같은 시뮬레이션 루프에서 확인하려면 `--production --webcam`을 함께 사용합니다. 이 모드는 `server` 선택 의존성, WeDetect repo 또는 adapter 설정, CUDA/YOLO 모델이 준비되어 있어야 하며, 실제 웹캠 JPEG 프레임이 WeDetect 초기 잠금/재검출과 YOLO 지속 추적으로 전달됩니다. WeDetect repo 위치를 직접 넘길 때는 `--wedetect-repo /path/to/WeDetect`를 사용합니다.

웹에서 실제 웹캠 비전만 빠르게 검증하려면 별도 Streamlit 검증 화면을 실행합니다. 이 화면은 웹에서 TRACK 명령 패킷을 만들고, Streamlit이 실행 중인 장비의 웹캠 프레임을 자동 캡처한 뒤 WeDetect 초기 잠금과 YOLO 지속 추적 결과를 bbox가 그려진 이미지와 테이블로 보여줍니다.

```bash
streamlit run src/iot_servo_tracker/web/vision_validation_app.py
```

라즈베리파이 없이 실제 웹캠 영상과 정의된 통신 흐름까지 함께 검증하려면 라이브 스택 검증 화면을 실행합니다. 이 화면은 웹에서 입력한 대상 문자열을 TRACK 명령으로 만들고, in-memory MQTT로 엣지 런타임에 전달한 뒤, PC 웹캠 JPEG 프레임을 in-memory ZMQ multipart 프레임으로 비전 런타임에 보냅니다. 비전 결과는 다시 ZMQ 결과 패킷으로 엣지에 돌아오고, ToF/초음파 센서값은 화면에서 지정한 근사값으로 넣어 상태머신을 진행합니다. 마지막으로 MQTT 상태 패킷이 웹으로 돌아오며, bbox와 `WEB`/`EDGE`/비전 source가 웹캠 영상 위에 표시됩니다.

```bash
streamlit run src/iot_servo_tracker/web/live_stack_app.py
```

같은 경로를 터미널에서 실제 웹캠으로 바로 검증하려면 아래 명령을 사용합니다. 이 명령은 fake camera를 쓰지 않고 `OpenCvCamera`로 `--camera-index`의 실제 PC 웹캠을 엽니다. 자동 단위 테스트의 fake camera는 카메라가 없는 환경에서도 통신/상태 루프를 검사하기 위한 테스트 대역일 뿐입니다.

```bash
python3 scripts/validate_live_webcam_stack.py --camera-index 0 --frames 60 --query "person" --save-last-frame /tmp/iot-live-stack.jpg
```

기본 모드는 모델 없이도 실제 웹캠 프레임과 통신/상태 흐름이 도는 `scripted` 비전입니다. RTX 서버 환경에서 WeDetect adapter, YOLO 모델, CUDA 의존성이 준비되어 있으면 화면에서 `Real WeDetect + YOLO`를 켜거나 CLI에 `--production`을 붙여 같은 웹캠/통신 루프를 실제 비전 모델로 검증합니다.

패키지 형태로 설치해서 실행하려면:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[web,edge,server,dev]"
iot-simulate
```

웹 화면은 선택 의존성을 설치한 뒤 실행합니다.

```bash
streamlit run src/iot_servo_tracker/web/app.py
streamlit run src/iot_servo_tracker/web/sim_app.py
streamlit run src/iot_servo_tracker/web/vision_validation_app.py
streamlit run src/iot_servo_tracker/web/live_stack_app.py
```

## 실제 프로세스 실행

MQTT broker와 ZMQ 통신 주소를 `config/settings.toml`에 맞춘 뒤 실행합니다. WeDetect는 네트워크 API 호출 없이 RTX 서버 프로세스 안에서 실행합니다. WeDetect-Ref와 WeDetect-Uni 체크포인트는 Hugging Face Hub에서 내려받고, 실제 추론은 프로젝트 로컬 adapter 모듈이나 스크립트가 담당합니다.

WeDetect-Ref 설정 예시:

```toml
[server]
wedetect_ref_repo_id = "fushh7/WeDetect-Ref-2B"
wedetect_uni_repo_id = "fushh7/WeDetect"
wedetect_uni_filename = "wedetect_base_uni.pth"
wedetect_cache_dir = ""
wedetect_ref_model_dir = ""
wedetect_uni_checkpoint = ""
wedetect_ref_module = "my_wedetect_ref_runtime:detect"
wedetect_ref_script = ""
wedetect_device = "cuda:0"
yolo_lost_frames = 30
yolo_suspect_frames = 5
yolo_max_center_jump_px = 320.0
yolo_max_area_growth_ratio = 16.0
yolo_max_frame_area_ratio = 0.85
yolo_max_aspect_ratio_change = 8.0
yolo_min_iou_on_id_change = 0.0
```

`yolo_lost_frames = 30`는 12 FPS 기준 약 2.5초 분량입니다. 짧은 YOLO/ByteTrack 흔들림에는 빈 결과를 반환하며 기다리고, 연속 상실이 길어질 때만 무거운 WeDetect-Ref 재검출로 넘어갑니다.

`wedetect_ref_module` callable은 `frame_bytes`, `query`, `ts_req`, `wedetect_ref_model_dir`, `wedetect_uni_checkpoint`, `device` 키워드 인자를 받고, `{"bbox": [x1, y1, x2, y2], "confidence": 0.9, "track_id": 1}` 형태의 dict 또는 `TrackingResult`를 반환하면 됩니다. `wedetect_ref_model_dir`와 `wedetect_uni_checkpoint`를 비워두면 `huggingface-hub`가 설정된 repo에서 자동으로 다운로드합니다. 독립 스크립트를 쓰는 경우 `wedetect_ref_script`에 경로를 넣으면 임시 이미지 파일, WeDetect-Ref 경로, WeDetect-Uni 체크포인트, query가 인자로 전달됩니다.

RTX 서버:

```bash
iot-server --config config/settings.toml --serve --production
```

라즈베리파이 엣지:

```bash
iot-edge --config config/settings.toml --run
```

PCA9685 서보와 실제 거리센서를 사용할 때:

```bash
iot-edge --config config/settings.toml --run --hardware-servo --hardware-sensors
```

하드웨어 없이 통신 루프만 확인할 때:

```bash
iot-edge --config config/settings.toml --run --simulated-camera
```

## 실제 장비 연결 시 다음 작업

1. `config/settings.example.toml`을 `config/settings.toml`로 복사하고 MQTT/ZMQ 주소와 서보 한계를 환경에 맞춥니다.
2. `wedetect_ref_module` 또는 `wedetect_ref_script`, Hugging Face repo 설정, `yolo_model`, `tracker`를 RTX 서버 환경에 맞춥니다.
3. 라즈베리파이에서는 `iot-edge --run --hardware-servo --hardware-sensors`로 PCA9685 서보와 ToF/초음파/리밋스위치를 사용합니다.
4. 하드웨어 핀 번호가 다르면 `RaspberryPiSensorReader` 생성 인자를 장비 배선에 맞춥니다.
5. Kp/Kd, `omega_max`, `alpha_max`, `Tpix`, `Ttof`, `Tultra`, `Nh`를 실제 카메라/서보/전원 환경에서 보수적으로 튜닝합니다.
6. 유사 물체 오검출이 잦으면 `yolo_max_center_jump_px`와 `yolo_min_iou_on_id_change`를 낮추고, 큰 배경 bbox로 흡수되면 `yolo_max_area_growth_ratio`와 `yolo_max_frame_area_ratio`를 낮춥니다.
